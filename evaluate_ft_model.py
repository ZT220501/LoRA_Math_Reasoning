import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk
import yaml
from tqdm import tqdm
import os
import json

torch.manual_seed(0)

def extract_answer(text):
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    
    # Fallback — flag it so you know it wasn't a clean extraction
    numbers = re.findall(r"-?[\d,]+", text)
    if numbers:
        # print("⚠️  Warning: used fallback extraction, no #### found")
        return numbers[-1].replace(",", "")
    
    return None

# Define system prompt
SYSTEM_PROMPT = "You are a helpful math expert. Solve the problem step by step, then give the final answer after '####'."
# ── Format prompt for evaluation ──────────────────────────────────────
def format_prompt_evaluation(sample):
    """For SFT — includes question + full CoT answer."""
    chat = (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample["question"]},
    )
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt

# ── Inference for one model ──────────────────────────────────────
def evaluate_model(model, tokenizer, test_data, sampling_budget=1, max_new_tokens=512, device="cuda", base_model=False):
    model.eval()
    correct = 0
    total = len(test_data)

    result_json = {
        "sampling_budget": sampling_budget,
        "max_new_tokens": max_new_tokens,
        "accuracy": 0,
        "correct": 0,
        "total": 0,
    }

    pbar = tqdm(
        enumerate(test_data),
        total=total,
        desc="Evaluating model on GSM8K test set",
    )    
    for i, sample in pbar:
        question = sample["question"]
        gt = extract_answer(sample["answer"])

        result_json[f"sample_{i}"] = {
            "question": question,
        }

        # Match the format DeepSeek-R1-Distill was trained on
        prompt = format_prompt_evaluation(sample)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy for reproducibility
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=sampling_budget,
            )

        raw_answers = []
        answers = []
        for output in outputs:
            generated = tokenizer.decode(output, skip_special_tokens=True)
            raw_answers.append(generated)
            if base_model:
                predicted = extract_answer_deepseek_r1(generated)
            else:
                predicted = extract_answer(generated)
            answers.append(predicted)

        result_json[f"sample_{i}"]["raw_answers"] = raw_answers
        result_json[f"sample_{i}"]["answers"] = answers

        if any(predicted and gt and predicted == gt for predicted in answers):
            correct += 1

        # Set the progress bar postfix
        pbar.set_postfix(
            acc=f"{100 * correct / (i + 1):.2f}%",
            n_correct=f"{correct}/{i + 1}",
        )

        # Progress log every 100 samples
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] Running accuracy: {correct/(i+1)*100:.2f}%")

    accuracy = correct / total * 100
    result_json["accuracy"] = accuracy
    result_json["correct"] = correct
    result_json["total"] = total
    return accuracy, correct, total, result_json


if __name__ == "__main__":
    # ── Load GSM8K test set ──────────────────────────────────────────
    data_save_dir = "dataset/gsm8k"
    ds = load_from_disk(data_save_dir)
    test_data = ds["test"]
    # test_data = test_data.select(range(5))

    # Load the base model config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    model_name = config['model']['name']
    model_dtype = config['model']['torch_dtype']

    sampling_budget = 1
    print("Sampling budget: ", sampling_budget)

    # ── Evaluate FINE-TUNED model ────────────────────────────────────
    print("\n=== Evaluating FINE-TUNED model ===")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Load the fine-tuned model and merge it into the base model
    tokenizer = AutoTokenizer.from_pretrained(model_name, dtype=model_dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=model_dtype,
        device_map="auto"
    )
    ft_model = PeftModel.from_pretrained(base_model, "training_output_gsm8k/config_initial/final_result")
    # Evaluate the fine-tuned model
    eval_save_dir = f"eval_results_gsm8k/{model_name}_fine_tuned_sampling_budget_{sampling_budget}"
    os.makedirs(eval_save_dir, exist_ok=True)
    ft_acc, ft_correct, total, ft_result_json = evaluate_model(ft_model, tokenizer, test_data, sampling_budget=sampling_budget, device=device, base_model=False)
    # Save the fine-tuned model result to a json file
    with open(os.path.join(eval_save_dir, "fine_tuned_result.json"), "w") as f:
        json.dump(ft_result_json, f)
    print(f"\nFine-tuned Model Accuracy: {ft_acc:.2f}% ({ft_correct}/{total})")