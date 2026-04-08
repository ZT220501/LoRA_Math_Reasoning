import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk
import yaml
from tqdm import tqdm
import os
import json
from typing import Optional

torch.manual_seed(0)

def extract_gt_answer(text: str) -> Optional[str]:
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    
    # Fallback — flag it so you know it wasn't a clean extraction
    numbers = re.findall(r"-?[\d,]+", text)
    if numbers:
        # print("⚠️  Warning: used fallback extraction, no #### found")
        return numbers[-1].replace(",", "")
    
    return None

def normalize_numeric_string(s: str) -> str:
    s = s.strip()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace(" ", "")

    # remove surrounding LaTeX braces if they remain
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()

    # simple normalization: 3.0 -> 3
    try:
        x = float(s)
        if x.is_integer():
            return str(int(x))
        return str(x)
    except ValueError:
        return s

def extract_boxed_content(text: str) -> Optional[str]:
    """
    Extract content inside the last \\boxed{...}, handling nested braces.
    """
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None

    # use the last boxed occurrence
    start = matches[-1].end()
    brace_level = 1
    i = start

    while i < len(text):
        if text[i] == "{":
            brace_level += 1
        elif text[i] == "}":
            brace_level -= 1
            if brace_level == 0:
                return text[start:i].strip()
        i += 1

    return None

def extract_final_answer(text: str) -> Optional[str]:
    # 1. Try \boxed{...}
    boxed = extract_boxed_content(text)
    if boxed is not None:
        return normalize_numeric_string(boxed)

    # 2. Try "Final Answer: 42"
    m = re.search(
        r"final\s+answer\s*[:：]\s*\$*\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE
    )
    if m:
        return normalize_numeric_string(m.group(1))

    # 3. Try GSM8K style #### 42
    m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return normalize_numeric_string(m.group(1))

    # 4. Try "the answer is 42"
    m = re.search(
        r"(?:the\s+answer\s+is|answer\s*[:：])\s*\$*\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE
    )
    if m:
        return normalize_numeric_string(m.group(1))

    # 5. Fallback: take the last standalone number
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        return normalize_numeric_string(nums[-1])

    return None

# Define system prompt
SYSTEM_PROMPT = "You are a careful and rigorous mathematician. Solve problems step by step, and put your final answer within \\boxed{}. Do not output anything after the boxed answer."
# ── Format prompt for evaluation ──────────────────────────────────────
def format_prompt_evaluation(sample):
    """For SFT — includes question + full CoT answer."""
    chat = (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample["question"]},
    )
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prompt

# ── Inference for one model ──────────────────────────────────────
def evaluate_ft_model(model, tokenizer, test_data, sampling_budget=1, max_new_tokens=512, device="cuda", base_model=False, temperature=1.0, top_p=0.95, top_k=0):
    # IMPORTANT: Set the model to evaluation mode
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
        gt = extract_gt_answer(sample["answer"])

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
                do_sample=True,          # greedy for reproducibility
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=sampling_budget,
            )

        raw_answers = []
        answers = []
        for output in outputs:
            # Append the generated full answer
            generated = tokenizer.decode(output, skip_special_tokens=True)
            raw_answers.append(generated)
            # Append the extracted final answer
            predicted = extract_final_answer(generated)
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
    # test_data = test_data.select(range(1))

    # Load the base model config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    model_name = config['model']['name']
    model_dtype = config['model']['torch_dtype']
    max_new_tokens = config['inference']['max_new_tokens']
    temperature = config['inference']['temperature']
    top_p = config['inference']['top_p']
    top_k = config['inference']['top_k']
    sampling_budget = config['inference']['sampling_budget']

    print("Config: ", "model_name: ", model_name, "model_dtype: ", model_dtype, "sampling_budget: ", sampling_budget, "max_new_tokens: ", max_new_tokens, "temperature: ", temperature, "top_p: ", top_p, "top_k: ", top_k)

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
    
    config_timestamp = "config_2026_04_07_21_05_32"
    model_save_dir = f"training_output_gsm8k/{config_timestamp}/final_result"
    ft_model = PeftModel.from_pretrained(base_model, model_save_dir)
    # Evaluate the fine-tuned model
    eval_save_dir = f"eval_results_gsm8k/{model_name}_fine_tuned_sampling_budget_{sampling_budget}"
    os.makedirs(eval_save_dir, exist_ok=True)
    ft_acc, ft_correct, total, ft_result_json = evaluate_ft_model(ft_model, tokenizer, test_data, sampling_budget=sampling_budget, device=device, base_model=False, temperature=temperature, top_p=top_p, top_k=top_k)

    # Save the fine-tuned model result to a json file
    with open(os.path.join(eval_save_dir, f"fine_tuned_result_config_{config_timestamp}.json"), "w") as f:
        json.dump(ft_result_json, f)
    print(f"\nFine-tuned Model Accuracy: {ft_acc:.2f}% ({ft_correct}/{total})")