import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os

# load_dir = "training_output_gsm8k/config_initial/checkpoint-10000"
# load_dir = "training_output_gsm8k/config_2026_04_07_00_37_45/checkpoint-2000"
# load_dir = "training_output_gsm8k/config_2026_04_07_12_26_52/checkpoint-10000"
# load_dir = "training_output_gsm8k/config_2026_04_07_20_10_46/checkpoint-1500"
load_dir = "training_output_gsm8k/config_2026_04_07_21_05_32/checkpoint-1200"
with open(os.path.join(load_dir, "trainer_state.json"), "r") as f:
    state = json.load(f)

log_history = state["log_history"]
train_logs = [x for x in log_history if "loss" in x and "eval_loss" not in x]
eval_logs  = [x for x in log_history if "eval_loss" in x]

train_steps  = [x["step"] for x in train_logs]
train_losses = [x["loss"] for x in train_logs]
eval_steps   = [x["step"] for x in eval_logs]
eval_losses  = [x["eval_loss"] for x in eval_logs]

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "font.size": 9,
    # Math in plots: closer to LaTeX CM than default
    "mathtext.fontset": "cm",
})
ax.plot(
    train_steps,
    train_losses,
    color="#2563eb",
    linewidth=2.2,
    alpha=0.92,
    label="Train loss",
    zorder=2,
)
ax.plot(
    eval_steps,
    eval_losses,
    color="#dc2626",
    linewidth=2,
    marker="o",
    markersize=7,
    markeredgecolor="white",
    markeredgewidth=0.8,
    label="Eval loss",
    zorder=3,
)
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Training vs validation loss", fontsize=14, fontweight="600", pad=12)
ax.legend(frameon=True, fancybox=True, shadow=False, fontsize=11)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(
    "loss_curve.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05,
    metadata={"Title": "Training vs validation loss"},
)
plt.show()