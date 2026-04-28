import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).parent

# Load both runs
with open(base / "training_log_run1.json") as f:
    run1 = json.load(f)
with open(base / "training_log.json") as f:
    run2 = json.load(f)

RUNS = [
    ("First Run (30 ep)", run1, "#e74c3c", "#c0392b"),
    ("Latest Run (22 ep)", run2, "#2ecc71", "#27ae60"),
]

# --- Figure 1: Reward Comparison ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("LivePatch — Training Comparison: First Run vs Latest Run", fontsize=16, fontweight="bold")

# Panel 1: Mean Reward
ax = axes[0, 0]
for label, data, c1, c2 in RUNS:
    eps = [d["episode"] for d in data]
    rewards = [d["mean_reward"] for d in data]
    ax.plot(eps, rewards, color=c1, alpha=0.4, linewidth=1)
    w = max(3, len(eps) // 5)
    rolling = np.convolve(rewards, np.ones(w)/w, mode="valid")
    ax.plot(eps[w-1:], rolling, color=c2, linewidth=2.5, label=label)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
ax.set_title("Mean Reward")
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Best Score
ax = axes[0, 1]
for label, data, c1, c2 in RUNS:
    eps = [d["episode"] for d in data]
    scores = [d["best_score"] for d in data]
    ax.plot(eps, scores, color=c1, alpha=0.4, linewidth=1, marker="o", markersize=3)
    w = max(3, len(eps) // 5)
    rolling = np.convolve(scores, np.ones(w)/w, mode="valid")
    ax.plot(eps[w-1:], rolling, color=c2, linewidth=2.5, label=label)
ax.set_title("Best Episode Score")
ax.set_xlabel("Episode")
ax.set_ylabel("Score (0–1)")
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Fix Quality
ax = axes[0, 2]
for label, data, c1, c2 in RUNS:
    eps = [d["episode"] for d in data]
    fix = [d["best_fix_quality"] for d in data]
    ax.bar([e + (0.2 if "Latest" in label else -0.2) for e in eps],
           fix, width=0.4, color=c1, alpha=0.7, label=label)
ax.set_title("Fix Quality (EXPLAIN Cost Improvement)")
ax.set_xlabel("Episode")
ax.set_ylabel("Fix Quality")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 4: Uptime
ax = axes[1, 0]
for label, data, c1, c2 in RUNS:
    eps = [d["episode"] for d in data]
    uptime = [d["best_uptime"] for d in data]
    ax.plot(eps, uptime, color=c2, linewidth=2, marker="o", markersize=3, label=label)
ax.set_title("Traffic Uptime During Fix")
ax.set_xlabel("Episode")
ax.set_ylabel("Uptime (0–1)")
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 5: Safety
ax = axes[1, 1]
for label, data, c1, c2 in RUNS:
    eps = [d["episode"] for d in data]
    safety = [d["best_safety"] for d in data]
    ax.plot(eps, safety, color=c2, linewidth=2, marker="s", markersize=3, label=label)
ax.set_title("Safety (CONCURRENTLY usage)")
ax.set_xlabel("Episode")
ax.set_ylabel("Safety (0–1)")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 6: Loss
ax = axes[1, 2]
for label, data, c1, c2 in RUNS:
    eps = [d["episode"] for d in data]
    loss = [d["loss"] for d in data]
    ax.plot(eps, loss, color=c2, linewidth=2, marker="s", markersize=3, label=label)
ax.set_title("Training Loss")
ax.set_xlabel("Episode")
ax.set_ylabel("Loss")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(base / "training_curves.png", dpi=150, bbox_inches="tight")
print(f"Saved training_curves.png (First: {len(run1)} ep, Latest: {len(run2)} ep)")

# --- Figure 2: Summary bar chart ---
fig2, ax2 = plt.subplots(figsize=(10, 6))

metrics = ["Avg Score", "Avg Fix Quality", "Avg Uptime", "Avg Safety"]
run1_vals = [
    np.mean([d["best_score"] for d in run1]),
    np.mean([d["best_fix_quality"] for d in run1]),
    np.mean([d["best_uptime"] for d in run1]),
    np.mean([d["best_safety"] for d in run1]),
]
run2_vals = [
    np.mean([d["best_score"] for d in run2]),
    np.mean([d["best_fix_quality"] for d in run2]),
    np.mean([d["best_uptime"] for d in run2]),
    np.mean([d["best_safety"] for d in run2]),
]

x = np.arange(len(metrics))
w = 0.35
bars1 = ax2.bar(x - w/2, run1_vals, w, label="First Run", color="#e74c3c", alpha=0.8)
bars2 = ax2.bar(x + w/2, run2_vals, w, label="Latest Run", color="#2ecc71", alpha=0.8)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

ax2.set_title("LivePatch — First Run vs Latest Run Summary", fontsize=14, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.set_ylim(0, 1)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig2.savefig(base / "training_dashboard.png", dpi=150, bbox_inches="tight")
print("Saved training_dashboard.png")

# --- Standalone reward curves per run (like K8s SRE style) ---
STANDALONE = [
    ("training_reward_first.png", "First Run", run1),
    ("training_reward_latest.png", "Latest Run", run2),
]

for fname, run_label, data in STANDALONE:
    episodes = [d["episode"] for d in data]
    mean_rewards = [d["mean_reward"] for d in data]
    max_rewards = [d["max_reward"] for d in data]
    n = len(episodes)

    fig_s, ax_s = plt.subplots(figsize=(14, 7))

    # Per-episode line + dots
    ax_s.plot(episodes, mean_rewards, color="#9999ff", marker="o", markersize=5,
              linewidth=1, alpha=0.6, label="Per episode")

    # Rolling average
    window = max(3, n // 5)
    rolling = np.convolve(mean_rewards, np.ones(window)/window, mode="valid")
    ax_s.plot(episodes[window-1:], rolling, color="blue", linewidth=3,
              label=f"Rolling avg ({window})")

    # Trend line
    z = np.polyfit(episodes, mean_rewards, 1)
    trend = np.poly1d(z)
    direction = "\u2191" if z[0] > 0 else "\u2193"
    ax_s.plot(episodes, trend(episodes), color="red", linewidth=2, linestyle="--",
              label=f"Trend ({direction} {abs(z[0]):.3f}/ep)")

    # Zero reference
    ax_s.axhline(y=0, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    # Annotation box
    final_avg = np.mean(mean_rewards[-window:]) if n >= window else np.mean(mean_rewards)
    best_reward = max(max_rewards)
    ax_s.text(0.02, 0.05,
              f"Episodes: {n} | Final avg: {final_avg:.2f} | Best: {best_reward:.2f}",
              transform=ax_s.transAxes, fontsize=10,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="orange"))

    ax_s.set_xlabel("Episode", fontsize=13)
    ax_s.set_ylabel("Total Reward", fontsize=13)
    ax_s.set_title(f"LivePatch DBA Agent \u2014 GRPO Training Reward Curve ({run_label})", fontsize=15)
    ax_s.legend(loc="upper left", fontsize=11)
    ax_s.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_s.savefig(base / fname, dpi=150, bbox_inches="tight")
    plt.close(fig_s)
    print(f"Saved {fname}")

# --- Individual metric comparison charts ---
INDIVIDUAL = [
    ("reward", "Mean Reward", "mean_reward", "Total Reward", None),
    ("score", "Best Episode Score", "best_score", "Score (0\u20131)", (0, 1)),
    ("fix_quality", "Fix Quality (EXPLAIN Cost Improvement)", "best_fix_quality", "Fix Quality", (0, 1)),
    ("uptime", "Traffic Uptime During Fix", "best_uptime", "Uptime (0\u20131)", (0, 1)),
    ("safety", "Safety (CONCURRENTLY usage)", "best_safety", "Safety (0\u20131)", (0, 1.05)),
    ("loss", "Training Loss", "loss", "Loss", None),
]

for fname, title, key, ylabel, ylim in INDIVIDUAL:
    fig_i, ax_i = plt.subplots(figsize=(10, 5))
    for label, data, c1, c2 in RUNS:
        eps = [d["episode"] for d in data]
        vals = [d[key] for d in data]
        ax_i.plot(eps, vals, color=c1, alpha=0.4, linewidth=1, marker="o", markersize=3)
        w = max(3, len(eps) // 5)
        rolling = np.convolve(vals, np.ones(w)/w, mode="valid")
        ax_i.plot(eps[w-1:], rolling, color=c2, linewidth=2.5, label=label)
    ax_i.set_title(f"LivePatch \u2014 {title}", fontsize=14, fontweight="bold")
    ax_i.set_xlabel("Episode")
    ax_i.set_ylabel(ylabel)
    if ylim:
        ax_i.set_ylim(*ylim)
    ax_i.legend(fontsize=10)
    ax_i.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_i.savefig(base / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig_i)
    print(f"Saved {fname}.png")
