# LivePatch: Teaching a 1.5B Model to Be an On-Call DBA with Reinforcement Learning

*What happens when you give a tiny language model a pager, a PostgreSQL terminal, and tell it to fix production?*

## TL;DR

We used **Group Relative Policy Optimization (GRPO)** to train **Qwen2.5-1.5B** to autonomously diagnose and remediate PostgreSQL database incidents — missing indexes, table bloat, and stale statistics — in a live simulated production environment with real-time traffic. The model went from generating gibberish to successfully creating indexes and resolving faults within 22 training episodes. Along the way, we discovered five hard-won lessons about applying GRPO to interactive environments that we wish someone had told us before we started.

**Project Links:**
- [Live Environment (HF Space)](https://huggingface.co/spaces/rupeshreddy7/livepatch-env) — Try the environment yourself
- [Training Dashboard (HF Space)](https://huggingface.co/spaces/rupeshreddy7/livepatch-training) — Watch training in real-time
- [GitHub Repository](https://github.com/rupeshreddy007/livepatch-env) — Full source code, training scripts, and logs

---

## Motivation: Why Automate Incident Response?

It's 3 AM. Your phone buzzes. P99 latency on the checkout service just crossed 5 seconds. Customers are dropping off. The on-call DBA connects to PostgreSQL, checks `pg_stat_activity`, runs `EXPLAIN ANALYZE` on the slow queries, spots a missing index, creates it with `CREATE INDEX CONCURRENTLY`, and watches latency drop back to normal. Total time: 15 minutes of high-stress work.

This workflow — **observe symptoms, diagnose root cause, apply fix, verify resolution** — is remarkably structured. It follows a pattern that should be learnable. But it also requires:

1. **Reading complex, noisy observations** (traffic metrics, system stats, query plans)
2. **Generating precise SQL commands** (not prose, not explanations — executable SQL)
3. **Multi-step reasoning** (diagnosis before treatment)
4. **Working under a time budget** (every step costs uptime)

We wanted to see: can reinforcement learning teach a small model this workflow from scratch?

---

## The Environment: A PostgreSQL Incident Simulator

We built a full incident response environment that simulates a production PostgreSQL database under load. The environment is available as an [OpenEnv-compatible HuggingFace Space](https://huggingface.co/spaces/rupeshreddy7/livepatch-env) with a REST API (`/reset`, `/step`, `/state`).

### Fault Types

| Fault | What Goes Wrong | What the Agent Sees | Correct Fix |
|-------|----------------|--------------------| ------------|
| **Missing Index** | Sequential scans on large tables | p99 > 4000ms, high failed requests | `CREATE INDEX CONCURRENTLY ...` |
| **Table Bloat** | Dead tuples accumulating, disk I/O spike | Elevated dead tuple ratio, slow writes | `VACUUM ANALYZE tablename;` |
| **Stale Statistics** | Query planner makes bad estimates | Row estimate errors, suboptimal plans | `ANALYZE tablename;` |

### What Makes It Challenging

Each episode gives the agent a **15-step budget**. At each step, the agent sees a rich text observation:

```
[DATABASE STATUS]
PostgreSQL 14.2 | Uptime: 99.2% | Connections: 45/100

[TRAFFIC]
[Traffic] OK:342 Failed:158 p99:4952.4ms

[SYSTEM METRICS]  
CPU: 67.3% | Memory: 4.2GB/8GB | Disk I/O: 234 IOPS

[ACTIVE FAULTS]
- missing_index [OPEN] (severity: medium)

[RECENT QUERIES]
Sequential scans detected on: checkout_orders (est. rows: 1.2M)
```

The agent must respond with a single SQL command. Not an explanation. Not a step-by-step plan. Just the command. This turns out to be surprisingly hard for a 1.5B model.

### Difficulty Curriculum

| Phase | Episodes | Faults | Thresholds |
|-------|----------|--------|------------|
| **Easy** | 0–24 | Single fault | Lenient SLA targets |
| **Medium** | 25–39 | Single fault | Tighter targets |
| **Hard** | 40–49 | Multiple simultaneous faults | Strict targets |

---

## Training Pipeline

### Phase 1: SFT Bootstrapping (Warm Start)

Before any RL, we run **3 epochs of supervised fine-tuning** on **51 expert demonstration trajectories**. These demonstrations show the model what good incident response looks like:

```
Observation: [Traffic] OK:342 Failed:158 p99:4952.4ms ... missing_index [OPEN]
Expert Action: CREATE INDEX CONCURRENTLY idx_checkout_order_id ON checkout_orders(order_id);
```

This SFT phase is critical. Without it, the model's initial policy is so random that GRPO has no useful signal to work with — every response in the group is equally bad, so advantages are near zero.

### Phase 2: GRPO (The Main Event)

**Group Relative Policy Optimization** is our core training algorithm. Here's how it works in our setup:

```
For each episode:
  1. Sample a fault scenario (type + seed)
  2. Generate 8 rollouts, all facing the SAME scenario
  3. Each rollout: agent interacts with environment for up to 15 steps
  4. Collect total_reward for each rollout
  5. Compute advantages: advantage_i = reward_i - mean(rewards)
  6. Update policy: reinforce positive advantages, suppress negative ones
```

**Key hyperparameters:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Group size | 8 | Enough variance for meaningful comparisons |
| Learning rate | 5e-5 | Conservative to avoid catastrophic forgetting |
| LoRA rank | 32 | Balance between capacity and efficiency |
| Max sequence length | 1024 | Fits observation + response |
| KL coefficient | 0.05 | Light regularization toward SFT policy |
| Quantization | 4-bit (Unsloth) | Fits on single GPU with 8 concurrent rollouts |

### The Reward Function: Three Iterations to Get Right

This is where we spent most of our debugging time. Here's the evolution:

**Version 1 (Broken):** Simple sum of component scores.
```
reward = fix_quality + sla_bonus + safety_bonus + efficiency_bonus - penalties
```
*Problem:* The model learned to submit immediately and collect free SLA/safety/efficiency bonuses without fixing anything. Score: ~0.5 while fixing nothing.

**Version 2 (Better):** Gate bonuses on actually fixing something.
```
if resolved_faults > 0:
    reward += sla_bonus + safety_bonus + efficiency_bonus
reward -= 2.0 * unresolved_faults  # harsh penalty
```
*Problem:* Traffic penalty (-0.5 per step) accumulated to -7.5 over 15 steps, drowning out the fix signal. The model learned that the fastest way to reduce penalty was to submit early, not to fix the problem.

**Version 3 (Final):** Balanced penalties + diagnostic shaping.
```
traffic_penalty: capped at -0.15 per step (max -2.25 total)
unresolved_penalty: -2.0 per fault
diagnostic_shaping: +0.15 for EXPLAIN, +0.1 for pg_stat queries
all bonuses: gated on resolved > 0
```
This version finally creates the right incentive gradient: diagnose first (small positive rewards), then fix (large positive reward), with manageable penalties for time spent.

---

## Five Hard-Won Lessons

### Lesson 1: Same Seed or Bust

In our first GRPO implementation, each of the 8 group members faced a **different random fault**. Member 1 might get an easy `stale_stats`, while Member 5 gets a hard `missing_index`. The computed advantages reflected **fault difficulty, not strategy quality**.

**The fix:** All 8 members face the exact same fault with the same random seed. Now GRPO compares 8 different *strategies* for the same *problem*. This is the single most important implementation detail.

### Lesson 2: Train on ALL Advantages

Following some GRPO implementations, we initially only updated on responses with **positive advantages** (better than group mean). This sounds reasonable — reinforce the good stuff.

But it throws away 50% of the training signal. The model needs to learn what *not* to do just as much as what to do. When we switched to training on all advantages — reinforcing good responses and **actively suppressing bad ones** — learning improved dramatically.

### Lesson 3: Reward Hacking Finds Every Loophole

Our model discovered that submitting in 3 steps without doing anything scored ~0.5 (free SLA bonus + safety bonus + efficiency bonus for being "fast"). This was *higher* than many episodes where it tried to fix things but failed.

The model is a perfect optimizer. If your reward function has a shortcut, the model will find it. Gating every bonus on `resolved_faults > 0` closed the loophole.

### Lesson 4: 1.5B Parameters Hit a Ceiling on Structured Output

The most frustrating failure mode: the model would generate text like:

```
2/15]
99:4952.4ms
Step 2 - Diagnose: Execute EXPLAIN ANALYZE on the slow queries:
```

Instead of:

```
EXPLAIN ANALYZE SELECT * FROM checkout_orders WHERE order_id = 12345;
```

It's *echoing observations* and *writing prose* instead of generating SQL. The model understands the concept (it mentions the right tools) but can't reliably produce well-formed commands. This is a fundamental limitation at the 1.5B scale — larger models (7B+) or constrained decoding would likely solve this.

### Lesson 5: Auto-Submit Closes the Reward Loop

Episodes that hit the 15-step limit without the agent calling `submit` never received a final score. These episodes generated zero reward signal — wasted compute. Adding automatic submission at max steps ensured every episode produced a usable reward, even if the agent ran out of time.

---

## Results

### Training Progression

![Training Curves](training/training_curves.png)

| Milestone | Episode | What Happened |
|-----------|---------|---------------|
| Training starts | 0 | Mean reward: -0.67, mostly random commands |
| Mode collapse | 1–16 | Model generates observation fragments, rewards stay negative |
| **First fix** | **17** | `missing_index` resolved with `CREATE INDEX CONCURRENTLY` |
| **Best score** | **19** | Score: 0.673, fix quality: 0.421 |
| Continued training | 20–21 | Mixed results, model inconsistent |

### Episode 17: The Breakthrough

After 16 episodes of generating garbage, the model produced its first successful fix:

```sql
CREATE INDEX CONCURRENTLY idx_checkout_order_status ON checkout_order(status);
```

This single command resolved the `missing_index` fault. The model had learned — from GRPO's relative comparisons — that this command format leads to higher rewards than echoing observation text.

### Honest Assessment

- The model **can** fix `missing_index` faults (the most straightforward: one CREATE INDEX command)
- It **struggles** with `table_bloat` and `stale_stats` (require VACUUM/ANALYZE on specific tables)
- **Command extraction** remains the bottleneck — most episodes still produce invalid output
- A larger model or constrained decoding is needed for production reliability

---

## Technical Architecture

```
+------------------+     REST API      +------------------+
|                  |  /reset, /step    |                  |
|   GRPO Trainer   | <--------------> |   Environment    |
|   (A100 GPU)     |   /state          |   (CPU Space)    |
|                  |                   |                  |
|  Qwen2.5-1.5B   |                   |  Fault Injection |
|  + LoRA (r=32)   |                   |  Traffic Sim     |
|  + 4-bit quant   |                   |  Metrics Engine  |
|                  |                   |                  |
+------------------+                   +------------------+
    |                                       |
    | Unsloth fast inference                | Flask + Gunicorn
    | 8 concurrent rollouts                 | Docker container
    | ~80s per episode                      | OpenEnv compatible
```

### Stack

| Component | Technology |
|-----------|-----------|
| Base model | Qwen2.5-1.5B-Instruct |
| Fine-tuning | LoRA (r=32, 36.9M trainable params) |
| Quantization | 4-bit via Unsloth |
| Training | Custom GRPO loop |
| Environment | Flask REST API in Docker |
| Training infra | HF Spaces (A100-SXM4-80GB) |
| Env hosting | HF Spaces (Free CPU) |

---

## Reproducing This Work

### Quick Start

```bash
# Clone the repo
git clone https://github.com/rupeshreddy007/livepatch-env
cd livepatch-env

# Run the environment locally
pip install -r requirements.txt
python server/app.py

# In another terminal, test with the random agent
python examples/run_random_agent.py
```

### Training

The full training script is at [`training/train.py`](training/train.py). You'll need:
- A GPU with 40GB+ VRAM (A100 recommended)
- The environment running (locally or on HF Spaces)
- Unsloth installed (`pip install unsloth`)

A Colab notebook is also available: [`training/livepatch_grpo_training.ipynb`](training/livepatch_grpo_training.ipynb)

---

## What We'd Do Differently

1. **Start with a 7B model.** The 1.5B command extraction problem consumed most of our debugging time. A larger model would likely produce valid SQL from the start, letting GRPO focus on *strategy* rather than *syntax*.

2. **Use constrained decoding.** Force the model to output valid SQL by restricting the token vocabulary during generation. This would eliminate the observation-echoing failure mode entirely.

3. **More expert demonstrations.** 51 SFT examples weren't enough to fully teach the output format. 200+ examples covering edge cases would give GRPO a much stronger starting policy.

4. **Longer training.** 50 episodes is minimal. With more compute budget, 200+ episodes with the curriculum would likely show continued improvement, especially on harder fault types.

5. **Multi-turn memory.** Currently each step is independent — the model doesn't remember what it tried before. Adding a condensed history of previous commands and results would prevent the model from repeating failed approaches.

---

## Conclusion

LivePatch demonstrates that **GRPO can teach small models structured, multi-step workflows** — even complex ones like database incident response. The model went from zero knowledge to successfully diagnosing and fixing PostgreSQL index issues in 22 episodes of training.

The five lessons we learned — same-seed groups, training on all advantages, reward gating, auto-submit, and the 1.5B ceiling — are broadly applicable to anyone applying GRPO to interactive agent environments. We hope this writeup saves someone else the debugging time we spent discovering them.

**The code is open source. Try breaking our environment, training your own DBA, or scaling up to 7B. We'd love to see what you build.**

---

*Built for the [OpenEnv Hackathon](https://huggingface.co/openenv) by [rupeshreddy7](https://huggingface.co/rupeshreddy7)*
