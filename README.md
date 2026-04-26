---
title: LivePatch Environment
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - postgresql
  - database
  - incident-response
---

# LivePatch

**A reinforcement learning environment for autonomous PostgreSQL incident response.**

LivePatch trains agents to diagnose and remediate database faults -- missing indexes, table bloat, stale statistics -- while production traffic continues to flow. The agent must balance speed of resolution against operational safety: a fix that causes downtime scores worse than no fix at all.

> Built as a solo entry for the [OpenEnv Hackathon 2026](https://huggingface.co/openenv).

---

## Table of Contents

- [Overview](#overview)
- [Environment Design](#environment-design)
- [Training Approach](#training-approach)
- [Results](#results)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Resources](#resources)

---

## Overview

| | |
|---|---|
| **Task** | Diagnose and fix PostgreSQL incidents under live traffic |
| **Observation** | Text-based database metrics, traffic stats, fault indicators |
| **Action Space** | Free-form SQL commands (CREATE INDEX, VACUUM, ANALYZE, etc.) |
| **Reward** | Composite: fix quality + uptime + efficiency + safety |
| **Step Budget** | 15 steps per episode |
| **Fault Types** | missing_index, table_bloat, stale_stats |

The environment is fully simulated -- no real PostgreSQL instance required. All database operations, traffic patterns, and query costs are modeled in-process, enabling fast iteration and deterministic evaluation.

---

## Environment Design

### Architecture

```
Agent ──── SQL command ────> SimulatedDatabase ──── observation ────> Agent
                                    |
                              ┌─────┴─────┐
                         TrafficSim    CostModel
                         (QPS/SLA)    (EXPLAIN)
                              |           |
                          uptime       fix_quality
```

### Fault Types

| Fault | Symptoms | Safe Resolution | Unsafe Alternative |
|-------|----------|-----------------|-------------------|
| missing_index | Sequential scans, p99 > 4000ms | `CREATE INDEX CONCURRENTLY` | `CREATE INDEX` (table lock) |
| table_bloat | Dead tuple accumulation, disk I/O | `VACUUM ANALYZE` | `VACUUM FULL` (blocks reads) |
| stale_stats | Query planner estimation errors | `ANALYZE tablename` | -- |

### Scoring

Episodes are evaluated on four axes:

| Metric | Weight | Description |
|--------|--------|-------------|
| Fix Quality | 35% | Query cost improvement after remediation |
| Uptime | 30% | Fraction of traffic served successfully |
| Efficiency | 20% | Steps consumed relative to budget |
| Safety | 15% | Ratio of non-blocking operations used |

### Difficulty Curriculum

| Tier | Episodes | Faults | Description |
|------|----------|--------|-------------|
| Easy | 0 -- 24 | 1 | Single fault, lenient SLA thresholds |
| Medium | 25 -- 39 | 1 | Tighter thresholds |
| Hard | 40 -- 49 | 2+ | Multiple concurrent faults, strict thresholds |

---

## Training Approach

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2.5-1.5B-Instruct |
| Adaptation | LoRA (r=32, 36.9M trainable parameters) |
| Quantization | 4-bit via Unsloth |
| Hardware | NVIDIA A100-SXM4-80GB |
| Episode Duration | ~80 seconds |

### Method: SFT + GRPO

**Phase 1 -- Supervised Fine-Tuning.** 3 epochs on 51 expert demonstration trajectories covering all fault types. Establishes the command vocabulary and output format.

**Phase 2 -- Group Relative Policy Optimization (GRPO).** 50 episodes with GROUP_SIZE=8. For each scenario:

1. Generate 8 rollouts against the same fault (same seed)
2. Compute advantages relative to group mean reward
3. Update policy on all advantages (positive and negative)

### Reward Design

All bonuses (SLA, safety, efficiency) are gated on `resolved_faults > 0` to prevent reward hacking. Diagnostic commands (`EXPLAIN ANALYZE`, `pg_stat` queries) receive small shaping rewards (+0.1 to +0.15) to encourage investigation before action.

### Key Design Decisions

1. **Same-seed groups** -- All 8 group members face the identical fault scenario. GRPO compares strategies, not luck.
2. **Train on all advantages** -- Both positive (reinforce) and negative (suppress) advantages are used. Discarding negative advantages wastes 50% of the training signal.
3. **Bonus gating** -- Without gating, the model learns to submit immediately and collect free SLA/safety bonuses without resolving any faults.
4. **Auto-submit at max steps** -- Ensures every episode produces a complete reward signal.

---

## Results

### Best Performance (Episode 19, Easy Difficulty)

| Metric | Score |
|--------|-------|
| **Overall** | **0.673** |
| Fix Quality | 0.421 |
| Uptime | 0.721 |
| Safety | 1.000 |

### Training Progression

| Episode | Event |
|---------|-------|
| 0 | First valid SQL command generated |
| 5 | Safety score stabilized at 1.0 |
| 17 | First fault resolved (missing_index via CREATE INDEX CONCURRENTLY) |
| 19 | Peak composite score: 0.673 |

### Comparison with Baselines

| Agent | Score | Fix Quality | Uptime | Safety |
|-------|-------|-------------|--------|--------|
| Random | 0.60 | 0.00 | 0.66 | 0.50 |
| Heuristic | 0.65 | 0.00 | 0.78 | 1.00 |
| **GRPO (best)** | **0.67** | **0.42** | **0.72** | **1.00** |

![Training Curves](training/training_curves.png)

### Limitations

- The 1.5B model frequently generates malformed output (observation fragments, step counters) instead of valid SQL. A larger model (7B+) or constrained decoding would likely improve command reliability.
- Fix quality is inconsistent across episodes. The model resolves `missing_index` faults but struggles with `table_bloat` and `stale_stats`.

---

## Quick Start

### Local Setup

```bash
pip install flask numpy pyyaml gunicorn

# Start the server
python -m server.app

# Run the example agent
python examples/run_random_agent.py

# Run tests
python -m pytest tests/ -v
```

### Docker

```bash
docker build -t livepatch .
docker run -p 7860:7860 livepatch
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page with interactive console |
| `/reset` | POST | Initialize new incident scenario |
| `/step` | POST | Execute a SQL command |
| `/state` | GET | Retrieve current environment state |
| `/grade` | GET | Evaluate episode performance |
| `/health` | GET | Service health check |

### Example: Reset

```bash
curl -X POST https://rupeshreddy7-livepatch-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy", "max_steps": 15}'
```

### Example: Step

```bash
curl -X POST https://rupeshreddy7-livepatch-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"command": "EXPLAIN ANALYZE SELECT * FROM orders WHERE order_id = 42;"}'
```

### Python Client

```python
from src.environment import LivePatchEnv
from src.config import EnvironmentConfig

env = LivePatchEnv(EnvironmentConfig(difficulty='easy', seed=42))
obs = env.reset()

obs = env.step({'command': '\\dt'})
obs = env.step({'command': 'EXPLAIN ANALYZE SELECT * FROM users WHERE email = \'test@test.com\''})
obs = env.step({'command': 'CREATE INDEX CONCURRENTLY idx_users_email ON users(email)'})
obs = env.step({'command': 'submit'})
```

---

## Project Structure

```
livepatch-env/
  server/app.py            Flask REST API with landing page
  src/
    config.py              Tables, queries, indexes, fault definitions
    models.py              Data models (TableState, IndexState, FaultSpec)
    environment.py         LivePatchEnv (OpenEnv reset/step/state interface)
    tasks.py               Task presets and grading logic
  training/
    train.py               GRPO training script
    livepatch_grpo_training.ipynb   Colab notebook
    training_log.json      Episode-level training data
    training_curves.png    Reward and loss plots
  tests/                   Test suite
  examples/                Example agents
  BLOG.md                  Technical writeup
  openenv.yaml             OpenEnv manifest
  Dockerfile               Container configuration
```

---

## Resources

| Resource | Link |
|----------|------|
| Live Environment | [huggingface.co/spaces/rupeshreddy7/livepatch-env](https://huggingface.co/spaces/rupeshreddy7/livepatch-env) |
| Training Dashboard | [huggingface.co/spaces/rupeshreddy7/livepatch-training](https://huggingface.co/spaces/rupeshreddy7/livepatch-training) |
| Source Code | [github.com/rupeshreddy007/livepatch-env](https://github.com/rupeshreddy007/livepatch-env) |
| Technical Writeup | [BLOG.md](BLOG.md) |
| Training Notebook | [training/livepatch_grpo_training.ipynb](training/livepatch_grpo_training.ipynb) |
| Training Script | [train.py](https://huggingface.co/spaces/rupeshreddy7/livepatch-training/blob/main/train.py) |

---

## OpenEnv Compatibility

Implements the standard OpenEnv interface:

- `openenv.yaml` -- Environment manifest
- `reset()` / `step()` / `state()` -- Standard RL API
- `Dockerfile` -- Containerized deployment on port 7860
- `inference.py` -- HuggingFace-compatible agent interface