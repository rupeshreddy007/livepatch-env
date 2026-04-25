"""
LivePatch — GRPO Training Script
Run on HF Spaces (A10G GPU) or Colab.

Usage:
    python training/train.py

Outputs:
    - livepatch-qwen-1.5b-lora/      (LoRA adapter)
    - livepatch-checkpoint-ep*/       (intermediate checkpoints)
    - training_log.json               (per-episode stats)
    - training_curves.png             (6-panel plot)
    - agent_comparison.png            (bar chart vs baselines)
"""

import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

import sys
import json
import time
import random
import re
from collections import defaultdict

import numpy as np
import torch
from torch.optim import AdamW

# ── Setup paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.environment import LivePatchEnv, SYSTEM_PROMPT
from src.config import EnvironmentConfig, DIFFICULTY_EASY, DIFFICULTY_MEDIUM
from src.tasks import grade_episode, run_task, TASKS
from baseline import heuristic_agent, random_agent

# ── Config ──
MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'
MAX_SEQ_LENGTH = 1024
NUM_EPISODES = 300
GROUP_SIZE = 6
LEARNING_RATE = 5e-5
KL_COEFF = 0.05
CLIP_EPSILON = 0.2
MAX_GEN_LEN = 128

DIFFICULTY_SCHEDULE = {
    0: 'easy',       # episodes 0-99
    100: 'medium',   # episodes 100-199
    200: 'hard',     # episodes 200-299
}

AGENT_SYSTEM_PROMPT = SYSTEM_PROMPT + """

RULES:
- Respond with ONLY the psql command. No explanations.
- First explore (\\dt, \\di, pg_stat_activity), then diagnose (EXPLAIN), then fix.
- Use CONCURRENTLY for CREATE INDEX to avoid locking.
- Use VACUUM ANALYZE (not VACUUM FULL) to avoid downtime.
- When done, respond with: submit
"""


# ── Helpers ──

def get_difficulty(episode):
    diff = 'easy'
    for ep, d in sorted(DIFFICULTY_SCHEDULE.items()):
        if episode >= ep:
            diff = d
    return diff


def extract_command(text):
    text = text.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    for line in text.split('\n'):
        line = line.strip().strip('`').strip()
        if line and not line.startswith(('#', '//', '--', 'I ', 'The ', 'Let', 'Based')):
            return line
    return text.split('\n')[0].strip() if text else 'submit'


# ── Adversarial Curriculum ──

class AdversarialCurriculum:
    def __init__(self):
        self.fault_attempts = defaultdict(int)
        self.fault_resolved = defaultdict(int)
        self.fault_types = [
            'missing_index', 'table_bloat', 'stale_stats', 'bad_config',
            'lock_storm', 'connection_flood', 'runaway_query', 'sort_spill'
        ]

    def record(self, faults_injected, faults_resolved):
        for f in faults_injected:
            self.fault_attempts[f] += 1
        for f in faults_resolved:
            self.fault_resolved[f] += 1

    def get_weakness_scores(self):
        scores = {}
        for f in self.fault_types:
            attempts = self.fault_attempts.get(f, 0)
            resolved = self.fault_resolved.get(f, 0)
            scores[f] = 1.0 if attempts == 0 else 1.0 - (resolved / attempts)
        return scores

    def get_adversarial_seed(self, difficulty, base_seed):
        weaknesses = self.get_weakness_scores()
        if sum(self.fault_attempts.values()) < 20:
            return base_seed

        best_seed, best_score = base_seed, -1
        for candidate_seed in range(base_seed, base_seed + 5):
            config = EnvironmentConfig(difficulty=difficulty, seed=candidate_seed, max_steps=20)
            env = LivePatchEnv(config)
            env.reset()
            injected = env.state().get('faults_injected', [])
            score = sum(weaknesses.get(f, 0.5) for f in injected)
            if score > best_score:
                best_score = score
                best_seed = candidate_seed
        return best_seed

    def report(self):
        scores = self.get_weakness_scores()
        sorted_faults = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print("\n  Adversarial Curriculum — Weakness Report:")
        for fault, score in sorted_faults:
            attempts = self.fault_attempts.get(fault, 0)
            resolved = self.fault_resolved.get(fault, 0)
            bar = '#' * int(score * 20)
            print(f"    {fault:<20} weakness={score:.2f} [{bar:<20}] ({resolved}/{attempts})")


# ── Episode Runner ──

def run_episode(model, tokenizer, difficulty='easy', seed=None, max_gen_len=150):
    if seed is None:
        seed = random.randint(0, 999999)

    config = EnvironmentConfig(difficulty=difficulty, seed=seed, max_steps=20)
    env = LivePatchEnv(config)
    obs = env.reset()

    messages = [{'role': 'system', 'content': AGENT_SYSTEM_PROMPT}]
    episode_prompts, episode_completions, step_rewards = [], [], []

    while not obs['done']:
        traffic = obs.get('traffic', {})
        user_msg = obs['observation']
        user_msg += f"\n\n[Traffic] OK:{traffic.get('requests_ok',0)} "
        user_msg += f"Failed:{traffic.get('requests_failed',0)} "
        user_msg += f"p99:{traffic.get('p99_latency_ms',0):.1f}ms"
        user_msg += f"\n[Step {obs['step']}/{obs['max_steps']}]"

        messages.append({'role': 'user', 'content': user_msg})

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors='pt', truncation=True,
                          max_length=MAX_SEQ_LENGTH - max_gen_len).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_gen_len,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        command = extract_command(completion)

        messages.append({'role': 'assistant', 'content': command})
        obs = env.step({'command': command})

        episode_prompts.append(prompt_text)
        episode_completions.append(completion)
        step_rewards.append(obs['reward'])

    result = grade_episode(env)
    total_reward = obs.get('total_reward', sum(step_rewards))

    return {
        'prompts': episode_prompts,
        'completions': episode_completions,
        'step_rewards': step_rewards,
        'total_reward': total_reward,
        'grade': result,
        'commands': [extract_command(c) for c in episode_completions],
    }


# ── Main ──

def main():
    from unsloth import FastLanguageModel

    print('=' * 60)
    print('LivePatch GRPO Training')
    print('=' * 60)

    # ── Baselines ──
    print('\n>> Running baselines...')
    baseline_results = {}
    for agent_name, agent_fn in [('random', random_agent), ('heuristic', heuristic_agent)]:
        for task in ['easy', 'medium', 'hard']:
            result = run_task(task, agent_fn, n_episodes=3)
            baseline_results[f'{agent_name}_{task}'] = result
            print(f"  {agent_name:>10} | {task:<6} | score={result['avg_score']:.4f}")

    # ── Load Model ──
    print('\n>> Loading model...')
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        lora_alpha=32,
        lora_dropout=0,
        bias='none',
        use_gradient_checkpointing='unsloth',
    )
    print(f'  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    # ── Training ──
    print(f'\n>> Training: {NUM_EPISODES} episodes, GROUP_SIZE={GROUP_SIZE}, LR={LEARNING_RATE}')
    print(f'   Schedule: {DIFFICULTY_SCHEDULE}')
    print(f'   Adversarial curriculum: ENABLED\n')

    adversarial = AdversarialCurriculum()
    training_log = []
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    for episode in range(NUM_EPISODES):
        t0 = time.time()
        difficulty = get_difficulty(episode)
        base_seed = random.randint(0, 999999)
        seed = adversarial.get_adversarial_seed(difficulty, base_seed)

        # Rollouts
        group_results = []
        FastLanguageModel.for_inference(model)
        for g in range(GROUP_SIZE):
            result = run_episode(model, tokenizer, difficulty=difficulty,
                                seed=seed, max_gen_len=MAX_GEN_LEN)
            group_results.append(result)
        FastLanguageModel.for_training(model)

        # Record for adversarial curriculum
        for result in group_results:
            adversarial.record(result['grade'].faults_injected, result['grade'].faults_resolved)

        # GRPO advantages
        rewards = [r['total_reward'] for r in group_results]
        mean_reward = np.mean(rewards)
        std_reward = max(np.std(rewards), 1e-6)
        advantages = [(r - mean_reward) / std_reward for r in rewards]

        # Policy update
        total_loss, n_updates = 0.0, 0
        for g_idx, (result, advantage) in enumerate(zip(group_results, advantages)):
            if advantage <= 0:
                continue
            for prompt, completion in zip(result['prompts'], result['completions']):
                full_text = prompt + completion
                inputs = tokenizer(full_text, return_tensors='pt', truncation=True,
                                  max_length=MAX_SEQ_LENGTH).to(model.device)
                prompt_len = len(tokenizer(prompt, truncation=True,
                                          max_length=MAX_SEQ_LENGTH)['input_ids'])

                outputs = model(**inputs, labels=inputs['input_ids'])
                logits = outputs.logits[:, prompt_len-1:-1, :]
                labels = inputs['input_ids'][:, prompt_len:]
                if labels.shape[1] == 0:
                    continue

                loss = torch.nn.CrossEntropyLoss()(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
                )
                (-advantage * loss).backward()
                total_loss += loss.item()
                n_updates += 1

        if n_updates > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Log
        best_result = group_results[np.argmax(rewards)]
        elapsed = time.time() - t0
        log_entry = {
            'episode': episode,
            'difficulty': difficulty,
            'mean_reward': float(mean_reward),
            'max_reward': float(max(rewards)),
            'min_reward': float(min(rewards)),
            'std_reward': float(std_reward),
            'loss': total_loss / max(1, n_updates),
            'n_updates': n_updates,
            'best_score': best_result['grade'].score,
            'best_fix_quality': best_result['grade'].fix_quality,
            'best_uptime': best_result['grade'].uptime,
            'best_safety': best_result['grade'].safety,
            'best_commands': best_result['commands'],
            'faults_injected': best_result['grade'].faults_injected,
            'faults_resolved': best_result['grade'].faults_resolved,
            'adversarial_seed_used': seed != base_seed,
            'elapsed_s': elapsed,
        }
        training_log.append(log_entry)

        resolved = len(best_result['grade'].faults_resolved)
        injected = len(best_result['grade'].faults_injected)
        adv = '*' if seed != base_seed else ' '
        print(f"Ep {episode:>3}{adv}[{difficulty:>6}] "
              f"reward={mean_reward:>+7.2f} (±{std_reward:.2f}) "
              f"score={best_result['grade'].score:.3f} "
              f"fix={best_result['grade'].fix_quality:.3f} "
              f"uptime={best_result['grade'].uptime:.3f} "
              f"safe={best_result['grade'].safety:.3f} "
              f"resolved={resolved}/{injected} "
              f"loss={total_loss/max(1,n_updates):.4f} "
              f"updates={n_updates} [{elapsed:.1f}s]")

        if (episode + 1) % 50 == 0:
            model.save_pretrained(f'livepatch-checkpoint-ep{episode+1}')
            tokenizer.save_pretrained(f'livepatch-checkpoint-ep{episode+1}')
            print(f'  >> Checkpoint saved at episode {episode+1}')
            adversarial.report()

    print(f'\n✅ Training complete — {NUM_EPISODES} episodes')
    adversarial.report()

    # ── Save Model ──
    model.save_pretrained('livepatch-qwen-1.5b-lora')
    tokenizer.save_pretrained('livepatch-qwen-1.5b-lora')
    print('\n✅ Model saved to livepatch-qwen-1.5b-lora/')

    # ── Save Training Log ──
    with open('training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2, default=str)
    print(f'✅ Training log saved ({len(training_log)} episodes)')

    # ── Plots ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        matplotlib.rcParams['figure.dpi'] = 120

        episodes_list = [e['episode'] for e in training_log]
        mean_rewards = [e['mean_reward'] for e in training_log]
        max_rewards = [e['max_reward'] for e in training_log]
        scores = [e['best_score'] for e in training_log]
        fix_quality = [e['best_fix_quality'] for e in training_log]
        uptimes = [e['best_uptime'] for e in training_log]
        safeties = [e['best_safety'] for e in training_log]
        losses = [e['loss'] for e in training_log]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle('LivePatch — GRPO Training Results', fontsize=14, fontweight='bold')

        ax = axes[0, 0]
        ax.plot(episodes_list, mean_rewards, 'b-', label='Mean', linewidth=2)
        ax.fill_between(episodes_list, [e['min_reward'] for e in training_log],
                        max_rewards, alpha=0.2, color='blue')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        for ep, diff in DIFFICULTY_SCHEDULE.items():
            if ep > 0:
                ax.axvline(x=ep, color='red', linestyle=':', alpha=0.5)
                ax.text(ep+0.3, ax.get_ylim()[1]*0.9, diff, fontsize=8, color='red')
        ax.set_xlabel('Episode'); ax.set_ylabel('Total Reward')
        ax.set_title('Reward (mean ± range)'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(episodes_list, scores, 'g-', linewidth=2)
        ax.set_xlabel('Episode'); ax.set_ylabel('Score (0-1)')
        ax.set_title('Overall Score'); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        ax.plot(episodes_list, fix_quality, 'r-', linewidth=2)
        ax.set_xlabel('Episode'); ax.set_ylabel('Fix Quality (0-1)')
        ax.set_title('EXPLAIN Cost Improvement'); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(episodes_list, uptimes, 'm-', linewidth=2)
        ax.set_xlabel('Episode'); ax.set_ylabel('Uptime (0-1)')
        ax.set_title('Traffic Uptime During Fix'); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(episodes_list, safeties, 'c-', linewidth=2)
        ax.set_xlabel('Episode'); ax.set_ylabel('Safety (0-1)')
        ax.set_title('Safe Operations Ratio'); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        ax.plot(episodes_list, losses, 'k-', linewidth=2)
        ax.set_xlabel('Episode'); ax.set_ylabel('Loss')
        ax.set_title('Training Loss'); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_curves.png', bbox_inches='tight')
        print('✅ Saved training_curves.png')

        # ── Post-Training Eval ──
        print('\n>> Post-training evaluation...')

        def trained_agent(obs, state):
            if not hasattr(trained_agent, 'messages'):
                trained_agent.messages = [{'role': 'system', 'content': AGENT_SYSTEM_PROMPT}]
            traffic = obs.get('traffic', {})
            user_msg = obs['observation']
            user_msg += f"\n[Traffic] OK:{traffic.get('requests_ok',0)} "
            user_msg += f"Failed:{traffic.get('requests_failed',0)} "
            user_msg += f"p99:{traffic.get('p99_latency_ms',0):.1f}ms"
            user_msg += f"\n[Step {obs['step']}/{obs['max_steps']}]"
            trained_agent.messages.append({'role': 'user', 'content': user_msg})

            prompt_text = tokenizer.apply_chat_template(
                trained_agent.messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors='pt', truncation=True,
                              max_length=MAX_SEQ_LENGTH - 100).to(model.device)
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=100, temperature=0.3,
                    do_sample=True, pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
            command = extract_command(completion)
            trained_agent.messages.append({'role': 'assistant', 'content': command})
            return {'command': command}

        def reset_trained_agent():
            if hasattr(trained_agent, 'messages'):
                del trained_agent.messages

        eval_results = {}
        for task in ['easy', 'medium', 'hard']:
            r_random = run_task(task, random_agent, n_episodes=3)
            r_heuristic = run_task(task, heuristic_agent, n_episodes=3)

            trained_scores = []
            for ep_seed in [9001, 9002, 9003]:
                reset_trained_agent()
                config = EnvironmentConfig(
                    difficulty=task, seed=ep_seed,
                    max_steps=TASKS[task].max_steps,
                    traffic_rps=TASKS[task].traffic_rps,
                )
                env = LivePatchEnv(config)
                obs = env.reset()
                while not obs['done']:
                    action = trained_agent(obs, env.state())
                    obs = env.step(action)
                trained_scores.append(grade_episode(env))

            avg_trained = sum(r.score for r in trained_scores) / len(trained_scores)
            eval_results[task] = {
                'random': r_random['avg_score'],
                'heuristic': r_heuristic['avg_score'],
                'trained': avg_trained,
            }
            print(f"  {task:>8}: Random={r_random['avg_score']:.4f}  "
                  f"Heuristic={r_heuristic['avg_score']:.4f}  "
                  f"Trained={avg_trained:.4f}")

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        tasks = list(eval_results.keys())
        x = np.arange(len(tasks))
        width = 0.25
        bars1 = ax.bar(x - width, [eval_results[t]['random'] for t in tasks],
                       width, label='Random', color='#ff6b6b', alpha=0.8)
        bars2 = ax.bar(x, [eval_results[t]['heuristic'] for t in tasks],
                       width, label='Heuristic', color='#ffd93d', alpha=0.8)
        bars3 = ax.bar(x + width, [eval_results[t]['trained'] for t in tasks],
                       width, label='GRPO Trained', color='#6bcb77', alpha=0.8)
        ax.set_xlabel('Task Difficulty', fontsize=12)
        ax.set_ylabel('Score (0-1)', fontsize=12)
        ax.set_title('LivePatch — Agent Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in tasks])
        ax.set_ylim(0, 1); ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis='y')
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig('agent_comparison.png', bbox_inches='tight')
        print('✅ Saved agent_comparison.png')

    except Exception as e:
        print(f'⚠️  Plotting failed (non-fatal): {e}')

    # ── Summary ──
    print('\n' + '=' * 60)
    print('TRAINING COMPLETE')
    print('=' * 60)
    print(f"Episodes: {len(training_log)}")
    print(f"First 5 avg reward: {np.mean([e['mean_reward'] for e in training_log[:5]]):.3f}")
    print(f"Last 5 avg reward:  {np.mean([e['mean_reward'] for e in training_log[-5:]]):.3f}")
    print(f"First 5 avg score:  {np.mean([e['best_score'] for e in training_log[:5]]):.3f}")
    print(f"Last 5 avg score:   {np.mean([e['best_score'] for e in training_log[-5:]]):.3f}")
    print(f"\nOutputs:")
    print(f"  livepatch-qwen-1.5b-lora/  (LoRA adapter)")
    print(f"  training_log.json          ({len(training_log)} episodes)")
    print(f"  training_curves.png")
    print(f"  agent_comparison.png")


if __name__ == '__main__':
    main()
