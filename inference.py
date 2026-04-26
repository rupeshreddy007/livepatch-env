"""HuggingFace-compatible inference script for LivePatch environment."""
import os
import sys
import json
import time
from typing import Dict, Any

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment import LivePatchEnv, SYSTEM_PROMPT
from src.config import EnvironmentConfig
from src.tasks import TASKS, grade_episode


SYSTEM_MSG = SYSTEM_PROMPT + """

IMPORTANT: Respond with ONLY the psql command to execute. No explanations.
When you are done fixing the issue, respond with: submit
"""


def run_inference():
    """Run inference with an LLM agent against the environment."""
    from openai import OpenAI

    api_key = os.environ.get("HF_TOKEN", os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    client = OpenAI(api_key=api_key, base_url=base_url)

    for task_name in ["easy", "medium", "hard"]:
        config = TASKS[task_name]
        env = LivePatchEnv(config)
        obs = env.reset()

        print(f"[START] task={task_name} difficulty={config.difficulty}")
        history = [{"role": "system", "content": SYSTEM_MSG}]

        step = 0
        while not obs["done"]:
            traffic = obs.get("traffic", {})
            user_msg = obs["observation"]
            user_msg += f"\n\n[Traffic] OK:{traffic.get('requests_ok',0)} "
            user_msg += f"Failed:{traffic.get('requests_failed',0)} "
            user_msg += f"p99:{traffic.get('p99_latency_ms',0):.1f}ms"
            user_msg += f"\n[Step {obs['step']}/{obs['max_steps']}]"

            history.append({"role": "user", "content": user_msg})

            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=history,
                    max_tokens=200,
                    temperature=0.2,
                )
                command = resp.choices[0].message.content.strip().strip("`")
                # Take first non-comment line
                for line in command.split("\n"):
                    line = line.strip().strip("`")
                    if line and not line.startswith(("#", "//", "--")):
                        command = line
                        break
            except Exception as e:
                print(f"[ERROR] LLM call failed: {e}", file=sys.stderr)
                command = "submit"

            history.append({"role": "assistant", "content": command})

            print(f"[STEP] task={task_name} step={step} command={json.dumps(command)} "
                  f"reward={obs.get('reward', 0)}")

            obs = env.step({"command": command})
            step += 1

        result = grade_episode(env)
        print(f"[END] task={task_name} score={result.score} "
              f"fix_quality={result.fix_quality} uptime={result.uptime} "
              f"efficiency={result.efficiency} safety={result.safety} "
              f"reward={result.total_reward}")


if __name__ == "__main__":
    run_inference()
