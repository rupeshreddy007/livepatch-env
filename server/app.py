"""Flask server exposing LivePatch as OpenEnv REST API."""
from flask import Flask, request, jsonify
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import LivePatchEnv
from src.config import EnvironmentConfig
from src.tasks import grade_episode

app = Flask(__name__)
env = LivePatchEnv()


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True) or {}
    config = EnvironmentConfig(
        difficulty=data.get("difficulty", "medium"),
        max_steps=data.get("max_steps", 25),
        seed=data.get("seed"),
        traffic_rps=data.get("traffic_rps", 100),
    )
    obs = env.reset(config)
    return jsonify(obs)


@app.route("/step", methods=["POST"])
def step():
    data = request.get_json(silent=True) or {}
    command = data.get("command", "")
    obs = env.step({"command": command})
    return jsonify(obs)


@app.route("/state", methods=["GET"])
def state():
    return jsonify(env.state())


@app.route("/grade", methods=["GET"])
def grade():
    result = grade_episode(env)
    return jsonify({
        "score": result.score,
        "fix_quality": result.fix_quality,
        "uptime": result.uptime,
        "efficiency": result.efficiency,
        "safety": result.safety,
        "faults_injected": result.faults_injected,
        "faults_resolved": result.faults_resolved,
        "steps_used": result.steps_used,
        "total_reward": result.total_reward,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/", methods=["GET"])
def index():
    return """<html><head><title>LivePatch Environment</title></head>
<body style="font-family:sans-serif;max-width:800px;margin:40px auto;padding:20px;">
<h1>LivePatch - PostgreSQL Incident Response Environment</h1>
<p>An OpenEnv-compatible environment for training RL agents to diagnose and fix PostgreSQL database incidents.</p>
<h2>API Endpoints</h2>
<ul>
<li><b>POST /reset</b> - Reset environment with new fault scenario</li>
<li><b>POST /step</b> - Execute a command (JSON body: {"command": "SQL..."})</li>
<li><b>GET /state</b> - Get current environment state</li>
<li><b>GET /grade</b> - Grade the current episode</li>
<li><b>GET /health</b> - Health check</li>
</ul>
<h2>Links</h2>
<ul>
<li><a href="https://github.com/rupeshreddy007/livepatch-env">GitHub Repository</a></li>
<li><a href="https://huggingface.co/spaces/rupeshreddy7/livepatch-training">Training Dashboard</a></li>
</ul>
</body></html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
