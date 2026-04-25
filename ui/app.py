"""LivePatch — web UI for interactive demos."""
from flask import Flask, render_template, request, jsonify
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import LivePatchEnv
from src.config import EnvironmentConfig
from src.tasks import grade_episode

app = Flask(__name__)
env = LivePatchEnv()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reset", methods=["POST"])
def api_reset():
    data = request.get_json(silent=True) or {}
    config = EnvironmentConfig(
        difficulty=data.get("difficulty", "medium"),
        max_steps=data.get("max_steps", 25),
        seed=data.get("seed"),
        traffic_rps=data.get("traffic_rps", 100),
    )
    obs = env.reset(config)
    return jsonify(obs)


@app.route("/api/step", methods=["POST"])
def api_step():
    data = request.get_json(silent=True) or {}
    command = data.get("command", "")
    obs = env.step({"command": command})
    return jsonify(obs)


@app.route("/api/state", methods=["GET"])
def api_state():
    return jsonify(env.state())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
