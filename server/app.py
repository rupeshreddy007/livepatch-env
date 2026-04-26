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
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LivePatch - PostgreSQL Incident Response Environment</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0a0e1a; color: #c9d1d9; min-height: 100vh; }
  .header { background: #0d1117; border-bottom: 1px solid #21262d; padding: 48px 20px 36px; }
  .header-inner { max-width: 960px; margin: 0 auto; }
  .tag { display: inline-block; background: #1c2333; color: #7ee787; border: 1px solid #238636; padding: 3px 10px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 14px; }
  .header h1 { font-size: 2rem; font-weight: 600; color: #f0f6fc; margin-bottom: 6px; letter-spacing: -0.5px; }
  .header p { font-size: 1rem; color: #8b949e; max-width: 640px; line-height: 1.5; }
  .nav-links { display: flex; gap: 12px; margin-top: 20px; flex-wrap: wrap; }
  .nav-links a { color: #8b949e; background: #161b22; border: 1px solid #30363d; padding: 7px 16px; border-radius: 6px; text-decoration: none; font-size: 0.82rem; font-weight: 500; transition: all 0.15s; }
  .nav-links a:hover { color: #f0f6fc; border-color: #58a6ff; }
  .metrics-bar { display: flex; gap: 1px; background: #21262d; border-bottom: 1px solid #21262d; }
  .metric { flex: 1; text-align: center; padding: 20px 12px; background: #0d1117; }
  .metric .val { font-size: 1.6rem; font-weight: 700; color: #f0f6fc; font-variant-numeric: tabular-nums; }
  .metric .lbl { font-size: 0.7rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 4px; }
  .main { max-width: 960px; margin: 0 auto; padding: 32px 20px; }
  .section { margin-bottom: 32px; }
  .section-header { font-size: 0.8rem; font-weight: 600; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 14px; padding-bottom: 8px; border-bottom: 1px solid #21262d; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 700px) { .grid-2 { grid-template-columns: 1fr; } .metrics-bar { flex-wrap: wrap; } .metric { min-width: 50%; } }
  .panel { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; overflow: hidden; }
  .panel-head { background: #161b22; padding: 10px 16px; border-bottom: 1px solid #21262d; font-size: 0.82rem; font-weight: 600; color: #f0f6fc; }
  .panel-body { padding: 16px; }
  .tbl { width: 100%; border-collapse: collapse; }
  .tbl th, .tbl td { padding: 8px 12px; text-align: left; font-size: 0.82rem; border-bottom: 1px solid #21262d; }
  .tbl th { color: #8b949e; font-weight: 600; }
  .tbl td { color: #c9d1d9; }
  .tbl tr:last-child td { border-bottom: none; }
  .tbl .good { color: #7ee787; font-weight: 600; }
  .item-list { list-style: none; padding: 0; }
  .item-list li { padding: 8px 0; border-bottom: 1px solid #21262d; font-size: 0.82rem; color: #c9d1d9; }
  .item-list li:last-child { border-bottom: none; }
  .item-list li strong { color: #f0f6fc; }
  .item-list code { background: #161b22; color: #79c0ff; padding: 2px 6px; border-radius: 3px; font-size: 0.78rem; }
  .demo-section { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; overflow: hidden; margin-bottom: 32px; }
  .demo-head { background: #161b22; padding: 10px 16px; border-bottom: 1px solid #21262d; font-size: 0.82rem; font-weight: 600; color: #f0f6fc; }
  .demo-body { padding: 16px; }
  .demo-controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-end; }
  .form-group { display: flex; flex-direction: column; gap: 4px; }
  .form-group label { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
  .form-group select, .form-group input { background: #0a0e1a; border: 1px solid #30363d; color: #c9d1d9; padding: 7px 10px; border-radius: 4px; font-size: 0.82rem; font-family: inherit; }
  .form-group input:focus { outline: none; border-color: #58a6ff; }
  .cmd-field { flex: 1; min-width: 180px; }
  .btn { border: none; padding: 7px 16px; border-radius: 4px; font-size: 0.82rem; font-weight: 600; cursor: pointer; font-family: inherit; transition: background 0.15s; }
  .btn-primary { background: #238636; color: #fff; }
  .btn-primary:hover { background: #2ea043; }
  .btn-secondary { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; }
  .btn-secondary:hover { background: #30363d; }
  .btn-run { background: #1f6feb; color: #fff; }
  .btn-run:hover { background: #388bfd; }
  #output { background: #0a0e1a; border: 1px solid #21262d; border-radius: 4px; padding: 14px; margin-top: 14px; font-family: 'Cascadia Code', 'SF Mono', 'Fira Code', monospace; font-size: 0.78rem; color: #79c0ff; white-space: pre-wrap; max-height: 360px; overflow-y: auto; display: none; line-height: 1.5; }
  .footer { text-align: center; padding: 24px 20px; color: #484f58; font-size: 0.75rem; border-top: 1px solid #21262d; margin-top: 20px; }
</style>
</head>
<body>
<div class="header">
  <div class="header-inner">
    <div class="tag">OpenEnv Hackathon 2026</div>
    <h1>LivePatch</h1>
    <p>A reinforcement learning environment for training agents to diagnose and remediate PostgreSQL database incidents under simulated production traffic.</p>
    <div class="nav-links">
      <a href="https://github.com/rupeshreddy007/livepatch-env">Source Code</a>
      <a href="https://huggingface.co/spaces/rupeshreddy7/livepatch-env/blob/main/BLOG.md">Technical Writeup</a>
      <a href="https://huggingface.co/spaces/rupeshreddy7/livepatch-env/blob/main/training/livepatch_grpo_training.ipynb">Training Notebook</a>
      <a href="https://huggingface.co/spaces/rupeshreddy7/livepatch-training">Training Dashboard</a>
    </div>
  </div>
</div>

<div class="metrics-bar">
  <div class="metric"><div class="val">3</div><div class="lbl">Fault Types</div></div>
  <div class="metric"><div class="val">15</div><div class="lbl">Step Budget</div></div>
  <div class="metric"><div class="val">1.5B</div><div class="lbl">Parameters</div></div>
  <div class="metric"><div class="val">0.673</div><div class="lbl">Best Score</div></div>
  <div class="metric"><div class="val">22</div><div class="lbl">Episodes</div></div>
</div>

<div class="main">
  <div class="demo-section">
    <div class="demo-head">Interactive Console</div>
    <div class="demo-body">
      <div class="demo-controls">
        <div class="form-group">
          <label>Difficulty</label>
          <select id="difficulty">
            <option value="easy">Easy</option>
            <option value="medium" selected>Medium</option>
            <option value="hard">Hard</option>
          </select>
        </div>
        <div class="form-group">
          <label>&nbsp;</label>
          <button class="btn btn-secondary" onclick="resetEnv()">Initialize</button>
        </div>
        <div class="form-group cmd-field">
          <label>Command</label>
          <input type="text" id="command" placeholder="SELECT * FROM pg_stat_activity;" onkeydown="if(event.key==='Enter')sendCmd()">
        </div>
        <div class="form-group">
          <label>&nbsp;</label>
          <button class="btn btn-run" onclick="sendCmd()">Execute</button>
        </div>
        <div class="form-group">
          <label>&nbsp;</label>
          <button class="btn btn-primary" onclick="submitEp()">Submit</button>
        </div>
      </div>
      <div id="output"></div>
    </div>
  </div>

  <div class="section">
    <div class="section-header">Training Results</div>
    <div class="grid-2">
      <div class="panel">
        <div class="panel-head">Best Performance (Episode 19)</div>
        <div class="panel-body">
          <table class="tbl">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Overall Score</td><td class="good">0.673</td></tr>
            <tr><td>Fix Quality</td><td>0.421</td></tr>
            <tr><td>Uptime</td><td>0.721</td></tr>
            <tr><td>Safety</td><td class="good">1.000</td></tr>
          </table>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head">Training Milestones</div>
        <div class="panel-body">
          <ul class="item-list">
            <li><strong>Episode 0</strong> -- First valid SQL command generated</li>
            <li><strong>Episode 5</strong> -- Safety score reached 1.0</li>
            <li><strong>Episode 17</strong> -- First fault resolved (missing_index)</li>
            <li><strong>Episode 19</strong> -- Peak score: 0.673</li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-header">Environment Specification</div>
    <div class="grid-2">
      <div class="panel">
        <div class="panel-head">REST API</div>
        <div class="panel-body">
          <ul class="item-list">
            <li><code>POST /reset</code> -- Initialize new incident scenario</li>
            <li><code>POST /step</code> -- Execute SQL command</li>
            <li><code>GET /state</code> -- Retrieve current state</li>
            <li><code>GET /grade</code> -- Evaluate episode performance</li>
            <li><code>GET /health</code> -- Service health check</li>
          </ul>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head">Supported Faults</div>
        <div class="panel-body">
          <ul class="item-list">
            <li><strong>Missing Index</strong> -- Resolve with CREATE INDEX CONCURRENTLY</li>
            <li><strong>Table Bloat</strong> -- Resolve with VACUUM ANALYZE</li>
            <li><strong>Stale Statistics</strong> -- Resolve with ANALYZE</li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-header">Training Configuration</div>
    <div class="panel">
      <div class="panel-body">
        <table class="tbl">
          <tr><th>Parameter</th><th>Value</th></tr>
          <tr><td>Base Model</td><td>Qwen2.5-1.5B-Instruct</td></tr>
          <tr><td>Training Method</td><td>SFT (3 epochs, 51 demos) + GRPO (50 episodes)</td></tr>
          <tr><td>Adaptation</td><td>LoRA r=32, 36.9M trainable parameters</td></tr>
          <tr><td>Quantization</td><td>4-bit via Unsloth</td></tr>
          <tr><td>GRPO Group Size</td><td>8 (same-seed comparison)</td></tr>
          <tr><td>Hardware</td><td>NVIDIA A100-SXM4-80GB</td></tr>
          <tr><td>Episode Duration</td><td>~80 seconds</td></tr>
        </table>
      </div>
    </div>
  </div>
</div>

<div class="footer">
  LivePatch -- Built for the OpenEnv Hackathon by rupeshreddy7
</div>

<script>
const out = document.getElementById('output');
function show(data) { out.style.display = 'block'; out.textContent = JSON.stringify(data, null, 2); }
async function resetEnv() {
  try {
    const r = await fetch('/reset', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({difficulty: document.getElementById('difficulty').value, max_steps: 15}) });
    show(await r.json());
  } catch(e) { show({error: e.message}); }
}
async function sendCmd() {
  const cmd = document.getElementById('command').value;
  if (!cmd) return;
  try {
    const r = await fetch('/step', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({command: cmd}) });
    show(await r.json());
    document.getElementById('command').value = '';
  } catch(e) { show({error: e.message}); }
}
async function submitEp() {
  try {
    const r = await fetch('/step', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({command: 'submit'}) });
    show(await r.json());
  } catch(e) { show({error: e.message}); }
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
