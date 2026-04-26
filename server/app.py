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
<title>LivePatch - PostgreSQL Incident Response</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
  .hero { background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1a1a2e 100%); padding: 60px 20px 40px; text-align: center; border-bottom: 1px solid #334155; }
  .hero h1 { font-size: 2.5rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px; }
  .hero h1 span { color: #38bdf8; }
  .hero .subtitle { font-size: 1.1rem; color: #94a3b8; max-width: 600px; margin: 0 auto; }
  .badge { display: inline-block; background: #164e63; color: #67e8f9; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; margin-bottom: 16px; }
  .container { max-width: 1000px; margin: 0 auto; padding: 40px 20px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 40px; }
  @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 24px; transition: border-color 0.2s; }
  .card:hover { border-color: #38bdf8; }
  .card h3 { color: #f8fafc; font-size: 1.1rem; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
  .card h3 .icon { font-size: 1.3rem; }
  .card p, .card li { color: #94a3b8; font-size: 0.9rem; line-height: 1.6; }
  .card ul { list-style: none; padding: 0; }
  .card ul li { padding: 6px 0; border-bottom: 1px solid #334155; }
  .card ul li:last-child { border-bottom: none; }
  .card ul li code { background: #0f172a; color: #38bdf8; padding: 2px 6px; border-radius: 4px; font-size: 0.85rem; }
  .results-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
  .results-table th, .results-table td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #334155; font-size: 0.85rem; }
  .results-table th { color: #38bdf8; font-weight: 600; }
  .results-table td { color: #cbd5e1; }
  .results-table tr:hover td { background: #1a2744; }
  .highlight { color: #4ade80; font-weight: 600; }
  .section-title { font-size: 1.4rem; color: #f8fafc; margin-bottom: 20px; font-weight: 600; }
  .demo-box { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 24px; margin-bottom: 40px; }
  .demo-box h3 { color: #f8fafc; margin-bottom: 16px; }
  .demo-form { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end; }
  .demo-form label { color: #94a3b8; font-size: 0.85rem; display: block; margin-bottom: 4px; }
  .demo-form select, .demo-form input, .demo-form button { background: #0f172a; border: 1px solid #475569; color: #e2e8f0; padding: 8px 12px; border-radius: 6px; font-size: 0.9rem; }
  .demo-form button { background: #0ea5e9; border: none; color: white; font-weight: 600; cursor: pointer; padding: 8px 20px; }
  .demo-form button:hover { background: #0284c7; }
  .cmd-input { flex: 1; min-width: 200px; }
  #output { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-top: 16px; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.8rem; color: #67e8f9; white-space: pre-wrap; max-height: 400px; overflow-y: auto; display: none; }
  .links { display: flex; gap: 16px; justify-content: center; margin-top: 24px; flex-wrap: wrap; }
  .links a { display: inline-flex; align-items: center; gap: 6px; background: #1e293b; border: 1px solid #334155; color: #e2e8f0; padding: 10px 20px; border-radius: 8px; text-decoration: none; font-size: 0.9rem; font-weight: 500; transition: all 0.2s; }
  .links a:hover { border-color: #38bdf8; color: #38bdf8; }
  .stats { display: flex; gap: 32px; justify-content: center; margin: 32px 0; flex-wrap: wrap; }
  .stat { text-align: center; }
  .stat .num { font-size: 2rem; font-weight: 700; color: #38bdf8; }
  .stat .label { font-size: 0.8rem; color: #64748b; margin-top: 4px; }
</style>
</head>
<body>
<div class="hero">
  <div class="badge">OpenEnv Hackathon 2026</div>
  <h1><span>LivePatch</span> Environment</h1>
  <p class="subtitle">Train RL agents to diagnose and fix PostgreSQL database incidents under live production traffic</p>
  <div class="stats">
    <div class="stat"><div class="num">3</div><div class="label">Fault Types</div></div>
    <div class="stat"><div class="num">15</div><div class="label">Step Budget</div></div>
    <div class="stat"><div class="num">1.5B</div><div class="label">Model Params</div></div>
    <div class="stat"><div class="num">0.67</div><div class="label">Best Score</div></div>
  </div>
  <div class="links">
    <a href="https://github.com/rupeshreddy007/livepatch-env">GitHub</a>
    <a href="https://huggingface.co/spaces/rupeshreddy7/livepatch-env/blob/main/BLOG.md">Blog Post</a>
    <a href="https://huggingface.co/spaces/rupeshreddy7/livepatch-env/blob/main/training/livepatch_grpo_training.ipynb">Training Notebook</a>
    <a href="https://huggingface.co/spaces/rupeshreddy7/livepatch-training">Training Dashboard</a>
  </div>
</div>

<div class="container">
  <div class="demo-box">
    <h3>Try It Live</h3>
    <div class="demo-form">
      <div>
        <label>Difficulty</label>
        <select id="difficulty">
          <option value="easy">Easy</option>
          <option value="medium" selected>Medium</option>
          <option value="hard">Hard</option>
        </select>
      </div>
      <div>
        <label>&nbsp;</label>
        <button onclick="resetEnv()">Reset Environment</button>
      </div>
      <div class="cmd-input">
        <label>SQL Command</label>
        <input type="text" id="command" placeholder="e.g. EXPLAIN ANALYZE SELECT * FROM orders;" onkeydown="if(event.key==='Enter')sendCmd()">
      </div>
      <div>
        <label>&nbsp;</label>
        <button onclick="sendCmd()">Execute</button>
      </div>
      <div>
        <label>&nbsp;</label>
        <button onclick="submitEp()" style="background:#059669;">Submit</button>
      </div>
    </div>
    <div id="output"></div>
  </div>

  <h2 class="section-title">Training Results</h2>
  <div class="grid">
    <div class="card">
      <h3><span class="icon">&#x1f4ca;</span> Best Episode (Ep 19)</h3>
      <table class="results-table">
        <tr><th>Metric</th><th>Score</th></tr>
        <tr><td>Overall Score</td><td class="highlight">0.673</td></tr>
        <tr><td>Fix Quality</td><td>0.421</td></tr>
        <tr><td>Uptime</td><td>0.721</td></tr>
        <tr><td>Safety</td><td>1.000</td></tr>
      </table>
    </div>
    <div class="card">
      <h3><span class="icon">&#x1f3af;</span> Key Milestones</h3>
      <ul>
        <li><b>Ep 0:</b> First valid SQL generated</li>
        <li><b>Ep 17:</b> First fault resolved (missing_index)</li>
        <li><b>Ep 19:</b> Best score achieved (0.673)</li>
        <li><b>Ep 5:</b> Safety score reached 1.0</li>
      </ul>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h3><span class="icon">&#x2699;</span> API Endpoints</h3>
      <ul>
        <li><code>POST /reset</code> Reset with new fault scenario</li>
        <li><code>POST /step</code> Execute SQL command</li>
        <li><code>GET /state</code> Current environment state</li>
        <li><code>GET /grade</code> Grade current episode</li>
        <li><code>GET /health</code> Health check</li>
      </ul>
    </div>
    <div class="card">
      <h3><span class="icon">&#x1f6e0;</span> Fault Types</h3>
      <ul>
        <li><b>Missing Index</b> &mdash; CREATE INDEX CONCURRENTLY</li>
        <li><b>Table Bloat</b> &mdash; VACUUM ANALYZE</li>
        <li><b>Stale Statistics</b> &mdash; ANALYZE tablename</li>
      </ul>
    </div>
  </div>

  <h2 class="section-title">Training Configuration</h2>
  <div class="card" style="margin-bottom:40px;">
    <table class="results-table">
      <tr><th>Parameter</th><th>Value</th></tr>
      <tr><td>Model</td><td>Qwen2.5-1.5B-Instruct</td></tr>
      <tr><td>Method</td><td>SFT (3 epochs) + GRPO (50 episodes)</td></tr>
      <tr><td>LoRA</td><td>r=32, 36.9M trainable params</td></tr>
      <tr><td>Quantization</td><td>4-bit (Unsloth)</td></tr>
      <tr><td>Group Size</td><td>8 (same-seed comparison)</td></tr>
      <tr><td>GPU</td><td>NVIDIA A100-SXM4-80GB</td></tr>
      <tr><td>Episode Time</td><td>~80 seconds</td></tr>
    </table>
  </div>
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
