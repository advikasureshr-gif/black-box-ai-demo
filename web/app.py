import sys
from pathlib import Path

# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# IMPORTS
# ==========================================================

from flask import Flask, render_template, request, jsonify

from src.scheduler import TaskScheduler
from src.scheduler_data import generate_tasks
from src.explanation_service import ExplanationService


# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)


# ==========================================================
# INITIALIZE SCHEDULER + EXPLANATION SERVICE
# ==========================================================

reference_data = generate_tasks(
    n=1000,
    random_state=42,
)

scheduler = TaskScheduler()

service = ExplanationService(
    scheduler=scheduler,
    reference_data=reference_data,
    num_instances=500,
)


# ==========================================================
# MAIN PAGE
# ==========================================================

@app.route("/")
def index():
    return render_template("index.html")


# ==========================================================
# SCHEDULE A TASK
# ==========================================================

@app.route("/api/schedule", methods=["POST"])
def schedule():

    task = request.json

    result = service.process_task(
        task=task,
        explanation_requested=True,
    )

    return jsonify(result)


# ==========================================================
# RESET QUERY SESSION
# ==========================================================

@app.route("/api/reset", methods=["POST"])
def reset():

    service.logger.clear()

    return jsonify({
        "status": "reset"
    })


# ==========================================================
# RUN BOUNDARY PROBE
# ==========================================================

@app.route("/api/probe", methods=["POST"])
def probe():

    data = request.json or {}

    start = int(data.get("start", 6))
    end = int(data.get("end", 13))

    results = []

    for memory in range(start, end + 1):

        task = {
            "cpu_required": 4,
            "memory_required": memory,
            "priority": 5,
            "deadline": 200,
            "queue_length": 5,
        }

        result = service.process_task(
            task=task,
            explanation_requested=True,
        )

        results.append({
            "memory": memory,
            "decision": result["decision"],
            "probe_score": result["probe_score"],
            "is_probe": result["is_probe"],
            "explanation_allowed": (
                result["explanation_allowed"]
            ),
        })

    return jsonify({
        "results": results
    })


# ==========================================================
# RUN SERVER
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000,
    )