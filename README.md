# Revealing Hidden Decision Patterns in Black-Box AI Scheduling

### DEMO (scheduling and probing defence)
[View instructions for live demo and boundary probe simulation](./DEMO.md)

### Investigating LOREsa Explanations, Query Probing, and Explanation Access Control

> A research prototype for examining how a black-box task scheduler makes decisions, generating local explanations with LOREsa, and detecting query patterns that may indicate systematic probing.

## Overview

Machine-learning systems can make decisions that are difficult to inspect from their outputs alone. In task scheduling, an AI model may assign a task to a virtual machine (VM) based on factors such as CPU requirements, memory requirements, priority, deadline, and queue length. A prediction tells us which VM was selected, but not necessarily why.

This project builds a **black-box AI task scheduler** and wraps it in an explanation and query-monitoring workflow. It uses **LOREsa** to generate local explanations for individual scheduling decisions, records incoming queries, analyses recent query patterns for systematic probing, and controls whether an explanation is released.

The research motivation is to investigate whether explanations and observed model behaviour provide a useful picture of the scheduler's decision patterns, and how systematic queries can expose decision boundaries or unusual behaviour. The project is a controlled research prototype, not a production scheduling service or a proven bias-detection system.

## Research question

**How can local explanations and query-pattern monitoring help investigate the decision behaviour of a black-box AI task scheduler, and what are the limitations of restricting explanations when probing is suspected?**

The current implementation focuses on scheduling tasks across four virtual machines. It does not include a recruitment model or a recruitment experiment.

## What the project does

- **Predicts VM assignments:** a Random Forest classifier predicts one of four virtual machines for each task.
- **Uses synthetic scheduling data:** task inputs and reference examples are generated programmatically.
- **Generates local explanations:** an adapter connects the scheduler to LOREsa to request a rule, counterfactuals, fidelity, and feature-importance information for a task.
- **Logs requests:** each scheduling interaction and its decision is recorded in an in-memory query history.
- **Detects probing patterns heuristically:** the detector examines recent queries for systematic feature variation, monotonic scans, decision transitions, repeated inputs, and query volume.
- **Guards explanation release:** the scheduler still returns a decision, while the guard can withhold the explanation when the probe score reaches its configured threshold.
- **Provides a Flask API:** endpoints support task scheduling, resetting the query history, and running a memory-boundary probe.

## Architecture

```text
                 Synthetic Scheduling Data
                           |
                           v
                 Task Scheduler (Random Forest)
                           |
             Task input --> Prediction
                           |
                           v
                     Query Logger
                           |
                           v
                    Probe Detector
                           |
                           v
                    Explanation Guard
                           |
                +----------+----------+
                |                     |
          Explanation allowed?   Probe suspected
                |                     |
                v                     v
           LOREsa adapter       Withhold explanation
                |
                v
       Local rule / counterfactuals
                |
                v
       API response with decision,
       guard status, and explanation
       (when permitted)
```

The explanation guard controls explanation release, not the scheduler's VM assignment. A high probe score is a heuristic signal, not proof of malicious intent. Legitimate testing and auditing may also produce systematic query patterns.

## Scheduling model

### Input features

Each task is represented using five numeric features:

| Feature | Description |
|---|---|
| `cpu_required` | CPU capacity required by the task |
| `memory_required` | Memory required by the task |
| `priority` | Task priority |
| `deadline` | Task deadline value used by the scheduling policy |
| `queue_length` | Current queue length indicator |

### Output

The model predicts one of four VM labels:

- `VM-1`
- `VM-2`
- `VM-3`
- `VM-4`

### How the black-box model is created

`scheduler_data.py` defines a deterministic rule-based scheduling policy and generates synthetic tasks. `scheduler.py` creates training examples from that policy and trains a `RandomForestClassifier` on the generated labels.

The trained Random Forest is the predictor exposed by the application. The policy used to create training labels is available in the source code for reproducibility, but the application's prediction interface is the model's `predict()` method.

This setup makes it possible to study an interpretable synthetic environment. It does not establish that the model is biased, nor does it represent a realistic cloud scheduler in all its complexity.

## LOREsa local explanations

`lore_explainer.py` adapts the scheduler and reference data for the LOREsa library. It:

1. Converts a task input into the expected tabular format.
2. obtains the scheduler's predicted VM assignment.
3. passes the instance to LOREsa to generate a local explanation.
4. returns the prediction together with the explanation fields exposed by the adapter.

The adapter is designed to return:

- `rule` — the local rule associated with the instance.
- `counterfactuals` — alternative instances returned by LOREsa.
- `fidelity` — the fidelity value returned by the explanation library.
- `feature_importances` — feature-importance information returned by the explanation library.

The exact structure and interpretation of these fields depend on the installed LOREsa version. An explanation is a local approximation; it should not automatically be interpreted as a complete account of the model's global behaviour.

## Query monitoring and explanation guard

### Query logger

`query_logger.py` stores recent scheduling requests and related information, including the input task, predicted decision, whether an explanation was requested or released, and the probe score. The current history is held in memory and is cleared when the reset endpoint is called or the application restarts.

### Probe detector

`probe_detector.py` uses heuristic signals from a sliding window of recent queries. These include:

- A feature changing systematically while other features remain relatively stable.
- Monotonic scans of a feature.
- Changes in VM assignment across a query sequence.
- Repeated task inputs.
- Query volume within the observation window.

The detector combines these signals into a probe score and compares the score with a configurable threshold. This is a heuristic, not a trained adversarial-query classifier. Its performance must be measured; false positives and false negatives are possible.

### Explanation guard

`explanation_guard.py` evaluates the detector result and decides whether an explanation should be released. `explanation_service.py` coordinates the scheduler, logger, guard, and LOREsa adapter.

The design intentionally separates the scheduling decision from explanation access: a suspected probe can cause the explanation to be withheld, but the scheduler still returns its VM assignment. This trade-off should be evaluated carefully because it can also prevent legitimate users or auditors from receiving explanations.

## Flask API

The Flask application exposes the following routes.

### `GET /`

Renders the web interface template (`index.html`).

### `POST /api/schedule`

Submits a task for scheduling and requests an explanation. Example request body:

```json
{
  "cpu_required": 4,
  "memory_required": 8,
  "priority": 5,
  "deadline": 200,
  "queue_length": 5
}
```

The response includes the scheduler's `decision`, `probe_score`, `is_probe`, `explanation_allowed`, and detector `signals`. The `explanation` field contains the LOREsa result when an explanation is released; otherwise it is `null`.

### `POST /api/reset`

Clears the in-memory query history for the current service session.

### `POST /api/probe`

Runs a sequence of tasks while varying `memory_required` over a configurable range. Optional JSON fields:

```json
{
  "start": 6,
  "end": 13
}
```

For each tested memory value, the endpoint returns the scheduling decision, probe score, probe flag, and whether the explanation was allowed. This endpoint is intended for controlled local experimentation with boundary-style queries.

## Repository modules

| Module | Responsibility |
|---|---|
| `app.py` | Flask app and HTTP endpoints |
| `scheduler.py` | Trains and exposes the Random Forest task scheduler |
| `scheduler_data.py` | Defines scheduling features, synthetic task generation, and the label-generating policy |
| `lore_explainer.py` | Adapts the scheduler and tabular data to LOREsa |
| `explanation_service.py` | Coordinates prediction, logging, detection, and explanation release |
| `query_logger.py` | Stores the in-memory query history |
| `probe_detector.py` | Scores query sequences using heuristic probe signals |
| `explanation_guard.py` | Applies the explanation-release decision |
| `explainer.py` | Separate simplified local decision-tree explainer retained in the repository; it uses candidate age/proficiency fields and is not part of the scheduler API path |
| `data_generator.py`, `models.py`, `scaffold.py` | Separate candidate/recruitment experiment modules, not used by the current scheduler web-app path |

The scheduler workflow described in this README is the focus of the application. The candidate/recruitment files are not combined with the scheduler pipeline.

## Getting started

### Requirements

The project uses Python and the following core libraries:

- Flask
- NumPy
- pandas
- scikit-learn
- LOREsa (`lore_sa`)

The compatible LOREsa package version and installation method may depend on the environment. Install the version required by your project setup. If the repository has a `requirements.txt`, use its pinned dependencies.

### Create a virtual environment

```bash
python -m venv .venv
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### Install dependencies

Install the compatible LOREsa package separately if it is not already available in your environment. If your project contains a `requirements.txt`, prefer:

```bash
pip install -r requirements.txt
```

To install LORE_sa (whenever necessary) in .venv, type command: 
```
python -m pip install -e ./lore_sa
``` 

### Run the application

Run `app.py` from the project:

```bash
python -m web/app.py
```

The app is configured for local development at `http://127.0.0.1:5000`.

## Evaluation plan

The following are suggested evaluation measures.

| Measure | What it investigates |
|---|---|
| Prediction consistency | Whether repeated identical inputs produce the same VM assignment |
| Feature sensitivity | Whether changing one task feature while holding others fixed changes the VM assignment |
| Local fidelity | How closely the local explanation represents the scheduler in its local evaluation region |
| Explanation stability | Whether explanations remain similar across repeated runs or neighbourhood samples |
| Probe-detection precision and recall | How well the heuristic identifies labelled probe sequences, including false positives and false negatives |
| Explanation availability | How often explanations are released or withheld under normal and systematic query patterns |
| Boundary behaviour | Where VM assignments change as one feature is varied across a range |
| Runtime | Time taken for scheduling, detection, and explanation generation |

When evaluating the detector, construct both systematic probe sequences and ordinary task sequences. Define the criteria before measuring performance. A detector that flags systematic variation may also flag legitimate debugging or fairness audits.

## Limitations and responsible use

- Tasks and training labels are synthetic and generated from a simple policy.
- The Random Forest approximates that policy; its behaviour may differ from the label-generating rules.
- LOREsa explanations are local and depend on the data, configuration, and installed library version.
- The probe detector is heuristic and has not, by itself, demonstrated robust detection of adaptive or realistic probing.
- Query history is held in memory; the current implementation is not a persistent monitoring or multi-user session system.
- Withholding explanations can inconvenience legitimate users and auditors.
- Probe detection does not prove intent, bias, deception, or malicious activity.
- A scheduling disparity or feature sensitivity is a signal for investigation, not proof of unfairness.
- The prototype has not been validated as a production scheduler or a general-purpose fairness auditor.

Use this project for research and controlled experimentation. Do not rely on it for real infrastructure scheduling or other consequential decisions without appropriate testing, security review, monitoring, and domain-specific validation.

## Future work

- Evaluate LOREsa explanations against the scheduler's actual predictions on held-out, domain-valid task samples.
- Measure detector precision, recall, false-positive rates, and false-negative rates across normal and systematic query sequences.
- Compare explanation availability under benign usage, automated tests, and boundary scans.
- Add repeatable experiment scripts, tests, and structured result outputs.
- Improve session-aware query tracking and make logging configurable.
- Investigate how to support legitimate auditing without unnecessarily withholding explanations.
- Examine feature sensitivity and decision boundaries across the scheduler's valid input ranges.

## Research contribution

This project provides a controlled environment for studying the interaction between a black-box task scheduler, LOREsa-based local explanations, query-pattern monitoring, and explanation-release control. Its central focus is not merely to predict a VM assignment, but to investigate how explanations and systematic probing can help reviewers examine model behaviour—and how heuristic safeguards may affect access to those explanations.

## References / Citations
- Guidotti, R., Monreale, A., Ruggieri, S., Pedreschi, D., Turini, F., & Giannotti, F. (2019). Factual and Counterfactual Explanations for Black Box Decision Making. IEEE Intelligent Systems, 34(6), 14-23.
- Guidotti, R., Monreale, A., Ruggieri, S., Naretto, F., Turini, F., Pedreschi, D., & Giannotti, F. (2022). Stable and actionable explanations of black-box models through factual and counterfactual rules. Data Mining and Knowledge Discovery.
- Guidotti, R., Monreale, A., Ruggieri, S., Pedreschi, D., Turini, F., & Giannotti, F. (2018). Local rule-based explanations of black box decision systems. arXiv preprint arXiv:1805.10820. https://arxiv.org/abs/1805.10820

Github repo: https://github.com/kdd-lab/LORE_sa
