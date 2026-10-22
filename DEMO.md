# BLACK-BOX-AI-DEMO
## Live Application & Demonstration

This page provides access to the live AI-based task scheduler and a recorded demonstration of its probe-detection defence.

### 1. Live Web Application

**[Launch BLACK-BOX-AI-DEMO](https://black-box-ai-demo.onrender.com)**

The deployed application demonstrates the core scheduling and explainability pipeline:

- **Black-box task scheduler:** Assigns incoming tasks to virtual machines (VMs).
- **LOREsa explainability:** Generates local explanations for scheduling decisions.
- **Explanation guard:** Controls whether an explanation can be provided based on the probe detector's assessment.

**How to test the live application:**

1. Open the live website using the link above.
2. Enter task parameters and click **Schedule a Task**.
3. Observe the selected VM and the generated LOREsa explanation.

*Note: The hosted application runs on Render's free instance. The live interface is intended primarily for testing normal task scheduling and explanation generation.*

### 2. Boundary Probe Detection — Recorded Demonstration

https://github.com/user-attachments/assets/83327c43-8498-4ceb-b066-4cf189edf553

The recorded demonstration shows the probe-testing workflow, where task inputs are systematically varied to investigate the scheduler's decision boundaries.

The demonstration illustrates:

- Changes in task inputs during boundary probing.
- The resulting probe-risk score and detection status.
- The explanation guard's response when probing is detected.
- The distinction between returning a scheduling decision and providing an explanation.

#### Why is probe detection demonstrated separately?

The boundary-probe workflow repeatedly submits inputs and invokes computationally expensive LOREsa explanation generation. During testing on Render's free instance, the probe request returned an HTTP 500 error, and the Gunicorn worker was terminated with a possible out-of-memory condition.

Consequently, **the complete boundary-probe simulation is provided as a recorded demonstration rather than being presented as a reliably working live feature on the free-tier deployment.**

This is a hosting-resource limitation observed during testing; it does not, by itself, establish that the probe-detection logic is incorrect.

### 3. System Workflow

1. A task is submitted to the scheduler.
2. The black-box model selects a VM.
3. The probe detector evaluates the input pattern.
4. If an explanation is permitted, LOREsa generates a local explanation.
5. If probing is detected, the explanation guard can withhold the explanation while preserving the scheduling decision.

### 4. Project Scope

This project explores the use of local explainability in black-box scheduling systems and investigates whether systematic input probing can be detected and controlled.

The prototype combines task scheduling, LOREsa-based explanations, probe-risk assessment, and explanation guarding.
