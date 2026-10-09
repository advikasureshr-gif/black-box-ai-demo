# Revealing Hidden Bias in Black-Box AI

### Investigating Scaffolding Attacks, Local Explanations, and Hidden Decision Patterns

> A research prototype for investigating whether a black-box AI system's observable explanations faithfully reflect its ordinary prediction behaviour.

## Overview

Black-box AI systems are used to make or support decisions, but their internal decision processes may be difficult to inspect. Local explainability methods attempt to make these systems more understandable by approximating their behaviour around a particular input. However, an explanation can be misleading if a system behaves differently when it suspects that it is being examined.

This project explores how **systematic probing, local rule-based explanations, and query-pattern monitoring can help reveal discrepancies between a model's apparent behaviour and its ordinary predictions**. It investigates a scaffolding-style scenario in which a wrapper attempts to recognise explanation probes and return benign predictions during probing, while using a different model for ordinary inputs.

The project is **not limited to recruitment**. Hiring is one controlled example used to demonstrate an age-related decision disparity. The broader research idea can be adapted to investigate potential bias or unexpected decision patterns in other black-box AI systems, including lending, education, healthcare, content moderation, insurance, and resource allocation. Applying it to another domain requires suitable data, features, model interfaces, and evaluation criteria; those domain-specific pipelines are not all implemented here.

This repository is a proof of concept, not a production-ready fairness auditor. Its synthetic examples can demonstrate a method under controlled conditions, but they do not establish that any real-world system is biased or fair.

## The central question

**Can a model conceal a known decision pattern from a local explainer by behaving differently when it detects explanation queries, and can systematic testing help reveal the discrepancy?**

The recruitment case study makes this question concrete: if two synthetic candidates have the same technical proficiency but different ages, does the model treat them differently? If the system behaves differently when queried by an explainer, does the explanation reflect the model's ordinary behaviour?

The same investigation pattern can be adapted to other domains by defining the relevant inputs, outcomes, groups or conditions, and domain-appropriate metrics.

## Why this matters

A user may only see a final prediction, such as:

- **Hiring:** shortlisted / not shortlisted
- **Lending:** approved / rejected
- **Education:** score or assessment outcome
- **Content moderation:** allowed / flagged
- **Resource allocation:** assigned / not assigned

A local explanation may offer readable rules for one input, but reviewers still need to ask:

1. Does the explanation approximate the model that actually serves ordinary predictions?
2. Do outcomes change when one feature is varied while other relevant features are held constant?
3. Are explanations stable across reasonable neighbourhood samples and repeated runs?
4. Does the system appear to behave differently during explanation probing?
5. Could probe-detection logic block legitimate auditors or ordinary users?

A difference in outcomes is a signal to investigate, not automatic proof of discrimination. Fairness assessments require appropriate context, metrics, and domain expertise.

## Current implementation

The supplied code contains two related experimental paths. They share the broad motivation of investigating black-box behaviour, explanations, and probing, but **they are not yet integrated into one domain-independent end-to-end pipeline**.

### 1. Synthetic recruitment-bias experiment

`data_generator.py` generates synthetic candidate profiles with:

- `age` — a candidate's age
- `proficiency` — a simulated technical proficiency score
- `fair_hired` — a baseline hiring label
- `biased_hired` — a deliberately age-sensitive hiring label

The baseline and age-sensitive labels make it possible to construct a controlled example in which technical proficiency is isolated from age. No real applicant data is required.

`models.py` trains two logistic-regression models:

- **Baseline model:** uses technical proficiency as its input feature.
- **Age-sensitive model:** uses age and technical proficiency.

`scaffold.py` defines a `ScaffoldedRecruitmentModel` wrapper that routes ordinary predictions to the age-sensitive model and inputs marked as probes to the baseline model. This demonstrates the concept of selective behaviour during probing; it does not by itself prove that the detector can identify realistic or adaptive explanation queries.

### 2. Local explanation and query-monitoring path

The explanation and monitoring modules explore how a local surrogate and query history can be used in an investigation:

- `explainer.py` generates a local neighbourhood, queries a predictor, fits a shallow decision tree, and extracts readable rules with a local-fidelity estimate.
- `lore_explainer.py` adapts a predictor and tabular reference data for the LOREsa library.
- `query_logger.py` records query history and associated prediction/explanation information.
- `probe_detector.py` uses heuristic signals in query patterns to calculate a probe score.
- `explanation_guard.py` uses probe-detection output to control explanation release.
- `explanation_service.py` coordinates prediction, logging, probe detection, and explanation release.

The Flask application in `app.py` currently initialises the task-scheduling example and its explanation service. `scheduler.py` and `scheduler_data.py` use task-scheduling features such as CPU requirements, memory, priority, deadline, and queue length, with `assigned_vm` as the target. Therefore, the current web-app path is **a scheduling demo**, not a fully integrated recruitment-bias web application.

## Research workflow

The intended investigation can be described in the following stages:

```text
Choose a black-box model and define an audit question
                         |
                         v
              Generate or collect inputs
                         |
                         v
          Query the model and log outcomes
                         |
                         v
        Vary relevant features systematically
                         |
                         v
       Generate local explanations if supported
                         |
                         v
 Compare explanations with ordinary predictions
                         |
                         v
     Evaluate patterns, uncertainty, and limitations
```

This is the general research workflow, not a claim that every stage is currently connected through one executable pipeline in this repository.

## Methodology for the recruitment case study

### 1. Synthetic dataset

Candidate age and technical proficiency are generated for a controlled experiment. Synthetic data avoids using identifiable applicant information and makes the intended relationship easier to inspect. The experiment intentionally keeps the feature set small; real recruitment decisions involve many additional factors and legal considerations.

### 2. Baseline and age-sensitive models

A proficiency-only model serves as a reference, while a model using age and proficiency represents a deliberately age-sensitive alternative. Candidate pairs with equal proficiency but different ages can be used to examine whether age changes the prediction.

### 3. Scaffolding-style wrapper

The wrapper simulates a system that routes probe-labelled inputs to a baseline model and ordinary inputs to an age-sensitive model. This creates a controlled discrepancy to investigate. The current implementation should be understood as a limited proof of concept: probe recognition depends on how the query is identified, and this is not evidence that a real system could reliably recognise a sophisticated explainer.

### 4. Local rule-based explanation

The simplified explainer:

1. Takes one input as the instance to explain.
2. Generates nearby samples.
3. Queries the selected predictor for their outputs.
4. Fits a shallow decision tree to approximate those local outputs.
5. Extracts readable decision rules and estimates local fidelity.

This is **LORE-inspired**, not a complete reproduction of the original LORE algorithm or its full counterfactual-generation procedure.

### 5. Query-pattern monitoring

The query-monitoring modules look for patterns such as systematic feature variation, monotonic scans, repeated queries, decision transitions, and high query volume. Such signals can help identify unusual probing behaviour, but they can also arise from legitimate audits, testing, or normal usage. A probe score is not proof of malicious intent.

## Evaluation ideas

The following metrics are useful research directions. They should be reported only where the relevant experiment has actually been run.

| Metric | What it investigates |
|---|---|
| Selection-rate difference | Whether outcome rates differ between defined groups in the synthetic recruitment example |
| Feature sensitivity | Whether changing one feature while holding others fixed changes a prediction |
| Local fidelity | How closely a local surrogate matches the queried model on held-out neighbourhood samples |
| Explanation stability | Whether explanations change substantially across reasonable samples or random seeds |
| Probe-detection precision and recall | How well the detector identifies labelled probes, including false positives and false negatives |
| Attack success rate | How often an explanation conceals a known pattern while ordinary predictions still exhibit it, under a predefined criterion |
| Runtime | Approximate cost of data generation, querying, explanation, and evaluation |

For the scaffolding experiment, fidelity should be evaluated against the **actual deployed wrapper or predictor being audited**, not merely against the benign model used for selected probe queries. An explanation that omits a feature is not, on its own, proof that the model is fair.

## 9. Proposed Project Structure

```text
recruitment-explanation-attack/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── src/
│   ├── data_generator.py
│   ├── models.py
│   ├── scaffold.py
│   ├── explainer.py
│   ├── defence.py
│   └── evaluation.py
├── experiments/
│   └── run_experiment.py
├── results/
│   ├── metrics.json
│   ├── explanation_examples.json
│   └── figures/
└── tests/
    └── test_pipeline.py
```

The defence and test modules may initially contain only minimal implementations or be deferred until the core experiment works.

## 12. Getting Started

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the experiment:

```bash
python experiments/run_experiment.py
```

The script should generate the synthetic data, initialise the models, run the selected experiment configurations, and save evaluation results to the `results/` directory.

Use fixed random seeds where appropriate so that the experiment can be reproduced and its results compared across configurations.


## Extending the idea beyond recruitment

To adapt this research approach to another black-box AI domain:

1. **Define the audit question.** Identify the outcome or decision pattern to investigate.
2. **Map the input schema.** Specify features, valid ranges, missing-value behaviour, and which features may be varied.
3. **Choose meaningful comparisons.** Define groups or matched inputs in a way that fits the domain and avoids misleading conclusions.
4. **Connect the predictor.** Use a consistent interface to query the actual model being audited.
5. **Select explanations and metrics.** Choose local explanations and evaluation methods suitable for the model and data.
6. **Validate the pipeline.** Test fidelity, stability, detector false positives/negatives, and reproducibility before interpreting results.
7. **Document constraints.** Record assumptions, limitations, privacy concerns, and any domain-specific requirements.

Possible application areas include lending, education, healthcare, insurance, content moderation, and automated resource allocation. These are **potential future adaptations**, not domains for which the current repository has already been validated.

## Limitations and responsible use

- The recruitment example uses synthetic data and deliberately controlled labels.
- The local surrogate is LORE-inspired, not the complete LORE implementation.
- The probe detector uses heuristics and may fail on realistic, adaptive, or distribution-matched probes.
- The recruitment experiment and scheduling-oriented Flask pipeline are not yet unified behind a common model/data interface.
- A high probe score does not prove malicious intent; legitimate auditors may make systematic queries.
- Local explanations approximate behaviour around an input and may not represent the model globally.
- Differences in model outputs do not alone establish unlawful discrimination or explain why a difference exists.
- Results from synthetic examples should not be generalised to real systems without further testing.

Do not use this prototype to make real employment, lending, healthcare, education, or other consequential decisions. For real-world audits, use appropriate authorisation, protect sensitive data, involve domain and fairness experts, and follow relevant legal and ethical requirements.

## Future work

- Unify the hiring and scheduling examples behind a documented, domain-independent predictor interface.
- Connect data generation, model querying, explanation, monitoring, and evaluation in a reproducible end-to-end experiment.
- Add tests for feature sensitivity, local fidelity, explanation stability, and detector performance.
- Compare simple neighbourhood generation with more realistic, domain-valid neighbourhoods.
- Evaluate adaptive probing and document threat-model assumptions.
- Add structured experiment outputs and visualisations, with metrics and criteria defined before running experiments.
- Investigate how an audit can remain useful without unfairly blocking legitimate explanation requests.

## Research contribution

The project explores a practical question in explainable AI and model auditing: **how can reviewers investigate a black-box system when its behaviour during explanation queries may not match its ordinary behaviour?** A synthetic recruitment example provides a controlled setting for studying age-sensitive decisions, while the broader approach motivates future adaptations to other domains.

The repository is intended as a reproducible research prototype. It does not claim that scaffolding attacks are novel, that the current detector is robust, or that the pipeline is already validated across multiple domains. Those claims would require systematic experiments and a review of related work on deceptive model behaviour, explanation attacks, and local explainability methods.

---

**Status:** Research prototype; synthetic recruitment and scheduling examples; integration and validation ongoing.  
**Primary goal:** Investigate hidden decision patterns and discrepancies between black-box behaviour and explanations.  
**Scope:** Broader than hiring, but new domains require adaptation and validation.
