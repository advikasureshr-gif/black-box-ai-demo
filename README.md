# Unmasking Deceptive Explanations in AI-Assisted Recruitment
## A Study of Scaffolding Attacks on Local Rule-Based Explainers

## 1. Overview

AI-assisted recruitment systems can help evaluate candidates, but their decisions may reflect unintended or deliberately introduced biases. Explainability methods are often used to make these decisions more transparent. However, an explanation may not faithfully represent the behavior of the model being explained if the model can recognize when it is being examined.

This project investigates whether a **scaffolding attack** can conceal an age-based preference in a simulated recruitment model from a local rule-based explainer.

We will create synthetic candidate profiles with two primary features:

- **Age:** The candidate's age.
- **Technical proficiency:** A numerical score representing technical ability.

The experiment compares a fair baseline model, an age-biased model, and a scaffolded version of the biased model. A simplified local rule-based explainer will approximate the decision process around individual candidates and generate human-readable rules.

The implementation will be a lightweight, reproducible proof of concept using Python and scikit-learn. The explainer will be **LORE-inspired**, rather than a complete implementation of the original LORE algorithm.

## 2. Problem Statement

A recruitment model may favour younger candidates even when candidates have equivalent technical proficiency. A local explainer should ideally reveal the model's reliance on age.

However, a model designed to behave differently when it detects synthetic explanation queries could return apparently fair predictions to the explainer while continuing to make age-biased decisions on ordinary candidate profiles.

This creates a gap between the model's actual behaviour and the behaviour suggested by its explanation.

The project studies whether this discrepancy can be demonstrated in a controlled synthetic setting and whether more realistic neighbourhood generation can help expose it.

## 3. Research Question

**Can a scaffolding attack conceal an age-based preference in an AI recruitment model from a local rule-based explainer when candidate technical proficiency is held constant, and can more realistic neighbourhood generation improve detection?**

### Objectives

1. Generate synthetic candidate profiles with independently controlled age and technical proficiency.
2. Build a fair baseline model and a deliberately age-biased model.
3. Implement a scaffolding mechanism that attempts to recognise explanation queries and return benign predictions for them.
4. Build a simplified local rule-based explainer that approximates model decisions near a candidate profile.
5. Compare explanations and model decisions with and without scaffolding.
6. Evaluate whether realistic neighbourhood generation reduces the discrepancy between explanations and actual model behaviour.

## 4. Methodology

### 4.1 Synthetic Dataset

Generate fictional candidate profiles containing:

| Feature | Description |
|---|---|
| Age | Candidate age, sampled from a defined range |
| Technical proficiency | Score on a scale of 0–100 |
| Hiring label | Simulated hiring decision |

Age and technical proficiency will be generated independently to avoid accidentally associating older or younger candidates with higher proficiency.

No real candidate data or personally identifiable information will be used.

Experience is intentionally excluded from the initial experiment. This isolates age preference from the legitimate possibility that recruiters may value prior experience.

### 4.2 Fair Baseline Model

Build a baseline recruitment model whose decisions depend on technical proficiency and do not depend on age.

This model provides a reference for assessing whether the biased model treats equally proficient candidates differently based on age.

### 4.3 Age-Biased Model

Create a second model that deliberately favours younger candidates, even when technical proficiency is held constant.

The bias will be introduced through a controlled decision rule or synthetic training labels. Its effect will be tested using candidate pairs with equal proficiency and different ages.

The experiment will verify that the intended bias is actually present before evaluating the attack.

### 4.4 Scaffolding Attack

Wrap the biased model with a mechanism that attempts to identify inputs generated during explanation queries.

When a query is classified as an explanation probe, the wrapper will attempt to return a benign prediction. For ordinary candidate profiles, it will use the age-biased model.

This creates two behaviours to compare:

- **Ordinary inference:** The model uses the biased decision process.
- **Explanation probing:** The wrapper attempts to conceal that behaviour.

The detection mechanism and its limitations will be documented. In particular, a detector that exploits obvious differences between synthetic probes and normal profiles may demonstrate a limited proof of concept without establishing that the attack works against realistic or adaptive explainers.

### 4.5 LORE-Inspired Local Rule-Based Explainer

Implement a simplified local explainer that:

1. Takes a candidate profile as the instance to explain.
2. Generates nearby candidate profiles.
3. Queries the model for their hiring predictions.
4. Fits a shallow decision tree to approximate those local predictions.
5. Extracts readable decision rules from the tree.

The explainer will be used on the fair baseline, the biased model, and the scaffolded model.

This implementation captures the basic idea of approximating local decision behaviour with rules. It does **not** reproduce every component of LORE, such as its full genetic-algorithm-based counterfactual generation process.

### 4.6 Neighbourhood-Based Defence (Stretch Goal)

Investigate whether making explanation neighbourhoods more realistic helps reveal the attack.

Possible approaches include using training examples as a source of plausible nearby candidates, constraining generated profiles to valid ranges, and comparing predictions across different neighbourhood-generation strategies.

The defence will be evaluated experimentally rather than assumed to be effective.

## 5. Experimental Configurations

| Configuration | Model | Explanation procedure | Purpose |
|---|---|---|---|
| A | Fair baseline | Local rule-based explainer | Establish reference behaviour |
| B | Age-biased model | Local rule-based explainer | Determine whether age preference appears in the explanation |
| C | Scaffolded biased model | Same explainer | Test whether scaffolding conceals the bias |
| D *(stretch)* | Scaffolded biased model | Alternative, more realistic neighbourhood generation | Test a potential defence |

Where possible, the same candidate profiles, random seeds, and evaluation settings will be used across configurations to make comparisons reproducible.

## 6. Evaluation Metrics

### 6.1 Selection-Rate Difference

Compare hiring rates across age groups while holding technical proficiency constant.

This checks whether the model's decisions change with age when the job-relevant feature is equivalent.

### 6.2 Age Dependence in Explanations

Record whether extracted rules use age and how strongly age appears to influence local decisions.

The presence or absence of an age rule alone will not be treated as proof of fairness.

### 6.3 Local Fidelity

Measure how accurately the extracted local rule model predicts the decisions of the model being explained on held-out neighbourhood samples.

Fidelity should be evaluated against the **actual deployed model or wrapper**, not just the benign model used during explanation probes.

### 6.4 Attack Success Rate

Define attack success in advance as the proportion of tested cases where the explanation conceals the known age preference while the actual model continues to exhibit that preference.

Report the denominator and the exact success criteria alongside the result.

### 6.5 Probe-Detection Performance

Evaluate the scaffolding detector using precision, recall, F1-score, and a confusion matrix where applicable.

This helps establish whether the detector distinguishes explanation probes from ordinary candidate inputs and whether it simply exploits obvious differences in their distributions.

### 6.6 Runtime

Record the approximate execution time for data generation, explanation generation, and evaluation.

## 7. Expected Outputs

The project will aim to produce:

- A reproducible synthetic recruitment dataset generator.
- A fair baseline and an age-biased recruitment model.
- A scaffolding wrapper for controlled experimentation.
- A simplified local rule-based explainer.
- Side-by-side explanations for the experimental configurations.
- Metrics describing selection disparities, local fidelity, and attack performance.
- Plots comparing model decisions and explanation behaviour.
- A discussion of limitations and possible defences.

These are planned outputs, not predetermined results. The experiment may show that the attack fails, that its success depends on the neighbourhood distribution, or that the explainer exposes the bias despite scaffolding.

## 8. Technology Stack

- **Python** — implementation and experimentation
- **NumPy** — numerical operations and synthetic data generation
- **pandas** — dataset manipulation and result tables
- **scikit-learn** — classification models, decision trees, and evaluation metrics
- **Matplotlib** — visualisations
- **pytest** *(optional)* — basic pipeline tests

The initial implementation will avoid large pretrained models and unnecessary dependencies.

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

## 10. Scope and Limitations

This is an initial, controlled proof of concept rather than a comprehensive evaluation of recruitment systems or explainability methods.

Important limitations include:

- The data and hiring decisions are synthetic.
- The local explainer is LORE-inspired, not the complete LORE algorithm.
- A simple probe detector may exploit artefacts that a more realistic or adaptive explainer would avoid.
- Results from one synthetic data distribution may not generalise to real recruitment systems.
- Fairness metrics and local explanations capture different aspects of model behaviour; neither alone establishes that a recruitment system is fair.

A future extension could compare the implementation with the original LORE method, investigate adaptive probing, and evaluate stronger defences.

## 11. Research Contribution

The intended contribution is a small, reproducible experiment investigating the relationship between scaffolding attacks, local rule-based explanations, and age-based decision disparities in AI-assisted recruitment.

The study will focus on whether explanations faithfully reflect the deployed model's behaviour and under what conditions the discrepancy can be detected.

The project will not claim that scaffolding attacks against local explainers are unprecedented. Any claim of novelty will require a separate review of existing work on deceptive model behaviour, LIME/SHAP scaffolding attacks, and LORE-style explainers.

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
