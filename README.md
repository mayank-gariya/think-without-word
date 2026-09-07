# Think Without Words

An interactive educational explainer for **Recurrent Latent-Space Reasoning**.

> **Central claim:** Recurrent computation can be allocated through repeated updates to a fixed-size state, allowing a system to carry out additional latent computation without generating an intermediate reasoning token at every step.

## What this project teaches

`Think Without Words` is designed for an ML/data-science learner who understands basic neural networks and wants an intuitive introduction to recurrent latent computation.

The learner can:

1. Choose the depth of a synthetic multi-step reasoning task.
2. Change the recurrent computation budget.
3. Observe the resulting state.
4. Compare the current prediction with the known ground truth.
5. Inspect a visualization of the state trajectory.
6. Compare live behavior with a precomputed accuracy-vs-computation experiment.
7. Explore a simplified demonstration-memory abstraction and then connect it to BDH-CQ.

## Important scientific boundary

The main interactive simulator is an **educational toy substrate**. It is not an implementation, reproduction, or benchmark of BDH or BDH-CQ.

The recurrent-state experiment is deterministic and controlled so that learners can directly observe the relationship between computation steps and completion of synthetic multi-step transformations.

The project uses published research to explain why related mechanisms matter for frontier models. Research claims are not inferred from the toy experiment.

## Interactive experiment

The simulator represents a state as a vector over possible nodes.

For a transition matrix `M` and current state `p_t`, one recurrent update is:

    p_(t+1) = p_t M

A task consists of a synthetic chain such as:

    A → D
    D → B
    B → F
    F → C
    C → H

Starting from `A`, the correct state after four transitions is `C`.

The learner changes the **recurrent computation budget** and observes how far the state progresses.

### Recorded experiment

The current population-level experiment produced:

| Recurrent computation budget | Accuracy |
|---:|---:|
| 0 | 0.00% |
| 1 | 11.98% |
| 2 | 23.96% |
| 4 | 49.28% |
| 6 | 74.26% |
| 8 | 100.00% |
| 12 | 100.00% |
| 16 | 100.00% |
| 24 | 100.00% |
| 32 | 100.00% |

These values are retained as recorded experiment outputs, not hand-selected demonstration numbers.

## Research connection

The project introduces the learner to a broader research progression:

    verbal reasoning
          ↓
    latent reasoning
          ↓
    recurrent latent computation
          ↓
    recurrent memory + latent reasoning
          ↓
          BDH-CQ

### Primary research references

- Shibo Hao et al. (2024). **Training Large Language Models to Reason in a Continuous Latent Space (Coconut).**
  https://arxiv.org/abs/2412.06769

- Jonas Geiping et al. (2025). **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.** NeurIPS 2025.
  https://proceedings.neurips.cc/paper_files/paper/2025/hash/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Abstract-Conference.html

- Björn Engdahl et al. (2026). **BDH-CQ: In-Context Learning with Recurrent Latent Reasoning.**
  https://arxiv.org/abs/2608.09888

- Adrian Kosowski et al. (2025). **The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.**
  https://arxiv.org/abs/2509.26507

## BDH-CQ connection

BDH-CQ is presented as a related frontier system in which inference-time inputs update recurrent memory, followed by iterative latent computation for a query.

The project deliberately separates:

**Published evidence**
- BDH/BDH-CQ architecture and reported evaluations from their primary sources.

**Educational abstraction**
- The small memory demonstration in this app.

**Our experiment**
- The deterministic recurrent-state simulator.

The memory widget is therefore a teaching abstraction and must not be described as the implementation of BDH-CQ's learned memory mechanism.

## Repository structure

```text
think-without-words/
│
├── app.py
├── requirements.txt
│
├── data/
│   ├── __init__.py
│   └── generator.py
│
├── models/
│   ├── __init__.py
│   └── latent_reasoner.py
│
├── experiments/
│   ├── __init__.py
│   ├── evaluate.py
│   └── results.json
│
├── visualization/
│   ├── __init__.py
│   └── latent_plot.py
│
├── bdh/
│   └── __init__.py
|   └── concepts.py
|   └── demo.py
│
├── docs/
│   └── one_page_technical_summary.pdf
│
├── AI_DISCLOSURE.md
├── LICENSES.md
└── README.md
```

Remove obsolete files from earlier prototypes before submission, especially old trained-model artifacts and unused training scripts if they are no longer part of the final experiment.

## Running locally

### 1. Clone

```bash
git clone <YOUR-REPOSITORY-URL>
cd think-without-words
```

### 2. Create an environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the recorded experiment

```bash
python -m experiments.evaluate
```

This writes:

```text
experiments/results.json
```

### 5. Run the application

```bash
streamlit run app.py
```

## What is live, precomputed, and synthetic?

| Component | Status |
|---|---|
| Current interactive task | Live computation |
| State-transition updates | Live computation |
| Ground-truth comparison | Live deterministic computation |
| Latent-state trajectory | Live visualization |
| Accuracy-vs-budget curve | Precomputed from `experiments/results.json` |
| Reasoning tasks | Synthetic |
| BDH-CQ memory widget | Simplified educational abstraction |
| BDH / BDH-CQ research results | Published external evidence |

## Limitations

This project should not be interpreted as evidence that more computation always improves frontier-model reasoning.

The synthetic environment is deliberately controlled. The recurrent state is explicit and deterministic, unlike the learned, high-dimensional representations used in frontier language-model research.

The experiment demonstrates a mechanism and a learning principle; it does not reproduce the empirical behavior of a large language model.

## Reproducibility

The synthetic generator uses fixed seeds for the recorded evaluation. The repository should retain the generated `results.json` so the deployed educational experience can load the same population-level curve.

## AI assistance

AI tools were used during development. See `AI_DISCLOSURE.md` for the assistance categories and human responsibility.

## Licensing and provenance

See `LICENSES.md` for project code, research links, third-party packages, fonts, graphics, and reused assets.

## Competition alignment

This project is designed around the DataForge Pathway requirements:

- one focused technical claim;
- an interactive learning substrate;
- a meaningful learner-controlled variable;
- visible state;
- ground truth beside prediction;
- a substantive BDH-CQ connection;
- explicit evidence boundaries;
- primary research references;
- reproducibility and provenance documentation.

The project does not present the toy simulator as an official BDH/BDH-CQ implementation.
