from __future__ import annotations

import json
import os

import streamlit as st
import torch
import plotly.graph_objects as go

from data.generator import (
    generate_dataset,
    example_to_text,
    answer_to_label,
)

from models.latent_reasoner import (
    LatentReasoner,
)

from visualization.latent_plot import (
    latent_state_bar,
    latent_trajectory_figure,
)


# ==================================================
# Page configuration
# ==================================================

st.set_page_config(
    page_title="Think Without Words",
    page_icon="🧠",
    layout="wide",
)


# ==================================================
# Constants
# ==================================================

NUM_NODES = 12

RESULTS_PATH = (
    "experiments/results.json"
)


# ==================================================
# Custom styling
# ==================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.25rem;
        opacity: 0.8;
        margin-bottom: 2rem;
    }

    .concept-box {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# Model
# ==================================================

@st.cache_resource
def load_model():

    model = LatentReasoner(
        num_nodes=NUM_NODES
    )

    model.eval()

    return model


model = load_model()


# ==================================================
# Results
# ==================================================

@st.cache_data
def load_results():

    if not os.path.exists(
        RESULTS_PATH
    ):
        return {}

    with open(
        RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


results = load_results()


# ==================================================
# Generate a deterministic demo task
# ==================================================

@st.cache_data
def get_demo_example(
    hops: int,
):

    examples = generate_dataset(
        num_samples=1,
        num_nodes=NUM_NODES,
        min_hops=hops,
        max_hops=hops,
        seed=2026 + hops,
    )

    return examples[0]


# ==================================================
# Header
# ==================================================

st.markdown(
    '<div class="main-title">'
    '🧠 Think Without Words'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Can a system spend more computation updating a latent '
    'state instead of producing intermediate reasoning tokens?'
    '</div>',
    unsafe_allow_html=True,
)


# ==================================================
# Concept introduction
# ==================================================

with st.container():

    st.markdown(
        """
        <div class="concept-box">

        <strong>The idea</strong>

        In this experiment, reasoning is represented as
        repeated updates to a fixed-size internal state.

        Instead of producing a visible word for every
        intermediate step, the system repeatedly transforms
        its current state before producing an answer.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# Sidebar
# ==================================================

st.sidebar.title(
    "Experiment"
)

st.sidebar.markdown(
    "Control the amount of recurrent computation."
)

reasoning_budget = st.sidebar.slider(
    "Reasoning budget",
    min_value=0,
    max_value=32,
    value=6,
    step=1,
)

st.sidebar.caption(
    "Each step performs one recurrent state update."
)


task_hops = st.sidebar.slider(
    "Task depth",
    min_value=2,
    max_value=8,
    value=6,
    step=1,
)

st.sidebar.caption(
    "How many transitions the current problem requires."
)


# ==================================================
# Generate task
# ==================================================

example = get_demo_example(
    task_hops
)


# ==================================================
# Run model
# ==================================================

transition_matrix = torch.tensor(
    example.transitions,
    dtype=torch.float32,
).unsqueeze(0)

start_node = torch.tensor(
    [example.start],
    dtype=torch.long,
)


with torch.no_grad():

    logits, states = model(
        transition_matrix,
        start_node,
        reasoning_steps=reasoning_budget,
    )

    prediction = torch.argmax(
        logits,
        dim=1,
    ).item()


ground_truth = example.answer

correct = (
    prediction == ground_truth
)


# ==================================================
# Section 1 — Problem
# ==================================================

st.header(
    "1. Try the reasoning task"
)

left, right = st.columns(
    [1.1, 0.9]
)

with left:

    st.markdown(
        "### Problem"
    )

    st.code(
        example_to_text(example),
        language="text",
    )


with right:

    st.markdown(
        "### What the system is doing"
    )

    st.markdown(
        f"""
        The recurrent simulator starts at **{answer_to_label(example.start)}**
        and receives a recurrent computation budget of
        **{reasoning_budget} steps**.

        The task requires **{example.hops} transitions**.

        """

        + (
            "The budget is sufficient to reach the target."
            if reasoning_budget >= example.hops
            else
            "The budget is currently smaller than the "
            "required depth."
        )
    )


# ==================================================
# Section 2 — Truth vs prediction
# ==================================================

st.header(
    "2. Ground truth vs simulated prediction"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Ground truth",
        answer_to_label(
            ground_truth
        ),
    )


with col2:

    st.metric(
        "Simulated prediction",
        answer_to_label(
            prediction
        ),
    )


with col3:

    st.metric(
        "Reasoning steps",
        reasoning_budget,
    )


with col4:

    st.metric(
        "Task depth",
        example.hops,
    )


if correct:

    st.success(
        "✓ Correct — the available recurrent "
        "computation reached the target state."
    )

else:

    st.error(
        "✗ Incorrect — the available recurrent "
        "computation did not reach the target."
    )


st.caption(
    "The simulator produces a discrete current state; "
    "it does not estimate probabilistic confidence."
)


# ==================================================
# Section 3 — Latent state
# ==================================================

st.header(
    "3. Watch the latent state evolve"
)

st.markdown(
    """
    Every recurrent update changes the internal state.
    The labels below are interpretations of the current
    state; the underlying representation is a vector.
    """
)


labels = [
    answer_to_label(i)
    for i in range(NUM_NODES)
]


current_state = states[-1]

state_figure = latent_state_bar(
    current_state,
    labels,
)

st.plotly_chart(
    state_figure,
    use_container_width=True,
)


# ==================================================
# Section 4 — State trajectory
# ==================================================

st.subheader(
    "Latent-state trajectory"
)

trajectory = latent_trajectory_figure(
    states
)

st.plotly_chart(
    trajectory,
    use_container_width=True,
)

st.caption(
    "This is a 2D PCA projection of the latent states. "
    "It is a visualization aid, not the literal reasoning space."
)


# ==================================================
# Section 5 — Evidence curve
# ==================================================

st.header(
    "4. What happens when we give it more compute?"
)

if results:

    steps = [
        int(key)
        for key in results.keys()
    ]

    accuracies = [
        float(value) * 100
        for value in results.values()
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=steps,
            y=accuracies,
            mode="lines+markers",
            name="Test accuracy",
        )
    )

    # Current budget marker.
    figure.add_vline(
        x=reasoning_budget,
        line_dash="dash",
        annotation_text=(
            f"Current budget: "
            f"{reasoning_budget}"
        ),
    )

    figure.update_layout(
        xaxis_title="Recurrent computation budget",
        yaxis_title="Accuracy (%)",
        yaxis_range=[
            0,
            105,
        ],
        height=450,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


else:

    st.warning(
        "Run `python -m experiments.evaluate` "
        "to generate the population-level results."
    )


# ==================================================
# Section 6 — Explain the result
# ==================================================

st.header(
    "5. What did you just observe?"
)

st.markdown(
    f"""
    You gave the system **{reasoning_budget} recurrent
    computation step(s)**.

    This particular task requires **{example.hops} step(s)**
    to reach its target.

    """

    + (
        """
        Because the computation budget reached the required
        depth, the latent state was able to arrive at the
        correct answer.
        """
        if reasoning_budget >= example.hops
        else
        """
        Because the computation budget was smaller than the
        required depth, the state could not complete the
        full transformation.
        """
    )
)


# ==================================================
# Section 7 — Scientific honesty
# ==================================================

st.divider()

st.subheader(
    "What this experiment does — and does not — claim"
)

st.markdown(
    """
    **This demonstration:** a controlled toy simulation
    of recurrent state updates.

    **It does not claim:** that this toy system reproduces
    the behavior of a large language model, BDH, or BDH-CQ.

    **Why it matters:** recent research investigates
    related approaches in which additional inference
    computation can be spent through recurrent latent
    updates rather than producing additional verbal
    reasoning tokens.
    """
)

st.caption(
    "The toy substrate is an educational implementation, "
    "not an official BDH/BDH-CQ implementation."
)


# ==================================================
# Section 8 — BDH / BDH-CQ Connection
# ==================================================

st.divider()

st.header(
    "6. From recurrent computation to BDH-CQ"
)

st.markdown(
    """
    Our toy experiment demonstrates one ingredient:

    **a state can be repeatedly updated in latent space.**

    BDH-CQ studies a broader system in which
    inference-time demonstrations can update recurrent
    memory, followed by iterative latent reasoning.
    """
)

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        "Our toy experiment"
    )

    st.code(
        """
Input
  ↓
h₀
  ↓
h₁
  ↓
h₂
  ↓
h₃
  ↓
Answer
        """,
        language="text",
    )

    st.caption(
        "Fixed-size state repeatedly refined."
    )


with col2:

    st.subheader(
        "BDH-CQ at a high level"
    )

    st.code(
        """
Demonstrations
      ↓
Recurrent memory update
      ↓
Query
      ↓
Latent reasoning
      ↓
Answer
        """,
        language="text",
    )

    st.caption(
        "Inference-time inputs can update recurrent memory "
        "before latent reasoning."
    )


st.subheader(
    "The shared idea: recurrent state updates"
)

st.latex(
    r"""
    h_{t+1}=F(h_t,\,x_t)
    """
)

st.markdown(
    """
    In our toy model, the state is updated repeatedly
    according to the task's transition dynamics.

    In a learned recurrent reasoning system, the update
    function is learned and operates on a high-dimensional
    latent representation.

    BDH-CQ extends this idea by incorporating recurrent
    memory updates from inference-time demonstrations.
    """
)


# ==================================================
# Section 9 — Memory experiment
# ==================================================

st.subheader(
    "Memory experiment"
)

st.markdown(
    """
    Imagine that demonstrations arrive before the query.
    Each demonstration updates an internal memory.

    This is a simplified educational model of the idea,
    not the actual BDH-CQ memory mechanism.
    """
)


demo1_key = st.selectbox(
    "Demonstration 1 — input",
    ["A", "B", "C", "D"],
    index=0,
    key="demo1_key",
)

demo1_value = st.selectbox(
    "Demonstration 1 — output",
    ["A", "B", "C", "D"],
    index=1,
    key="demo1_value",
)


demo2_key = st.selectbox(
    "Demonstration 2 — input",
    ["A", "B", "C", "D"],
    index=1,
    key="demo2_key",
)

demo2_value = st.selectbox(
    "Demonstration 2 — output",
    ["A", "B", "C", "D"],
    index=2,
    key="demo2_value",
)


query = st.selectbox(
    "Query",
    ["A", "B", "C", "D"],
    index=0,
    key="query_key",
)

# Simple in-memory demonstration
memory = {}

if demo1_key and demo1_value:
    memory[demo1_key] = demo1_value

if demo2_key and demo2_value:
    memory[demo2_key] = demo2_value

st.markdown(
    "### Memory after demonstrations"
)

st.json(
    memory
)

answer = memory.get(query)

if answer is None:

    st.warning(
        f"No stored association for {query}."
    )

else:

    st.success(
        f"Memory retrieves: "
        f"{query} → {answer}"
    )


# ==================================================
# Section 10 — Evidence boundary
# ==================================================

st.subheader(
    "Evidence boundary"
)

st.markdown(
    """
    **Research-backed:** BDH-CQ uses inference-time inputs
    to update recurrent memory and performs iterative
    latent reasoning without verbalizing intermediate steps.

    **Primary source:** BDH-CQ technical report, 2026.

    The published system combines inference-time memory
    updates with recurrent latent reasoning. Our toy
    experiment is only an educational abstraction of this
    broader idea.

    **Our demonstration:** the small memory above is a
    deliberately simplified educational abstraction.
    It is not the learned memory implementation used by
    BDH-CQ.

    **Our toy recurrent experiment:** demonstrates repeated
    state updates on a controlled task. It is not a
    reproduction of BDH or BDH-CQ.
    """
)

st.divider()

st.header(
    "7. Why is this related?"
)

st.markdown(
    """
    Your experiment showed recurrent computation. 

    BDH-CQ adds recurrent memory to the same broad computational 
    picture: **information encountered during inference can alter 
    an internal state, which is then used for further latent 
    computation.**
    """
)

st.caption(
    "For more information, see the BDH-CQ technical report "
    "and the official BDH implementation on GitHub."
)