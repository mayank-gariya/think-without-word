from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def latent_state_bar(
    state,
    labels: list[str],
    title: str = "Current Latent State",
) -> go.Figure:
    """
    Display the current latent state as a probability bar chart.

    The state is the actual computational state used by
    our educational recurrent simulator.
    """

    values = (
        state.squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=labels,
            y=values,
        )
    )

    figure.update_layout(
        title=title,
        xaxis_title="Possible state",
        yaxis_title="State activation",
        yaxis_range=[0, 1],
        height=380,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return figure


def latent_trajectory_figure(
    states,
) -> go.Figure:
    """
    Project the latent states into 2D using PCA.

    Important:
    This is only a 2D visualization of the state.
    It is not the actual latent space itself.
    """

    vectors = []

    for state in states:

        vector = (
            state.squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        vectors.append(vector)

    vectors = np.asarray(vectors)

    # Handle the case where we only have one state.
    if len(vectors) == 1:

        vectors = np.vstack(
            [
                vectors,
                vectors + 1e-6,
            ]
        )

    if vectors.shape[1] == 1:

        vectors = np.hstack(
            [
                vectors,
                np.zeros(
                    (len(vectors), 1)
                ),
            ]
        )

    pca = PCA(
        n_components=2
    )

    projected = pca.fit_transform(
        vectors
    )

    steps = np.arange(
        len(projected)
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=projected[:, 0],
            y=projected[:, 1],
            mode="lines+markers",
            marker=dict(
                size=10,
            ),
            text=[
                f"Step {int(step)}"
                for step in steps
            ],
            hovertemplate=(
                "%{text}<br>"
                "x=%{x:.3f}<br>"
                "y=%{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    # Mark starting state.
    figure.add_annotation(
        x=projected[0, 0],
        y=projected[0, 1],
        text="h₀",
        showarrow=True,
        arrowhead=2,
    )

    # Mark final state.
    figure.add_annotation(
        x=projected[-1, 0],
        y=projected[-1, 1],
        text=f"h₍{len(projected)-1}₎",
        showarrow=True,
        arrowhead=2,
    )

    figure.update_layout(
        title="Latent-State Trajectory",
        xaxis_title="Latent dimension 1",
        yaxis_title="Latent dimension 2",
        height=450,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return figure