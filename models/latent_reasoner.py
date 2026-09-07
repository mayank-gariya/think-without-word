from __future__ import annotations

import torch
import torch.nn as nn


class LatentReasoner(nn.Module):
    """
    Educational recurrent-state simulator.

    This is NOT BDH or BDH-CQ.

    The state is represented as a fixed-size vector.
    Each recurrent step updates that state using the
    transition dynamics.

    p_(t+1) = p_t M
    """

    def __init__(
        self,
        num_nodes: int,
    ) -> None:

        super().__init__()

        self.num_nodes = num_nodes

    def forward(
        self,
        transition_matrix: torch.Tensor,
        start_node: torch.Tensor,
        reasoning_steps: int,
    ):

        # One-hot initial latent state.
        state = torch.nn.functional.one_hot(
            start_node,
            num_classes=self.num_nodes,
        ).float()

        states = [
            state.detach().clone()
        ]

        for _ in range(reasoning_steps):

            # One recurrent state update.
            state = torch.bmm(
                state.unsqueeze(1),
                transition_matrix,
            ).squeeze(1)

            states.append(
                state.detach().clone()
            )

        # State itself is the output distribution.
        logits = torch.log(
            state.clamp(
                min=1e-8
            )
        )

        return logits, states