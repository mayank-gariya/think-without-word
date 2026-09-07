from __future__ import annotations

import json

import torch

from data.generator import (
    generate_dataset,
)

from models.latent_reasoner import (
    LatentReasoner,
)


NUM_NODES = 12

TEST_SAMPLES = 5000

REASONING_BUDGETS = [
    0,
    1,
    2,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
]


def evaluate():

    model = LatentReasoner(
        num_nodes=NUM_NODES
    )

    model.eval()

    # IMPORTANT:
    # Tasks have different reasoning depths.
    examples = generate_dataset(
        num_samples=TEST_SAMPLES,
        num_nodes=NUM_NODES,
        min_hops=1,
        max_hops=8,
        seed=12345,
    )

    results = {}

    print(
        "\nRecurrent latent computation experiment\n"
    )

    with torch.no_grad():

        for budget in REASONING_BUDGETS:

            correct = 0

            for example in examples:

                transition_matrix = (
                    torch.tensor(
                        example.transitions,
                        dtype=torch.float32,
                    )
                    .unsqueeze(0)
                )

                start_node = torch.tensor(
                    [example.start],
                    dtype=torch.long,
                )

                logits, states = model(
                    transition_matrix,
                    start_node,
                    reasoning_steps=budget,
                )

                prediction = (
                    torch.argmax(
                        logits,
                        dim=1,
                    ).item()
                )

                if prediction == example.answer:

                    correct += 1

            accuracy = (
                correct /
                len(examples)
            )

            results[str(budget)] = accuracy

            print(
                f"Budget: {budget:2d} | "
                f"Accuracy: {accuracy:.4f}"
            )

    with open(
        "experiments/results.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        "\nSaved to experiments/results.json"
    )


if __name__ == "__main__":
    evaluate()