from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class ReasoningExample:
    transitions: np.ndarray
    start: int
    hops: int
    answer: int


def generate_example(
    num_nodes: int = 12,
    min_hops: int = 1,
    max_hops: int = 8,
    seed: int | None = None,
) -> ReasoningExample:

    rng = random.Random(seed)

    nodes = list(range(num_nodes))
    rng.shuffle(nodes)

    # Select a chain of states.
    max_possible_hops = num_nodes - 1

    max_hops = min(
        max_hops,
        max_possible_hops,
    )

    hops = rng.randint(
        min_hops,
        max_hops,
    )

    chain = nodes[: hops + 1]

    transitions = np.zeros(
        (num_nodes, num_nodes),
        dtype=np.float32,
    )

    # Build the actual reasoning chain.
    #
    # Example:
    # A -> D -> B -> F -> C
    #
    for i in range(hops):
        source = chain[i]
        target = chain[i + 1]

        transitions[source, target] = 1.0

    # Terminal state loops to itself.
    terminal = chain[-1]

    transitions[
        terminal,
        terminal,
    ] = 1.0

    # Every unused node also gets a self-loop.
    for node in nodes[hops + 1 :]:
        transitions[node, node] = 1.0

    start = chain[0]
    answer = chain[-1]

    return ReasoningExample(
        transitions=transitions,
        start=start,
        hops=hops,
        answer=answer,
    )


def generate_dataset(
    num_samples: int,
    num_nodes: int = 12,
    min_hops: int = 1,
    max_hops: int = 8,
    seed: int = 42,
):

    rng = random.Random(seed)

    dataset = []

    for _ in range(num_samples):

        sample_seed = rng.randint(
            0,
            1_000_000_000,
        )

        dataset.append(
            generate_example(
                num_nodes=num_nodes,
                min_hops=min_hops,
                max_hops=max_hops,
                seed=sample_seed,
            )
        )

    return dataset


def node_to_label(node: int) -> str:
    return chr(
        ord("A") + node
    )


def answer_to_label(answer: int) -> str:
    return node_to_label(answer)


def example_to_text(
    example: ReasoningExample,
) -> str:

    lines = []

    visited = set()

    current = example.start

    while current not in visited:

        visited.add(current)

        target = int(
            np.argmax(
                example.transitions[current]
            )
        )

        lines.append(
            f"{node_to_label(current)} → "
            f"{node_to_label(target)}"
        )

        if target == current:
            break

        current = target

    lines.append("")

    lines.append(
        f"Start: "
        f"{node_to_label(example.start)}"
    )

    lines.append(
        f"Question: Where are you after "
        f"{example.hops} transitions?"
    )

    return "\n".join(lines)