from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryState:
    """
    Educational representation of a small associative memory.

    This is NOT the internal memory representation of BDH-CQ.
    """

    associations: dict[str, str]

    def update(
        self,
        key: str,
        value: str,
    ) -> None:

        self.associations[key] = value

    def query(
        self,
        key: str,
    ) -> str | None:

        return self.associations.get(key)


def build_demo_memory(
    demonstrations: list[tuple[str, str]],
) -> MemoryState:

    memory = MemoryState(
        associations={}
    )

    for key, value in demonstrations:

        memory.update(
            key,
            value,
        )

    return memory


def run_memory_demo(
    demonstrations: list[tuple[str, str]],
    query: str,
):

    memory = build_demo_memory(
        demonstrations
    )

    return {
        "memory": memory.associations,
        "query": query,
        "answer": memory.query(query),
    }