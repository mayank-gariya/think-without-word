from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConceptMapping:
    concept: str
    our_toy_model: str
    bdh_cq: str
    important_distinction: str


RECURRENT_LATENT_REASONING = ConceptMapping(
    concept="Recurrent latent reasoning",
    our_toy_model=(
        "A fixed-size state is repeatedly updated to "
        "carry out a multi-step transformation."
    ),
    bdh_cq=(
        "BDH-CQ combines recurrent memory updates from "
        "inference-time demonstrations with iterative "
        "latent computation for solving a query."
    ),
    important_distinction=(
        "Our implementation is a controlled educational "
        "toy and is not an implementation or reproduction "
        "of BDH-CQ."
    ),
)


BDH_MEMORY = ConceptMapping(
    concept="Recurrent memory",
    our_toy_model=(
        "The transition structure remains fixed while "
        "the current state evolves."
    ),
    bdh_cq=(
        "Inference-time inputs update recurrent memory, "
        "which is then used during latent reasoning."
    ),
    important_distinction=(
        "The memory mechanism in our toy is symbolic and "
        "explicit; BDH-CQ uses a learned recurrent "
        "memory mechanism."
    ),
)