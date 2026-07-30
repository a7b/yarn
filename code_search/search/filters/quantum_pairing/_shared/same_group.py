"""Pair filter: both classical codes must come from the same group."""


def same_group(group_tag_A: str, group_tag_B: str) -> bool:
    """``True`` iff the two classical sides came from the same group.

    Pairing across distinct groups never makes sense for an LP code — the
    binary lifts use ``L[·]`` / ``R[·]`` for a single ``GroupData``. This
    predicate is the cheapest gate in the pairing dispatcher.
    """
    return group_tag_A == group_tag_B
