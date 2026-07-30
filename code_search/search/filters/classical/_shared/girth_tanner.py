"""Tanner-graph girth of a binary parity-check matrix.

Group-agnostic. Used at the classical level on ``A_bin``, ``B_bin``, and
at the quantum level on ``Hx``, ``Hz`` (same function, different inputs).
"""

from collections import deque

import numpy as np


def girth_tanner(H: np.ndarray) -> int | None:
    """Girth of the Tanner graph of binary parity-check matrix ``H``.

    The Tanner graph is bipartite:
        - Check nodes ``C_0 … C_{r-1}`` (one per row of H).
        - Variable nodes ``V_0 … V_{c-1}`` (one per column of H).
        - Edge ``(C_i, V_j)`` exists iff ``H[i, j] = 1``.

    All cycles in a bipartite graph are even, so the minimum possible
    girth is 4. Girth-6 is the standard target for good LDPC codes.

    BFS from every node; ``O((r+c) × (r+c+edges))`` overall.

    Returns:
        int — the girth, OR ``None`` if the graph is a forest (no cycles).
    """
    r, c = H.shape
    N = r + c
    # Node indexing: 0..r-1 = check nodes, r..r+c-1 = variable nodes.
    adj: list[list[int]] = [[] for _ in range(N)]
    for i in range(r):
        for j in np.where(H[i])[0]:
            adj[i].append(r + int(j))
            adj[r + int(j)].append(i)

    min_girth = float("inf")
    for start in range(N):
        dist = [-1] * N
        par = [-1] * N
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] + 1 >= min_girth:
                break
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    par[v] = u
                    queue.append(v)
                elif v != par[u]:
                    min_girth = min(min_girth, dist[u] + dist[v] + 1)

    return int(min_girth) if min_girth < float("inf") else None
