"""Canonical brute-force LP code search.

A dedicated, brute-force search for the *smallest* shape-(1,2) LP code with
a clean **canonical logical basis** at a target distance, over non-abelian
groups. Distinct from ``search.phases`` (random-sample + BP+OSD): this
package enumerates exhaustively, screens distance with **sqetch** (classical
*and* quantum), and is built to fan out across many GPU workers.

Design:

- shape (1,2) → ``n = 5|G|``, ``k = |G|`` (single orbit) → canonical basis.
- **Full-rank pool**, brute-forced on local CPU: all weight-w ring elements
  with the **identity forced into the support** whose binary lift is a unit.
- A = [a0, a1] (B = [b0, b1]); the unit anchor sits at the LAST block-col
  (canonical position), so codes are **canonical by construction**. ``a0``/
  ``b0`` range over *all* weight-w elements (brute free columns).
- ``d(A)``, ``d(Bᵀ)`` upper-bound the quantum distance, so we sqetch-screen
  the classical sides first (1M trials, ``k_sub`` = ker dim, aggressive
  ``d_target`` early-stop), keep the top-N by distance, then pair.
- Pairing: sqetch FIRST, BP+OSD confirms only the passers; cap the number of
  passed quantum codes.
- Per passer: build the G-orbit canonical logical basis, record its row
  weights, and save a rich self-describing JSON.
- Traversal: ascending group order, stop at the smallest order with a
  verified code; giant orders (96 / 144 / 160) deferred to a second pass.
"""
