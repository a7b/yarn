"""Render mitten-code compact-frame layouts as the README banner.

Reads the placements in `scq_hardware_layouts_HAL/placements/` and the check
matrices in `processor_codes/mitten/`, and writes `assets/mitten_layouts.svg`.

Only couplers whose endpoints are within Manhattan distance 2 cells of each
other in the compact frame are drawn (note this is endpoint separation, not
HAL's `avg_coupler_length`, which measures routed path length):
44% / 27% / 26% of the couplers in the three panels. The omitted ones are real
and long-range (distances of 4, 5, 7, 8 and beyond) — every check has weight 9,
so
most of its couplers leave the 3x3 cluster entirely. They are left out only
because drawing them at flat opacity washes the interior into a flat grey.

Also writes `assets/mitten_150_layout.svg`: the [[150,30,10]] panel alone,
drawing only tier 0 (360 couplers, all nearest-neighbour) in solid grey, plus
the full 9-coupler support of the X-check nearest the chip centre in a pale
purple — one star standing in for the 720 higher-tier couplers instead of
drawing them all. Each star arc is a cubic Bezier bent to keep clear of
every node marker except its endpoints. This panel is light-mode only (no
dark media query) and carries no code label or caption, just the legend.
The tier of every coupler is parsed from the layout's
`tiers/metrics_*.csv` (contracting face-switch chains back to their
check-qubit endpoints) and checked to partition the Tanner graph exactly.
X-checks are red instead of orange in this panel.

`assets/mitten_150_layout_3d.svg` is the same panel drawn isometrically:
the tier-0 grid lies on a tilted plane and the star's higher-tier couplers
route through the space above it — via up, straight run at a height that
grows with the coupler's HAL tier, via down.

    python assets/make_banner.py
"""
import base64
import csv
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# dataviz categorical slots 1-3; validated all-pairs in light and dark
LIGHT = {"data": "#2a78d6", "x": "#eb6834", "z": "#1baf7a"}
DARK = {"data": "#3987e5", "x": "#d95926", "z": "#199e70"}

# Single-panel variant: X-checks take dataviz slot 8 (red) instead of slot 2.
# blue/red/aqua re-validated all-pairs both modes: hard gates pass; red↔aqua
# CVD ΔE sits in the 6-8 warn band, carried by shape (square vs diamond vs
# circle) + legend as the secondary encoding.
X_RED = {"light": "#e34948", "dark": "#e66767"}

CODES = [("mitten_150_30_10", "150,30,10"),
         ("mitten_500_100_16", "500,100,16"),
         ("mitten_975_195_24", "975,195,24")]

CELL = 14.0          # px per grid cell; keeps every node centre on an integer
GAP = 52.0
PAD = 26.0
LABEL_H = 32.0
LEGEND_H = 26.0
MAX_DIST = 2         # cells; couplers reaching further are omitted
R_DATA = 2.6
S_CHK = 4.2          # X-check square, side
S_DIA = 6.0          # Z-check diamond, full diagonal. Equal area to the square
                     # needs S_CHK*sqrt(2) = 5.94; 6.0 keeps the points on whole
                     # units (no rounding drift) at 102% of the square's area


def _n(v):
    return str(int(round(v)))


def _sub(x, y, c1x, c1y, ex, ey):
    """One relative quadratic subpath, minimal separators."""
    out = "M" + _n(x) + " " + _n(y) + "q"
    for i, v in enumerate((c1x, c1y, ex, ey)):
        t = _n(v)
        if i and not t.startswith("-"):
            out += " "
        out += t
    return out


def load(code):
    p = f"{ROOT}/scq_hardware_layouts_HAL/placements/{code}_placement.npz"
    d = np.load(p, allow_pickle=True)
    return d, json.loads(str(d["meta"]))


def edges(nkd):
    """Tanner edges (check row of vstack((Hx, Hz)), qubit)."""
    base = f"{ROOT}/processor_codes/mitten/[[{nkd}]]"
    H = np.vstack((np.load(f"{base}/Hx.npy"), np.load(f"{base}/Hz.npy")))
    r, c = np.nonzero(H)
    return r, c


def swatch(code, nkd, ox, oy, classify):
    """classify(check, qubit, dx, dy) -> group key for the coupler, or None
    to omit it. Returns ({key: path data}, nodes markup)."""
    d, meta = load(code)
    isd = d["is_data"].astype(bool)
    ni = d["node_index"].astype(int)
    blk = [str(b) for b in d["block"]]
    px = ox + d["x_placed"].astype(float) * CELL + CELL / 2.0
    py = oy + d["y_placed"].astype(float) * CELL + CELL / 2.0

    qpos, cpos = {}, {}
    for i in range(len(ni)):
        (qpos if isd[i] else cpos)[ni[i]] = (px[i], py[i])

    seg = {}
    for e, (r, q) in enumerate(zip(*(a.tolist() for a in edges(nkd)))):
        x1, y1 = cpos[r]
        x2, y2 = qpos[q]
        dx, dy = x2 - x1, y2 - y1
        key = classify(r, q, dx, dy)
        if key is None:
            continue
        L = (dx * dx + dy * dy) ** 0.5 or 1.0
        bow = 0.18 * L * (1 if (e + r) % 2 == 0 else -1)
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        seg.setdefault(key, []).append(
            _sub(x1, y1, mx - dy / L * bow - x1, my + dx / L * bow - y1, dx, dy))

    circles, squares, diamonds = [], [], []
    h = S_CHK / 2.0
    hd = S_DIA / 2.0
    for i in range(len(ni)):
        cx, cy = px[i], py[i]
        if isd[i]:
            circles.append(f'<circle cx="{_n(cx)}" cy="{_n(cy)}" r="{R_DATA}"/>')
        elif blk[i].startswith("x"):
            squares.append(f'<rect x="{cx - h:.1f}" y="{cy - h:.1f}" '
                           f'width="{S_CHK}" height="{S_CHK}" rx="0.8"/>')
        else:
            diamonds.append(f'<path d="M{_n(cx)} {cy - hd:.1f}L{cx + hd:.1f} {_n(cy)}'
                            f'L{_n(cx)} {cy + hd:.1f}L{cx - hd:.1f} {_n(cy)}Z"/>')

    nodes = (f'<g class="z">{"".join(diamonds)}</g>'
             f'<g class="x">{"".join(squares)}</g>'
             f'<g class="d">{"".join(circles)}</g>')
    return {k: "".join(v) for k, v in seg.items()}, nodes


_NODE = r"(?:Node\(index=(\d+), is_data=(True|False)\)|\('switch', (\d+)\))"
_PAIR = re.compile(r"\(" + _NODE + ", " + _NODE + r"\)")


def tier_couplers(code):
    """Map (check row, qubit) -> HAL tier, from `tiers/metrics_*.csv`.

    A tier stores each coupler either directly (check, qubit) or as a chain of
    face-switch segments check-sw-...-sw-qubit; chains are contracted with a
    union-find over the switch nodes.
    """
    csv.field_size_limit(1 << 30)
    tiers, t = {}, 0
    while True:
        p = f"{ROOT}/scq_hardware_layouts_HAL/{code}_layout/tiers/metrics_{t}.csv"
        if not os.path.exists(p):
            return tiers
        with open(p, encoding="utf-8") as f:
            blob = next(csv.DictReader(f))["routed_edge_order"]
        sw_edges, attach = [], []
        for m in _PAIR.finditer(blob):
            i1, d1, s1, i2, d2, s2 = m.groups()
            a = ("sw", int(s1)) if s1 else (int(i1), d1 == "True")
            b = ("sw", int(s2)) if s2 else (int(i2), d2 == "True")
            if a[0] != "sw" and b[0] != "sw":
                tiers[(b[0], a[0]) if a[1] else (a[0], b[0])] = t
            elif a[0] == "sw" and b[0] == "sw":
                sw_edges.append((a[1], b[1]))
            else:
                attach.append((a, b) if a[0] == "sw" else (b, a))

        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for u, v in sw_edges:
            parent[find(u)] = find(v)
        groups = {}
        for (_, sw), real in attach:
            groups.setdefault(find(sw), []).append(real)
        for reals in groups.values():
            assert len(reals) == 2 and reals[0][1] != reals[1][1], reals
            (chk, _), (q, _) = sorted(reals, key=lambda r: r[1])
            tiers[(chk, q)] = t
        t += 1


# legend typeface: Latin Modern Sans, embedded as a data-URI WOFF subset so
# viewers without the font installed still see it. The subset covers only the
# legend glyphs; regenerate after changing legend text with
#   python -m fontTools.subset <lmsans10-regular.otf> \
#     --text="data qubit X-check Z-check" \
#     --output-file=assets/lmsans_legend.woff --flavor=woff --no-hinting
KEY_FONT = ("'LM Sans Legend','Latin Modern Sans','LM Sans 10','CMU Sans Serif',"
            "ui-sans-serif,-apple-system,'Segoe UI',sans-serif")


def _legend_font_face():
    p = f"{ROOT}/assets/lmsans_legend.woff"
    if not os.path.exists(p):
        return ""
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return ("@font-face{font-family:'LM Sans Legend';"
            f"src:url(data:font/woff;base64,{b64}) format('woff')}}")


def _legend_mark(kind, cx, cy):
    if kind == "d":
        return f'<g class="d"><circle cx="{cx:.0f}" cy="{cy:.0f}" r="{R_DATA}"/></g>'
    if kind == "x":
        h = S_CHK / 2.0
        return (f'<g class="x"><rect x="{cx - h:.1f}" y="{cy - h:.1f}" '
                f'width="{S_CHK}" height="{S_CHK}" rx="0.8"/></g>')
    hd = S_DIA / 2.0
    return (f'<g class="z"><path d="M{cx:.0f} {cy - hd:.1f}L{cx + hd:.1f} {cy:.0f}'
            f'L{cx:.0f} {cy + hd:.1f}L{cx - hd:.1f} {cy:.0f}Z"/></g>')


def _legend_keys(lx, ly):
    out = []
    for kind, text in (("d", "data qubit"), ("x", "X-check"), ("z", "Z-check")):
        cx, cy = lx + 5, ly - 4
        out.append(_legend_mark(kind, cx, cy) +
                   f'<text class="key" x="{cx + 10:.0f}" y="{ly:.0f}">{text}</text>')
        lx += 15 + len(text) * 6.2 + 24
    return "".join(out)


def build(out):
    metas = [load(c)[1] for c, _ in CODES]
    ws = [m["layout"]["chip_w"] * CELL for m in metas]
    hs = [m["layout"]["chip_h"] * CELL for m in metas]
    W = sum(ws) + GAP * (len(CODES) - 1) + 2 * PAD
    H = max(hs) + 2 * PAD + LABEL_H + LEGEND_H

    paths, nodes, labels = [], [], []
    ox, base_y = PAD, PAD + max(hs)
    near = lambda r, q, dx, dy: (abs(dx) + abs(dy)) / CELL <= MAX_DIST or None
    for (code, nkd), w, h in zip(CODES, ws, hs):
        seg, nd = swatch(code, nkd, ox, base_y - h, near)
        p = seg[True]
        paths.append(p)
        nodes.append(nd)
        labels.append(f'<text class="lbl" x="{ox + w / 2:.0f}" y="{base_y + 21:.0f}">'
                      f'[[{nkd}]]</text>')
        ox += w + GAP

    ly = base_y + LABEL_H + 14
    legend = [_legend_keys(PAD + 2, ly)]
    legend.append(f'<text class="key dim" x="{W - PAD:.0f}" y="{ly:.0f}" text-anchor="end">'
                  f'HAL layouts &#183; only couplers with Manhattan distance '
                  f'&#8804;&#8202;2 cells (26&#8211;44% of all couplers)</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" role="img" aria-label="Compact-frame chip layouts of three mitten codes, drawn as knitted swatches of increasing size: 150,30,10 then 500,100,16 then 975,195,24">
<title>mitten code layouts</title>
<style>
  .s{{fill:none;stroke:#8f8a81;stroke-opacity:.55;stroke-width:1;stroke-linecap:round}}
  .d{{fill:{LIGHT["data"]}}} .x{{fill:{LIGHT["x"]}}} .z{{fill:{LIGHT["z"]}}}
  .lbl{{font:600 13px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#57534e;text-anchor:middle}}
  .key{{font:500 11.5px ui-sans-serif,-apple-system,Segoe UI,sans-serif;fill:#57534e}}
  .dim{{fill:#8a857c}}
  @media (prefers-color-scheme:dark){{
    .s{{stroke:#7f7a72;stroke-opacity:.6}}
    .d{{fill:{DARK["data"]}}} .x{{fill:{DARK["x"]}}} .z{{fill:{DARK["z"]}}}
    .lbl,.key{{fill:#b9b3a8}} .dim{{fill:#8a857c}}
  }}
</style>
<path class="s" d="{"".join(paths)}"/>
{"".join(nodes)}
{"".join(labels)}
{"".join(legend)}
</svg>
'''
    with open(out, "w", encoding="utf-8") as f:
        f.write(svg)
    return W, H, os.path.getsize(out)


def _single_data():
    """Shared inputs of the single-code panels: placement arrays, tier map,
    Tanner edges, and the centre-most X-check that carries the star."""
    code, nkd = CODES[0]
    d, meta = load(code)
    tiers = tier_couplers(code)
    r, q = edges(nkd)
    assert set(zip(r.tolist(), q.tolist())) == set(tiers), \
        "tier CSVs do not partition the Tanner graph"

    isd = d["is_data"].astype(bool)
    ni = d["node_index"].astype(int)
    blk = [str(b) for b in d["block"]]
    xp = d["x_placed"].astype(float)
    yp = d["y_placed"].astype(float)
    ccx, ccy = meta["layout"]["chip_w"] / 2.0, meta["layout"]["chip_h"] / 2.0
    hi = min((i for i in range(len(ni)) if not isd[i] and blk[i].startswith("x")),
             key=lambda i: (xp[i] - ccx) ** 2 + (yp[i] - ccy) ** 2)
    return code, nkd, d, meta, tiers, r, q, isd, ni, xp, yp, ni[hi]


def build_single(out):
    code, nkd, d, meta, tiers, r, q, isd, ni, xp, yp, hi_chk = _single_data()
    w = meta["layout"]["chip_w"] * CELL
    h = meta["layout"]["chip_h"] * CELL
    W = w + 2 * PAD

    seg, nodes = swatch(
        code, nkd, PAD, PAD,
        lambda r, q, dx, dy: None if r == hi_chk
        else (0 if tiers[(r, q)] == 0 else None))

    # star arcs must not pass over other node markers: each is a cubic
    # Bezier whose two perpendicular control offsets are searched so the
    # curve clears every non-endpoint marker by >= 5px where possible
    # (markers reach ~3px from centre), gentlest passing curve preferred
    nx = PAD + xp * CELL + CELL / 2.0
    ny = PAD + yp * CELL + CELL / 2.0
    qpos = {n: (a, b) for n, a, b, s in zip(ni, nx, ny, isd) if s}
    cpos = {n: (a, b) for n, a, b, s in zip(ni, nx, ny, isd) if not s}
    ts = np.linspace(0.02, 0.98, 81)
    fs = [s * f for f in (.03, .07, .12, .18, .25, .33, .42, .55, .7, .9)
          for s in (1, -1)]
    star = []
    for rr, qq in zip(r.tolist(), q.tolist()):
        if rr != hi_chk:
            continue
        x1, y1 = cpos[rr]
        x2, y2 = qpos[qq]
        dx, dy = x2 - x1, y2 - y1
        L = (dx * dx + dy * dy) ** 0.5 or 1.0
        near = ((nx - x1) ** 2 + (ny - y1) ** 2 > 9) & \
               ((nx - x2) ** 2 + (ny - y2) ** 2 > 9)
        ex, ey = nx[near], ny[near]
        def clearance(c):
            c1x, c1y, c2x, c2y = c
            bx = ((1 - ts) ** 3 * x1 + 3 * (1 - ts) ** 2 * ts * c1x
                  + 3 * (1 - ts) * ts ** 2 * c2x + ts ** 3 * x2)
            by = ((1 - ts) ** 3 * y1 + 3 * (1 - ts) ** 2 * ts * c1y
                  + 3 * (1 - ts) * ts ** 2 * c2y + ts ** 3 * y2)
            return ((bx[:, None] - ex) ** 2 + (by[:, None] - ey) ** 2).min()

        cands = []
        for u1 in (.22, .33, .44):
            for f1 in fs:
                c1x = x1 + u1 * dx - dy * f1
                c1y = y1 + u1 * dy + dx * f1
                for u2 in (.56, .67, .78):
                    for f2 in fs:
                        c2x = x1 + u2 * dx - dy * f2
                        c2y = y1 + u2 * dy + dx * f2
                        c = (c1x, c1y, c2x, c2y)
                        cands.append((clearance(c) - 9 * (abs(f1) + abs(f2)), c))
        cands.sort(key=lambda t: -t[0])

        # greedy refinement from the best few starts: nudge control
        # coordinates singly and jointly to escape corridor local optima
        cap = 36.0  # stop improving past 6px of clearance
        moves = [(i, s) for i in range(4) for s in (1, -1)] + \
                [(None, (ax, sy)) for ax in (0, 1) for sy in (1, -1)]
        best, pick = -1.0, None
        for _, start in cands[:12]:
            cur = list(start)
            for _ in range(3):
                for i, s in moves:
                    for step in (12, 8, 4, 2, 1):
                        trial = cur.copy()
                        if i is None:
                            ax, sy = s
                            trial[ax] += sy * step
                            trial[ax + 2] += sy * step
                        else:
                            trial[i] += s * step
                        if min(clearance(trial), cap) > min(clearance(cur), cap):
                            cur = trial
            got = clearance(cur)
            if got > best:
                best, pick = got, cur
            if best >= cap:
                break
        c1x, c1y, c2x, c2y = pick
        star.append(f'M{_n(x1)} {_n(y1)}C{_n(c1x)} {_n(c1y)} '
                    f'{_n(c2x)} {_n(c2y)} {_n(x2)} {_n(y2)}')

    web = (f'<path class="s" d="{seg[0]}"/>'
           f'<path class="hi" d="{"".join(star)}"/>')

    ly = PAD + h + 20
    H = ly + 12
    legend = _legend_keys(PAD + 2, ly)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" role="img" aria-label="Compact-frame chip layout of the mitten 150,30,10 code drawn as a knitted swatch; solid grey couplers are the nearest-neighbour tier, and the nine couplers of one X-check are shown in pale purple to illustrate the long-range structure">
<title>mitten [[150,30,10]] layout</title>
<style>
  {_legend_font_face()}
  .s{{fill:none;stroke:#8f8a81;stroke-opacity:.55;stroke-width:1;stroke-linecap:round}}
  .hi{{fill:none;stroke:#b0a4e6;stroke-opacity:.6;stroke-width:1;stroke-linecap:round}}
  .d{{fill:{LIGHT["data"]}}} .x{{fill:{X_RED["light"]}}} .z{{fill:{LIGHT["z"]}}}
  .key{{font:500 11.5px {KEY_FONT};fill:#57534e}}
</style>
{web}
{nodes}
{legend}
</svg>
'''
    with open(out, "w", encoding="utf-8") as f:
        f.write(svg)
    return W, H, os.path.getsize(out)


# oblique projection for the 3D panel: plan (px, py) at height z ->
# screen (sx, sy); rows stay horizontal, the plane recedes by shearing
# right and compressing vertically, +z is straight up on screen
OBX, OBY = 0.6, 0.58


def _rounded(pts, rad=5.0):
    """Polyline through pts with corners rounded by quadratic shortcuts."""
    out = f"M{_n(pts[0][0])} {_n(pts[0][1])}"
    for (ax, ay), (bx, by), (cx, cy) in zip(pts, pts[1:], pts[2:]):
        d1 = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 or 1.0
        d2 = ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5 or 1.0
        rr = min(rad, d1 / 2, d2 / 2)
        out += (f"L{_n(bx - (bx - ax) / d1 * rr)} {_n(by - (by - ay) / d1 * rr)}"
                f"Q{_n(bx)} {_n(by)} "
                f"{_n(bx + (cx - bx) / d2 * rr)} {_n(by + (cy - by) / d2 * rr)}")
    out += f"L{_n(pts[-1][0])} {_n(pts[-1][1])}"
    return out


def _sample_rounded(pts, rad=5.0, step=1.2):
    """Dense point samples along the rounded polyline of _rounded."""
    out = [np.array(pts[0], float)]
    prev = out[0]
    for (ax, ay), (bx, by), (cx, cy) in zip(pts, pts[1:], pts[2:]):
        a, b, c = (np.array(v, float) for v in ((ax, ay), (bx, by), (cx, cy)))
        d1 = np.hypot(*(b - a)) or 1.0
        d2 = np.hypot(*(c - b)) or 1.0
        rr = min(rad, d1 / 2, d2 / 2)
        pin, pout = b - (b - a) / d1 * rr, b + (c - b) / d2 * rr
        n = max(2, int(np.hypot(*(pin - prev)) / step))
        out += [prev + (pin - prev) * k / n for k in range(1, n + 1)]
        out += [(1 - t) ** 2 * pin + 2 * t * (1 - t) * b + t * t * pout
                for t in np.linspace(1 / 6, 1, 6)]
        prev = pout
    last = np.array(pts[-1], float)
    n = max(2, int(np.hypot(*(last - prev)) / step))
    out += [prev + (last - prev) * k / n for k in range(1, n + 1)]
    return out


def _sample_quad(p0, c, p1, step=1.2):
    """Dense point samples along one quadratic Bezier."""
    p0, c, p1 = (np.array(v, float) for v in (p0, c, p1))
    n = max(8, int((np.hypot(*(c - p0)) + np.hypot(*(p1 - c))) / step))
    return [(1 - t) ** 2 * p0 + 2 * t * (1 - t) * c + t * t * p1
            for t in np.linspace(0, 1, n)]


def _yarn(pts, amp=1.1, wave=4.5):
    """Soft core plus two plies twisting tightly around the sampled
    centreline, tapered so the plies meet the endpoints. Returns path
    strings (core, ply A, ply B)."""
    p = np.asarray(pts)
    keep = np.concatenate(([True], np.hypot(*np.diff(p, axis=0).T) > 1e-6))
    p = p[keep]
    s = np.concatenate(([0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))))
    tan = np.gradient(p, s, axis=0)
    norm = np.hypot(tan[:, 0], tan[:, 1])
    norm[norm == 0] = 1
    off = (amp * np.sin(2 * np.pi * s / wave)
           * np.minimum(1, np.minimum(s, s[-1] - s) / 4))
    ox, oy = -tan[:, 1] / norm * off, tan[:, 0] / norm * off

    def fmt(xs, ys):
        return "M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in zip(xs, ys))

    return (fmt(p[:, 0], p[:, 1]),
            fmt(p[:, 0] + ox, p[:, 1] + oy), fmt(p[:, 0] - ox, p[:, 1] - oy))


def build_single_3d(out, yarn=False):
    code, nkd, d, meta, tiers, r, q, isd, ni, xp, yp, hi_chk = _single_data()
    w_plan = meta["layout"]["chip_w"] * CELL
    h_plan = meta["layout"]["chip_h"] * CELL

    px = xp * CELL + CELL / 2.0
    py = yp * CELL + CELL / 2.0
    qpos, cpos = {}, {}
    for i in range(len(ni)):
        (qpos if isd[i] else cpos)[ni[i]] = (px[i], py[i])

    def height(t):
        return (1.8 + 1.2 * t) * CELL

    # top headroom so the aerial runs stay inside the viewBox
    rise = max((height(tiers[(rr, qq)])
                - min(cpos[rr][1], qpos[qq][1]) * OBY
                for rr, qq in zip(r.tolist(), q.tolist())
                if rr == hi_chk and tiers[(rr, qq)] > 0), default=0)
    ox, oy = PAD + h_plan * OBX, PAD + max(0.0, rise)

    def proj(px, py, z=0.0):
        return ox + px - py * OBX, oy + py * OBY - z

    floor = " ".join(f"{_n(sx)},{_n(sy)}" for sx, sy in
                     (proj(0, 0), proj(w_plan, 0),
                      proj(w_plan, h_plan), proj(0, h_plan)))

    def inplane_pts(x1, y1, x2, y2, bow):
        """Projected (start, control, end) of an in-plane bowed coupler;
        the affine projection maps the plan-space quadratic to the same
        quadratic on projected points."""
        dx, dy = x2 - x1, y2 - y1
        L = (dx * dx + dy * dy) ** 0.5 or 1.0
        cx, cy = (x1 + x2) / 2 - dy / L * bow, (y1 + y2) / 2 + dx / L * bow
        return proj(x1, y1), proj(cx, cy), proj(x2, y2)

    def inplane(x1, y1, x2, y2, bow):
        (sx1, sy1), (scx, scy), (sx2, sy2) = inplane_pts(x1, y1, x2, y2, bow)
        return _sub(sx1, sy1, scx - sx1, scy - sy1, sx2 - sx1, sy2 - sy1)

    grid, star, strands = [], [], []
    for e, (rr, qq) in enumerate(zip(r.tolist(), q.tolist())):
        x1, y1 = cpos[rr]
        x2, y2 = qpos[qq]
        t = tiers[(rr, qq)]
        L = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 or 1.0
        bow = 0.18 * L * (1 if (e + rr) % 2 == 0 else -1)
        if rr == hi_chk:
            if t == 0:
                if yarn:
                    strands.append(_sample_quad(*inplane_pts(x1, y1, x2, y2, bow)))
                else:
                    star.append(inplane(x1, y1, x2, y2, bow))
            else:
                # via up, run straight at the tier's height, via down
                h = height(t)
                corners = [proj(x1, y1), proj(x1, y1, h),
                           proj(x2, y2, h), proj(x2, y2)]
                if yarn:
                    strands.append(_sample_rounded(corners))
                else:
                    star.append(_rounded(corners))
        elif t == 0:
            grid.append(inplane(x1, y1, x2, y2, bow))

    if yarn:
        plies = [_yarn(pts) for pts in strands]
        star_markup = (f'<path class="yc" d="{"".join(c for c, _, _ in plies)}"/>'
                       f'<path class="y2" d="{"".join(b for _, _, b in plies)}"/>'
                       f'<path class="y1" d="{"".join(a for _, a, _ in plies)}"/>')
    else:
        star_markup = f'<path class="hi" d="{"".join(star)}"/>'

    circles, squares, diamonds, anchors = [], [], [], []
    hs = S_CHK / 2.0
    hd = S_DIA / 2.0
    star_ends = {("c", hi_chk)} | {("q", qq) for rr, qq
                                   in zip(r.tolist(), q.tolist()) if rr == hi_chk}
    blk = [str(b) for b in d["block"]]
    for i in range(len(ni)):
        sx, sy = proj(px[i], py[i])
        if isd[i]:
            m = f'<circle cx="{_n(sx)}" cy="{_n(sy)}" r="{R_DATA}"/>'
            circles.append(m)
            if ("q", ni[i]) in star_ends:
                anchors.append(f'<g class="d">{m}</g>')
        elif blk[i].startswith("x"):
            m = (f'<rect x="{sx - hs:.1f}" y="{sy - hs:.1f}" '
                 f'width="{S_CHK}" height="{S_CHK}" rx="0.8"/>')
            squares.append(m)
            if ("c", ni[i]) in star_ends:
                anchors.append(f'<g class="x">{m}</g>')
        else:
            diamonds.append(f'<path d="M{_n(sx)} {sy - hd:.1f}L{sx + hd:.1f} {_n(sy)}'
                            f'L{_n(sx)} {sy + hd:.1f}L{sx - hd:.1f} {_n(sy)}Z"/>')
    nodes = (f'<g class="z">{"".join(diamonds)}</g>'
             f'<g class="x">{"".join(squares)}</g>'
             f'<g class="d">{"".join(circles)}</g>')

    W = ox + w_plan + PAD
    ly = oy + h_plan * OBY + 24
    H = ly + 12
    legend = _legend_keys(PAD + 2, ly)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" role="img" aria-label="Isometric view of the mitten 150,30,10 chip: the nearest-neighbour coupler grid lies on a tilted plane, and the nine couplers of one X-check route through the space above it, higher HAL tiers higher up">
<title>mitten [[150,30,10]] layout, 3D</title>
<style>
  {_legend_font_face()}
  .s{{fill:none;stroke:#8f8a81;stroke-opacity:.55;stroke-width:1;stroke-linecap:round}}
  .hi{{fill:none;stroke:#b0a4e6;stroke-opacity:.65;stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round}}
  .yc{{fill:none;stroke:#b0a4e6;stroke-opacity:.3;stroke-width:3.4;stroke-linecap:round;stroke-linejoin:round}}
  .y1{{fill:none;stroke:#b0a4e6;stroke-opacity:.9;stroke-width:1.4;stroke-linecap:round;stroke-linejoin:round}}
  .y2{{fill:none;stroke:#9a8cd9;stroke-opacity:.9;stroke-width:1.4;stroke-linecap:round;stroke-linejoin:round}}
  .d{{fill:{LIGHT["data"]}}} .x{{fill:{X_RED["light"]}}} .z{{fill:{LIGHT["z"]}}}
  .key{{font:500 11.5px {KEY_FONT};fill:#57534e}}
  .floor{{fill:none;stroke:#dcd9d2;stroke-width:.8}}
</style>
<polygon class="floor" points="{floor}"/>
<path class="s" d="{"".join(grid)}"/>
{nodes}
{star_markup}
{"".join(anchors)}
{legend}
</svg>
'''
    with open(out, "w", encoding="utf-8") as f:
        f.write(svg)
    return W, H, os.path.getsize(out)


if __name__ == "__main__":
    for fn, name in ((build, "mitten_layouts"), (build_single, "mitten_150_layout"),
                     (build_single_3d, "mitten_150_layout_3d"),
                     (lambda o: build_single_3d(o, yarn=True),
                      "mitten_150_layout_3d_yarn")):
        out = f"{ROOT}/assets/{name}.svg"
        w, h, size = fn(out)
        print(f"{out}\n  {w:.0f}x{h:.0f}  ratio {w / h:.2f}  {size / 1024:.0f} KB")
