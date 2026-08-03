"""Render mitten-code compact-frame layouts as the README banner.

Reads the placements in `scq_hardware_layouts_HAL/placements/` and the check
matrices in `processor_codes/mitten/`, and writes `assets/mitten_layouts.svg`.

Only nearest-neighbour couplers (Manhattan distance <= 2 cells in the compact
frame) are drawn — the long-range couplers are real but wash the interior into
a flat grey at banner scale.

    python assets/make_banner.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# dataviz categorical slots 1-3; validated all-pairs in light and dark
LIGHT = {"data": "#2a78d6", "x": "#eb6834", "z": "#1baf7a"}
DARK = {"data": "#3987e5", "x": "#d95926", "z": "#199e70"}

CODES = [("mitten_150_30_10", "150,30,10"),
         ("mitten_500_100_16", "500,100,16"),
         ("mitten_975_195_24", "975,195,24")]

CELL = 14.0          # px per grid cell; keeps every node centre on an integer
GAP = 52.0
PAD = 26.0
LABEL_H = 32.0
LEGEND_H = 26.0
MAX_SPAN = 2         # cells; longer couplers are omitted
R_DATA = 2.6
S_CHK = 4.2


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


def swatch(code, nkd, ox, oy):
    d, meta = load(code)
    isd = d["is_data"].astype(bool)
    ni = d["node_index"].astype(int)
    blk = [str(b) for b in d["block"]]
    px = ox + d["x_placed"].astype(float) * CELL + CELL / 2.0
    py = oy + d["y_placed"].astype(float) * CELL + CELL / 2.0

    qpos, cpos = {}, {}
    for i in range(len(ni)):
        (qpos if isd[i] else cpos)[ni[i]] = (px[i], py[i])

    seg = []
    for e, (r, q) in enumerate(zip(*(a.tolist() for a in edges(nkd)))):
        x1, y1 = cpos[r]
        x2, y2 = qpos[q]
        dx, dy = x2 - x1, y2 - y1
        if (abs(dx) + abs(dy)) / CELL > MAX_SPAN:
            continue
        L = (dx * dx + dy * dy) ** 0.5 or 1.0
        bow = 0.18 * L * (1 if (e + r) % 2 == 0 else -1)
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        seg.append(_sub(x1, y1, mx - dy / L * bow - x1, my + dx / L * bow - y1, dx, dy))

    circles, squares, diamonds = [], [], []
    h = S_CHK / 2.0
    for i in range(len(ni)):
        cx, cy = px[i], py[i]
        if isd[i]:
            circles.append(f'<circle cx="{_n(cx)}" cy="{_n(cy)}" r="{R_DATA}"/>')
        elif blk[i].startswith("x"):
            squares.append(f'<rect x="{cx - h:.1f}" y="{cy - h:.1f}" '
                           f'width="{S_CHK}" height="{S_CHK}" rx="0.8"/>')
        else:
            diamonds.append(f'<path d="M{_n(cx)} {cy - h:.1f}L{cx + h:.1f} {_n(cy)}'
                            f'L{_n(cx)} {cy + h:.1f}L{cx - h:.1f} {_n(cy)}Z"/>')

    nodes = (f'<g class="z">{"".join(diamonds)}</g>'
             f'<g class="x">{"".join(squares)}</g>'
             f'<g class="d">{"".join(circles)}</g>')
    return "".join(seg), nodes


def build(out):
    metas = [load(c)[1] for c, _ in CODES]
    ws = [m["layout"]["chip_w"] * CELL for m in metas]
    hs = [m["layout"]["chip_h"] * CELL for m in metas]
    W = sum(ws) + GAP * (len(CODES) - 1) + 2 * PAD
    H = max(hs) + 2 * PAD + LABEL_H + LEGEND_H

    paths, nodes, labels = [], [], []
    ox, base_y = PAD, PAD + max(hs)
    for (code, nkd), w, h in zip(CODES, ws, hs):
        p, nd = swatch(code, nkd, ox, base_y - h)
        paths.append(p)
        nodes.append(nd)
        labels.append(f'<text class="lbl" x="{ox + w / 2:.0f}" y="{base_y + 21:.0f}">'
                      f'[[{nkd}]]</text>')
        ox += w + GAP

    ly = base_y + LABEL_H + 14
    lx = PAD + 2
    legend = []
    for kind, text in (("d", "data qubit"), ("x", "X-check"), ("z", "Z-check")):
        cx, cy = lx + 5, ly - 4
        if kind == "d":
            mark = f'<g class="d"><circle cx="{cx:.0f}" cy="{cy:.0f}" r="{R_DATA}"/></g>'
        elif kind == "x":
            mark = (f'<g class="x"><rect x="{cx - 2.1:.1f}" y="{cy - 2.1:.1f}" '
                    f'width="4.2" height="4.2" rx="0.8"/></g>')
        else:
            mark = (f'<g class="z"><path d="M{cx:.0f} {cy - 2.1:.1f}L{cx + 2.1:.1f} {cy:.0f}'
                    f'L{cx:.0f} {cy + 2.1:.1f}L{cx - 2.1:.1f} {cy:.0f}Z"/></g>')
        legend.append(mark + f'<text class="key" x="{cx + 10:.0f}" y="{ly:.0f}">{text}</text>')
        lx += 15 + len(text) * 6.2 + 24
    legend.append(f'<text class="key dim" x="{W - PAD:.0f}" y="{ly:.0f}" text-anchor="end">'
                  f'compact-frame HAL layouts &#183; nearest-neighbour couplers</text>')

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


if __name__ == "__main__":
    out = f"{ROOT}/assets/mitten_layouts.svg"
    w, h, size = build(out)
    print(f"{out}\n  {w:.0f}x{h:.0f}  ratio {w / h:.2f}  {size / 1024:.0f} KB")
