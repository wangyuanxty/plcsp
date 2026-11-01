"""Emit fig_method.svg: four agents, network glyphs, two-stage machine encoder."""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent

INK = "#1A1A1A"
BLUE, BLUE_D = "#E8F0FB", "#0F4D92"
YEL, YEL_D = "#FFF6DA", "#B07D00"
GRN, GRN_D = "#E7F4E7", "#2E7D32"
PUR, PUR_D = "#F3E8F2", "#9A4D8E"
ORG, ORG_D = "#FFE6CC", "#C55A11"
RED_D = "#B64342"

rows = [
    ("operation agent", BLUE, BLUE_D, "Set Transformer"),
    ("machine agent", YEL, YEL_D, None),
    ("AGV agent", GRN, GRN_D, "Set Transformer"),
]


def glyph_attn(x: float, y: float, colour: str, w: float = 40) -> str:
    """three candidate nodes feeding one pooled node"""
    steps = w / 3.0
    left = x + steps * 0.5
    mid = x + w * 0.5
    right = x + w - steps * 0.5
    nodes = [(left, y + 20), (mid, y + 20), (right, y + 20), (mid, y + 6)]
    lines = "".join(
        f'<line x1="{nodes[i][0]:.1f}" y1="{nodes[i][1] - 3.6:.1f}" '
        f'x2="{mid:.1f}" y2="{y + 9.6:.1f}" stroke="{colour}" stroke-width="0.6"/>'
        for i in range(3))
    dots = "".join(
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.4" fill="#FFFFFF" '
        f'stroke="{colour}" stroke-width="0.9"/>' for px, py in nodes)
    return lines + dots


def glyph_mlp(x: float, y: float, colour: str, w: float = 40) -> str:
    cols = [[(x + 5, y + 5), (x + 5, y + 13), (x + 5, y + 21)],
            [(x + w * 0.5, y + 8), (x + w * 0.5, y + 18)],
            [(x + w - 5, y + 13)]]
    lines = ""
    for a in cols[0]:
        for b in cols[1]:
            lines += (f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" '
                      f'y2="{b[1]:.1f}" stroke="{colour}" stroke-width="0.5" '
                      f'opacity="0.75"/>')
    for b in cols[1]:
        c = cols[2][0]
        lines += (f'<line x1="{b[0]:.1f}" y1="{b[1]:.1f}" x2="{c[0]:.1f}" '
                  f'y2="{c[1]:.1f}" stroke="{colour}" stroke-width="0.5" '
                  f'opacity="0.75"/>')
    dots = ""
    for col in cols:
        for px, py in col:
            dots += (f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.9" fill="#FFFFFF" '
                     f'stroke="{colour}" stroke-width="0.9"/>')
    return lines + dots


def glyph_stack(x: float, y: float, w: float, h: float, n: int, colour: str) -> str:
    """n offset rectangles: a stack of attention blocks"""
    out = ""
    for i in range(n - 1, -1, -1):
        out += (f'<rect x="{x + i * 2.6:.1f}" y="{y - i * 2.6:.1f}" width="{w}" '
                f'height="{h}" rx="2" fill="#FFFFFF" stroke="{colour}" '
                f'stroke-width="0.7" opacity="{0.45 if i else 1:.2f}"/>')
    return out


def glyph_strip(x: float, y: float, cell: float, colour: str) -> str:
    """one feature vector per token"""
    filled = {0, 2, 3, 7, 8, 11}
    out = ""
    for i in range(12):
        fill = colour if i in filled else "#FFFFFF"
        out += (f'<rect x="{x + i * cell:.1f}" y="{y}" width="{cell - 0.8:.1f}" '
                f'height="{cell * 1.9:.1f}" fill="{fill}" stroke="{colour}" '
                f'stroke-width="0.4"/>')
    return out


def glyph_bars(x: float, y: float, w: float, h: float, dark: str) -> str:
    """five candidate scores, the chosen one filled"""
    heights = (0.45, 0.7, 0.35, 1.0, 0.55)
    step = w / len(heights)
    out = ""
    for k, frac in enumerate(heights):
        bh = h * frac
        bx = x + k * step + step * 0.2
        bw = step * 0.6
        fill = dark if k == 3 else "#D9D9D9"
        out += (f'<rect x="{bx:.1f}" y="{y + h - bh:.1f}" width="{bw:.1f}" '
                f'height="{bh:.1f}" fill="{fill}" stroke="{dark}" stroke-width="0.35"/>')
    return out


def build() -> str:
    s: list[str] = []
    add = s.append

    add('<?xml version="1.0" encoding="UTF-8"?>')
    add('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 570" width="7.1in" '
        'height="5.62in" font-family="Helvetica, Arial, sans-serif">')
    add('<defs>')
    for name, col in (("mB", "#4C78A8"), ("mR", RED_D), ("mG", GRN_D), ("mO", ORG_D)):
        add(f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5.5" '
            f'markerHeight="5.5" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
    add('</defs>')
    add('<rect width="720" height="570" fill="#FFFFFF"/>')
    add(f'<text x="8" y="16" font-size="9" font-weight="700" fill="{INK}">'
        '(a) Four agents and the flow of credit</text>')

    # selector
    add(f'<rect x="220" y="22" width="284" height="46" rx="8" fill="#FBE9E9" '
        f'stroke="{RED_D}" stroke-width="1.1"/>')
    add(f'<text x="362" y="36" font-size="8" font-weight="700" text-anchor="middle" '
        f'fill="#8E2F2E">rule selector agent</text>')
    for cx, label in ((265, "3 job rules"), (344, "4 machine rules"), (423, "3 AGV rules")):
        w = 66 if label != "4 machine rules" else 76
        add(f'<rect x="{cx - w // 2}" y="42" width="{w}" height="16" rx="8" fill="#FFFFFF" '
            f'stroke="{RED_D}" stroke-width="0.8"/>')
        add(f'<text x="{cx}" y="53" font-size="6" text-anchor="middle" '
            f'fill="#8E2F2E">{label}</text>')

    # panels
    add('<rect x="8" y="78" width="196" height="316" rx="10" fill="#F5F9FD" '
        'stroke="#BBD3E8" stroke-width="1"/>')
    add('<text x="18" y="94" font-size="7.4" font-weight="700" fill="#2C5F8A">'
        'State and candidates</text>')
    add('<rect x="212" y="78" width="292" height="316" rx="10" fill="#FAFAFA" '
        'stroke="#CFCFCF" stroke-width="1"/>')
    add('<text x="222" y="92" font-size="7.4" font-weight="700" fill="#444444">'
        'Actors: encode, score, mask</text>')
    add('<rect x="520" y="78" width="192" height="316" rx="10" fill="#F3F9F3" '
        'stroke="#BDDCC0" stroke-width="1"/>')
    add('<text x="530" y="94" font-size="7.4" font-weight="700" fill="#2E6B33">'
        'Shop environment</text>')

    ys = (104, 196, 288)          # row tops
    H = 52                        # box height
    names = ("operation agent", "machine agent", "AGV agent")
    fills = ((BLUE, BLUE_D), (YEL, YEL_D), (GRN, GRN_D))

    # token cards
    for k, y in enumerate(ys):
        fill, dark = fills[k]
        base = 96
        add(f'<rect x="20" y="{y + 6}" width="140" height="40" rx="4" fill="#FFFFFF" '
            f'stroke="#9C9C9C" stroke-width="0.8"/>')
        add(f'<rect x="17" y="{y + 3}" width="140" height="40" rx="4" fill="#FFFFFF" '
            f'stroke="#9C9C9C" stroke-width="0.8"/>')
        add(f'<rect x="14" y="{y}" width="140" height="40" rx="4" fill="{fill}" '
            f'stroke="{dark}" stroke-width="0.9"/>')
        add(f'<text x="84" y="{y + 13}" font-size="6.2" text-anchor="middle" '
            f'fill="{dark}">{names[k].split()[0]} tokens</text>')
        add(glyph_strip(48, y + 20, 6, dark))

    add('<text x="84" y="348" font-size="5.6" text-anchor="middle" fill="#777777">'
        'one token per candidate</text>')
    add('<text x="84" y="360" font-size="5.6" text-anchor="middle" fill="#777777">'
        'the set size varies with the step</text>')

    # agent rows
    for k, y in enumerate(ys):
        fill, dark = fills[k]
        add(f'<text x="262" y="{y - 3}" font-size="6.4" font-weight="700" '
            f'fill="{dark}">{names[k]}</text>')

        # encoder: a stack of attention blocks
        add(f'<rect x="220" y="{y}" width="92" height="{H}" rx="4" fill="{fill}" '
            f'stroke="{dark}" stroke-width="1"/>')
        if k == 1:
            add(glyph_stack(226, y + 12, 34, 24, 1, dark))
            add(glyph_attn(227, y + 4, dark, 32))
            add(f'<path d="M262 {y + 24} H266" stroke="#4C78A8" stroke-width="1" '
                f'marker-end="url(#mB)"/>')
            add(glyph_stack(270, y + 12, 34, 24, 3, dark))
            add(glyph_attn(271, y + 4, dark, 32))
            add(f'<text x="243" y="{y + 47}" font-size="5.2" text-anchor="middle" '
                f'fill="{dark}">within a machine</text>')
            add(f'<text x="287" y="{y + 47}" font-size="5.2" text-anchor="middle" '
                f'fill="{dark}">across machines</text>')
        else:
            nb = 2 if k == 0 else 1
            add(glyph_stack(256, y + 22, 28, 18, nb, dark))
            add(glyph_attn(257, y + 11, dark, 26))
            add(f'<text x="270" y="{y + 45}" font-size="5.4" text-anchor="middle" '
                f'fill="{dark}">{nb} block{"s" if nb > 1 else ""}</text>')

        # actor
        add(f'<rect x="324" y="{y}" width="76" height="{H}" rx="4" fill="#FFFFFF" '
            f'stroke="{dark}" stroke-width="1.1"/>')
        add(glyph_mlp(342, y + 5, dark, 40))
        add(f'<text x="362" y="{y + 41}" font-size="5.8" text-anchor="middle" '
            f'fill="{dark}">actor</text>')
        add(f'<text x="362" y="{y + 49}" font-size="5.4" text-anchor="middle" '
            f'fill="{dark}">scores candidates</text>')

        # output
        add(f'<rect x="412" y="{y}" width="88" height="{H}" rx="4" fill="{PUR}" '
            f'stroke="{PUR_D}" stroke-width="0.9"/>')
        add(f'<text x="456" y="{y + 12}" font-size="5.6" text-anchor="middle" '
            f'fill="{PUR_D}">masked softmax</text>')
        add(glyph_bars(420, y + 17, 72, 24, dark))
        add(f'<text x="456" y="{y + 49}" font-size="5.4" text-anchor="middle" '
            f'fill="{PUR_D}">{names[k].split()[0]} action</text>')

        # forward arrows: tokens -> encoder, encoder -> actor, actor -> output
        yc = y + H / 2
        add(f'<path d="M154 {yc:.0f} H216" stroke="#4C78A8" stroke-width="1" '
            f'marker-end="url(#mB)"/>')
        add(f'<path d="M314 {yc - 6:.0f} H322" stroke="#4C78A8" stroke-width="1" '
            f'marker-end="url(#mB)"/>')
        add(f'<path d="M402 {yc - 6:.0f} H410" stroke="#4C78A8" stroke-width="1" '
            f'marker-end="url(#mB)"/>')
        # rule prior: red, enters the lower half of the actor
        add(f'<path d="M318 {yc + 8:.0f} H322" stroke="{RED_D}" stroke-width="1" '
            f'stroke-dasharray="4 3" marker-end="url(#mR)"/>')
        # shared advantage: green, enters the lower half on the right
        add(f'<path d="M406 {yc + 8:.0f} H402" stroke="{GRN_D}" stroke-width="1" '
            f'stroke-dasharray="4 3" marker-end="url(#mG)"/>')

    # red bus and green bus
    add(f'<path d="M318 68 V{ys[-1] + H - 8}" fill="none" stroke="{RED_D}" '
        f'stroke-width="0.9" stroke-dasharray="4 3"/>')
    add(f'<path d="M406 {ys[-1] + H - 8} V92" fill="none" stroke="{GRN_D}" '
        f'stroke-width="0.9" stroke-dasharray="4 3"/>')

    # advantage node
    add(f'<rect x="300" y="344" width="164" height="28" rx="5" fill="{GRN}" '
        f'stroke="{GRN_D}" stroke-width="1"/>')
    add(f'<text x="382" y="362" font-size="6.6" font-weight="700" '
        f'text-anchor="middle" fill="{GRN_D}">group-relative advantage</text>')

    # environment column
    add(f'<rect x="530" y="100" width="172" height="30" rx="5" fill="{ORG}" '
        f'stroke="{ORG_D}" stroke-width="1"/>')
    add(f'<text x="616" y="119" font-size="7" font-weight="700" text-anchor="middle" '
        f'fill="#8F400C">joint action</text>')
    add(f'<rect x="530" y="146" width="172" height="30" rx="5" fill="{BLUE}" '
        f'stroke="{BLUE_D}" stroke-width="1"/>')
    add(f'<text x="616" y="165" font-size="7" font-weight="700" text-anchor="middle" '
        f'fill="{BLUE_D}">shop update</text>')
    add(f'<path d="M616 130 V144" stroke="#4C78A8" stroke-width="1" marker-end="url(#mB)"/>')

    add(f'<rect x="530" y="192" width="172" height="66" rx="5" fill="#FDEEE2" '
        f'stroke="{ORG_D}" stroke-width="1"/>')
    add(f'<text x="616" y="206" font-size="6.6" font-weight="700" text-anchor="middle" '
        f'fill="#A34A0E">per-agent costs</text>')
    add(f'<text x="538" y="220" font-size="5.8" fill="#A34A0E">operation: makespan + idle</text>')
    add(f'<text x="538" y="231" font-size="5.8" fill="#A34A0E">machine: PEC + transport</text>')
    add(f'<text x="538" y="242" font-size="5.8" fill="#A34A0E">AGV: transport + delay</text>')
    add(f'<path d="M616 176 V190" stroke="#4C78A8" stroke-width="1" marker-end="url(#mB)"/>')

    add(f'<rect x="530" y="274" width="172" height="28" rx="5" fill="{GRN}" '
        f'stroke="{GRN_D}" stroke-width="1"/>')
    add(f'<text x="616" y="292" font-size="6.8" font-weight="700" text-anchor="middle" '
        f'fill="{GRN_D}">team return (scalar)</text>')
    add(f'<path d="M616 258 V272" stroke="#4C78A8" stroke-width="1" marker-end="url(#mB)"/>')

    # team return -> advantage
    add(f'<path d="M530 288 H472 V358 H466" fill="none" stroke="{GRN_D}" stroke-width="1" '
        f'stroke-dasharray="4 3" marker-end="url(#mG)"/>')
    # costs -> selector
    add(f'<path d="M530 225 H510 V58 H506" fill="none" stroke="{ORG_D}" stroke-width="1" '
        f'stroke-dasharray="4 3" marker-end="url(#mO)"/>')
    add(f'<text x="504" y="74" font-size="5.8" text-anchor="end" fill="{ORG_D}">segment reward</text>')

    # legend
    add('<rect x="8" y="512" width="704" height="20" rx="4" fill="#FAFAFA" '
        'stroke="#DDDDDD" stroke-width="0.8"/>')
    items = [("#4C78A8", "", "state and action flow"),
             (RED_D, "4 3", "rule prior, selector to each actor"),
             (GRN_D, "4 3", "team return to the shared advantage"),
             (ORG_D, "4 3", "per-agent costs to the selector")]
    x = 20
    for col, dash, text in items:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        add(f'<path d="M{x} 522 H{x + 22}" stroke="{col}" stroke-width="1.1"{d}/>')
        add(f'<text x="{x + 28}" y="525" font-size="6.2" fill="#333333">{text}</text>')
        x += 28 + len(text) * 3.6 + 18

    # training pipeline
    add(f'<text x="8" y="424" font-size="9" font-weight="700" fill="{INK}">'
        '(b) Training pipeline</text>')
    steps = [("group sampling", BLUE, BLUE_D, 8, 162),
             ("rollout", YEL, YEL_D, 186, 162),
             ("segment close", GRN, GRN_D, 364, 162),
             ("update", PUR, PUR_D, 542, 170)]
    for label, fill, dark, bx, bw in steps:
        add(f'<rect x="{bx}" y="438" width="{bw}" height="26" rx="5" fill="{fill}" '
            f'stroke="{dark}" stroke-width="1"/>')
        add(f'<text x="{bx + bw // 2}" y="455" font-size="7" font-weight="700" '
            f'text-anchor="middle" fill="{dark}">{label}</text>')
    for x0, x1 in ((170, 184), (348, 362), (526, 540)):
        add(f'<path d="M{x0} 451 H{x1}" stroke="#4C78A8" stroke-width="1" marker-end="url(#mB)"/>')
    add(f'<path d="M627 464 V488 H89 V466" fill="none" stroke="{RED_D}" stroke-width="0.9" '
        f'stroke-dasharray="4 3" marker-end="url(#mR)"/>')
    add(f'<text x="358" y="500" font-size="6" text-anchor="middle" fill="{RED_D}">next group</text>')

    add('</svg>')
    return "\n".join(s)


if __name__ == "__main__":
    (OUT / "fig_method.svg").write_text(build(), encoding="utf-8")
    print("wrote fig_method.svg")