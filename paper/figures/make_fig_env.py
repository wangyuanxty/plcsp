"""Emit fig_environment.svg: four panels, tinted backing panels, vector icons."""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent

BLUE, BLUE_D = "#E8F0FB", "#0F4D92"
YEL, YEL_D = "#FFF6DA", "#B07D00"
GRN, GRN_D = "#E7F4E7", "#2E7D32"
PUR, PUR_D = "#F3E8F2", "#9A4D8E"
ORG, ORG_D = "#FFE6CC", "#C55A11"
RED_D = "#B64342"
FLOW = "#4C78A8"


def machine(x: float, y: float) -> str:
    """machining centre: window, spindle, control panel, base with feet"""
    return (
        f'<rect x="{x}" y="{y}" width="30" height="26" rx="2" fill="#FFFFFF" '
        f'stroke="#5F5F5F" stroke-width="0.9"/>'
        f'<rect x="{x + 4}" y="{y + 4}" width="14" height="11" rx="1" fill="#DDEBFA" '
        f'stroke="#5F5F5F" stroke-width="0.6"/>'
        f'<rect x="{x + 9}" y="{y + 17}" width="4" height="6" fill="#C9C9C9" '
        f'stroke="#5F5F5F" stroke-width="0.5"/>'
        f'<rect x="{x + 21}" y="{y + 4}" width="6" height="8" rx="1" fill="#F0F0F0" '
        f'stroke="#5F5F5F" stroke-width="0.6"/>'
        f'<circle cx="{x + 24}" cy="{y + 15}" r="1.5" fill="#B64342"/>'
        f'<rect x="{x + 1}" y="{y + 26}" width="28" height="4" fill="#DCDCDC" '
        f'stroke="#5F5F5F" stroke-width="0.6"/>'
        f'<rect x="{x + 3}" y="{y + 30}" width="3" height="2" fill="#999999"/>'
        f'<rect x="{x + 24}" y="{y + 30}" width="3" height="2" fill="#999999"/>')


def station(x: float, y: float) -> str:
    """load/unload dock: platform with a dashed transfer slot and rollers"""
    rollers = "".join(
        f'<circle cx="{x + 5 + i * 9}" cy="{y + 16}" r="2.2" fill="#FFFFFF" '
        f'stroke="#7A4E6D" stroke-width="0.7"/>' for i in range(3))
    return (
        f'<rect x="{x}" y="{y}" width="28" height="12" rx="2" fill="{PUR}" '
        f'stroke="#7A4E6D" stroke-width="0.9"/>'
        f'<path d="M{x + 3} {y + 6} H{x + 25}" stroke="#7A4E6D" stroke-width="0.8" '
        f'stroke-dasharray="3 2"/>' + rollers)


def agv(x: float, y: float) -> str:
    """low platform carrier: pallet on top, four wheels"""
    wheels = "".join(
        f'<circle cx="{x + 5 + i * 8}" cy="{y + 16}" r="2.6" fill="#FFFFFF" '
        f'stroke="{GRN_D}" stroke-width="0.8"/>' for i in range(4))
    return (
        f'<rect x="{x + 6}" y="{y}" width="20" height="5" rx="1" fill="#EFEFEF" '
        f'stroke="#888888" stroke-width="0.6"/>'
        f'<rect x="{x}" y="{y + 5}" width="32" height="9" rx="2" fill="{GRN}" '
        f'stroke="{GRN_D}" stroke-width="0.9"/>'
        f'<circle cx="{x + 27}" cy="{y + 9}" r="1.5" fill="{GRN_D}"/>' + wheels)


def step_no(x: float, y: float, n: int, colour: str) -> str:
    return (f'<circle cx="{x}" cy="{y}" r="4.6" fill="#FFFFFF" stroke="{colour}" '
            f'stroke-width="0.8"/>'
            f'<text x="{x}" y="{y + 1.9}" font-size="5.4" text-anchor="middle" '
            f'fill="{colour}">{n}</text>')


def build() -> str:
    s: list[str] = []
    add = s.append
    add('<?xml version="1.0" encoding="UTF-8"?>')
    add('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 580" width="7.1in" '
        'height="5.72in" font-family="Helvetica, Arial, sans-serif">')
    add('<defs>')
    for name, col in (("mB", FLOW), ("mR", RED_D), ("mG", GRN_D), ("mO", ORG_D), ("mS", "#7A4E6D")):
        add(f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5.5" '
            f'markerHeight="5.5" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
    add('</defs>')
    add('<rect width="720" height="580" fill="#FFFFFF"/>')

    # ---- panel frames -----------------------------------------------------
    for x, y, w, h in ((8, 24, 344, 240), (368, 24, 344, 240),
                       (8, 280, 344, 292), (368, 280, 344, 292)):
        add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#FBFCFE" '
            f'stroke="#DDE6F0" stroke-width="1"/>')

    # ===================== (a) decisions ==================================
    add(f'<text x="18" y="42" font-size="8.6" font-weight="700" fill="#1A1A1A">'
        '(a) Four agents, three coupled decisions</text>')

    add(f'<rect x="104" y="56" width="152" height="24" rx="7" fill="#FBE9E9" '
        f'stroke="{RED_D}" stroke-width="1"/>')
    add(f'<text x="180" y="72" font-size="6.8" font-weight="700" text-anchor="middle" '
        f'fill="#8E2F2E">rule selector agent</text>')
    add(f'<path d="M180 80 V88" stroke="{RED_D}" stroke-width="0.9" stroke-dasharray="4 3"/>')
    add(f'<path d="M62 88 H298" stroke="{RED_D}" stroke-width="0.9" stroke-dasharray="4 3"/>')
    for cx in (62, 180, 298):
        add(f'<path d="M{cx} 88 V96" stroke="{RED_D}" stroke-width="0.9" '
            f'stroke-dasharray="4 3" marker-end="url(#mR)"/>')
    add(f'<text x="304" y="92" font-size="5.6" fill="{RED_D}">rule prior</text>')

    add(f'<rect x="18" y="100" width="88" height="34" rx="5" fill="{BLUE}" '
        f'stroke="{BLUE_D}" stroke-width="1.1"/>')
    add(f'<text x="62" y="121" font-size="6.6" text-anchor="middle" fill="{BLUE_D}">'
        'operation agent</text>')
    add(f'<rect x="130" y="100" width="100" height="34" rx="5" fill="{YEL}" '
        f'stroke="{YEL_D}" stroke-width="1.1"/>')
    add(f'<text x="180" y="121" font-size="6.6" text-anchor="middle" fill="{YEL_D}">'
        'machine agent</text>')
    add(f'<rect x="254" y="100" width="88" height="34" rx="5" fill="{GRN}" '
        f'stroke="{GRN_D}" stroke-width="1.1"/>')
    add(f'<text x="298" y="121" font-size="6.6" text-anchor="middle" fill="{GRN_D}">'
        'AGV agent</text>')
    add(f'<path d="M108 117 H126" stroke="{FLOW}" stroke-width="1" marker-end="url(#mB)"/>')
    add(f'<path d="M232 117 H250" stroke="{FLOW}" stroke-width="1" marker-end="url(#mB)"/>')

    add(f'<text x="62" y="146" font-size="5.4" text-anchor="middle" fill="{BLUE_D}">'
        'picks the operation</text>')
    add(f'<text x="180" y="146" font-size="5.4" text-anchor="middle" fill="{YEL_D}">'
        'picks the machine</text>')
    add(f'<text x="298" y="146" font-size="5.4" text-anchor="middle" fill="{GRN_D}">'
        'picks the AGV</text>')
    add(f'<rect x="18" y="158" width="324" height="34" rx="6" fill="#F4F6F9"/>')
    add(f'<text x="26" y="179" font-size="6.6" font-weight="700" fill="#4A4A4A">objective</text>')
    for label, cx, w in (("makespan", 108, 58), ("PEC", 172, 34),
                         ("idle", 212, 34), ("transport", 270, 58)):
        add(f'<rect x="{cx - w // 2}" y="166" width="{w}" height="18" rx="9" '
            f'fill="{PUR}" stroke="{PUR_D}" stroke-width="0.9"/>')
        add(f'<text x="{cx}" y="178" font-size="5.8" text-anchor="middle" '
            f'fill="#7A3E71">{label}</text>')
    add('<text x="18" y="210" font-size="5.6" fill="#777777">'
        'the weights set the trade-off; the transport term counts loaded travel</text>')
    add('<text x="18" y="220" font-size="5.6" fill="#777777">plus half-weighted empty travel</text>')

    # ===================== (b) environment ===============================
    add(f'<text x="378" y="42" font-size="8.6" font-weight="700" fill="#1A1A1A">'
        '(b) Shop environment</text>')
    add(f'<rect x="378" y="52" width="138" height="196" rx="6" fill="#F5F9FD" '
        f'stroke="#BBD3E8" stroke-width="1" stroke-dasharray="5 4"/>')
    add(f'<rect x="560" y="52" width="138" height="196" rx="6" fill="#F5F9FD" '
        f'stroke="#BBD3E8" stroke-width="1" stroke-dasharray="5 4"/>')
    add(f'<text x="447" y="68" font-size="7" font-weight="700" text-anchor="middle" '
        f'fill="#2C5F8A">workshop 1</text>')
    add(f'<text x="629" y="68" font-size="7" font-weight="700" text-anchor="middle" '
        f'fill="#2C5F8A">workshop 2</text>')

    for i in range(3):
        add(machine(384 + i * 32, 78))
        add(machine(598 + i * 32, 78))
    add(station(486, 134))
    add(station(566, 134))
    add(f'<text x="500" y="162" font-size="5.2" text-anchor="middle" fill="#7A4E6D">L/U</text>')
    add(f'<text x="580" y="162" font-size="5.2" text-anchor="middle" fill="#7A4E6D">L/U</text>')

    for i in range(3):
        add(agv(382 + i * 36, 184))
        add(agv(594 + i * 36, 184))
        for x0 in (382, 596):
            pass

    for base in (382, 596):
        for i in range(3):
            col = "#B64342" if i == 1 else "#2E7D32"
            add(f'<circle cx="{base + i * 36 + 3}" cy="184" r="2.4" fill="{col}" '
                f'stroke="#FFFFFF" stroke-width="0.45"/>')

    add(f'<path d="M516 140 H562" stroke="{PUR_D}" stroke-width="1" '
        f'stroke-dasharray="4 3" marker-end="url(#mR)"/>')
    add(f'<text x="538" y="132" font-size="5.2" text-anchor="middle" fill="{PUR_D}">'
        'a job crosses</text>')
    add(f'<text x="538" y="214" font-size="5.6" text-anchor="middle" fill="#777777">'
        '3 machines and 3 AGVs per workshop</text>')
    add(f'<circle cx="470" cy="230" r="2.4" fill="#2E7D32" stroke="#FFFFFF" stroke-width="0.45"/>')
    add(f'<text x="477" y="233" font-size="5.2" fill="#555555">free AGV</text>')
    add(f'<circle cx="542" cy="230" r="2.4" fill="#B64342" stroke="#FFFFFF" stroke-width="0.45"/>')
    add(f'<text x="549" y="233" font-size="5.2" fill="#555555">busy AGV</text>')

    # ===================== (c) cross-workshop =============================
    add(f'<text x="18" y="298" font-size="8.6" font-weight="700" fill="#1A1A1A">'
        '(c) Cross-workshop transport</text>')
    for x0, label in ((18, "workshop 1"), (190, "workshop 2")):
        add(f'<rect x="{x0}" y="308" width="148" height="184" rx="6" fill="#F5F9FD" '
            f'stroke="#BBD3E8" stroke-width="1" stroke-dasharray="5 4"/>')
        add(f'<text x="{x0 + 74}" y="324" font-size="6.6" font-weight="700" '
            f'text-anchor="middle" fill="#2C5F8A">{label}</text>')

    add(machine(26, 344))
    add(station(134, 346))
    add(agv(26, 440))
    add(station(194, 346))
    add(machine(300, 344))
    add(agv(300, 440))
    add(f'<text x="148" y="366" font-size="5" text-anchor="middle" fill="#7A4E6D">L/U</text>')
    add(f'<text x="208" y="366" font-size="5" text-anchor="middle" fill="#7A4E6D">L/U</text>')

    add(f'<path d="M42 436 V378" stroke="{GRN_D}" stroke-width="1" stroke-dasharray="4 3" '
        f'marker-end="url(#mG)"/>')
    add(f'<path d="M57 356 H130" stroke="{FLOW}" stroke-width="1" marker-end="url(#mB)"/>')
    add(f'<path d="M166 353 H192" stroke="{ORG_D}" stroke-width="1" stroke-dasharray="4 3" '
        f'marker-end="url(#mO)"/>')
    add(f'<path d="M226 353 H298" stroke="{FLOW}" stroke-width="1" marker-end="url(#mB)"/>')
    add(f'<path d="M315 378 V436" stroke="{GRN_D}" stroke-width="1" stroke-dasharray="4 3" '
        f'marker-end="url(#mG)"/>')

    add(step_no(28, 406, 1, GRN_D))
    add(step_no(94, 348, 2, FLOW))
    add(step_no(179, 340, 3, ORG_D))
    add(step_no(262, 348, 4, FLOW))
    add(step_no(322, 406, 5, GRN_D))

    add(f'<text x="24" y="498" font-size="5.8" font-weight="700" fill="#2C5F8A">'
        'one trip, as the AGV spends it</text>')
    segs = ((44, "empty", GRN_D, "4 3"), (58, "loaded", FLOW, ""),
            (30, "transfer", ORG_D, "4 3"), (58, "loaded", FLOW, ""),
            (44, "empty", GRN_D, "4 3"))
    bx = 24
    for k, (w, name, col, dash) in enumerate(segs, start=1):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        add(f'<rect x="{bx}" y="512" width="{w}" height="16" fill="#FFFFFF" '
            f'stroke="{col}" stroke-width="1"{d}/>')
        add(f'<text x="{bx + w / 2}" y="523" font-size="5.2" text-anchor="middle" '
            f'fill="{col}">{name}</text>')
        add(step_no(bx + w / 2, 504, k, col))
        bx += w
    add(f'<path d="M24 536 H258" stroke="#999999" stroke-width="0.8" '
        f'marker-end="url(#mB)"/>')
    add(f'<text x="262" y="539" font-size="5.2" fill="#777777">time</text>')
    add(f'<text x="24" y="558" font-size="5.4" fill="#777777">'
        'the two loaded legs belong to different fleets; the transfer uses no AGV</text>')

    # ===================== (d) same workshop ==============================
    add(f'<text x="378" y="298" font-size="8.6" font-weight="700" fill="#1A1A1A">'
        '(d) Same-workshop transport</text>')
    add(f'<rect x="378" y="308" width="324" height="184" rx="6" fill="#F5F9FD" '
        f'stroke="#BBD3E8" stroke-width="1" stroke-dasharray="5 4"/>')
    add(f'<text x="540" y="324" font-size="6.6" font-weight="700" text-anchor="middle" '
        f'fill="#2C5F8A">one workshop</text>')

    add(machine(410, 352))
    add(machine(628, 352))
    add(agv(506, 430))

    add(f'<path d="M520 426 L438 384" stroke="{GRN_D}" stroke-width="1" '
        f'stroke-dasharray="4 3" marker-end="url(#mG)"/>')
    add(f'<path d="M438 384 H626" stroke="{FLOW}" stroke-width="1.2" '
        f'marker-end="url(#mB)"/>')
    add(step_no(486, 416, 1, GRN_D))
    add(step_no(556, 374, 2, FLOW))

    add(f'<text x="384" y="498" font-size="5.8" font-weight="700" fill="#2C5F8A">'
        'one trip, as the AGV spends it</text>')
    segs = ((56, "empty", GRN_D, "4 3"), (150, "loaded", FLOW, ""))
    bx = 384
    for k, (w, name, col, dash) in enumerate(segs, start=1):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        add(f'<rect x="{bx}" y="512" width="{w}" height="16" fill="#FFFFFF" '
            f'stroke="{col}" stroke-width="1"{d}/>')
        add(f'<text x="{bx + w / 2}" y="523" font-size="5.2" text-anchor="middle" '
            f'fill="{col}">{name}</text>')
        add(step_no(bx + w / 2, 504, k, col))
        bx += w
    add(f'<path d="M384 536 H576" stroke="#999999" stroke-width="0.8" '
        f'marker-end="url(#mB)"/>')
    add(f'<text x="580" y="539" font-size="5.2" fill="#777777">time</text>')
    add(f'<text x="384" y="558" font-size="5.4" fill="#777777">'
        'one fleet, no station hop, so the trip has no transfer segment</text>')

    add('</svg>')
    return "\n".join(s)


if __name__ == "__main__":
    (OUT / "fig_environment.svg").write_text(build(), encoding="utf-8")
    print("wrote fig_environment.svg")