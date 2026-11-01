"""Convert the hand-written SVG figures into editable draw.io files.

Shapes, texts, circles and connectors become separate mxCells so they can be
dragged in the VS Code draw.io editor.  Curves are straightened to their
endpoints; text is placed as its own cell with an estimated box.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).resolve().parent
NS = "{http://www.w3.org/2000/svg}"

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_path(d: str) -> list[tuple[float, float]]:
    """Return the polyline of a path, ignoring curve control points."""
    tokens = re.findall(r"[MLHVCmlhvc]|-?\d+(?:\.\d+)?", d)
    pts: list[tuple[float, float]] = []
    cx = cy = 0.0
    cmd = "M"
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.isalpha():
            cmd = t
            i += 1
            continue
        rel = cmd.islower()
        c = cmd.upper()
        if c in ("M", "L"):
            x = float(tokens[i]); y = float(tokens[i + 1]); i += 2
            if rel:
                x, y = cx + x, cy + y
            pts.append((x, y)); cx, cy = x, y
        elif c == "H":
            x = float(tokens[i]); i += 1
            x = cx + x if rel else x
            pts.append((x, cy)); cx = x
        elif c == "V":
            y = float(tokens[i]); i += 1
            y = cy + y if rel else y
            pts.append((cx, y)); cy = y
        elif c in ("C", "S"):
            n = 6 if c == "C" else 4
            vals = [float(v) for v in tokens[i:i + n]]; i += n
            x, y = vals[-2], vals[-1]
            if rel:
                x, y = cx + x, cy + y
            pts.append((x, y)); cx, cy = x, y
        elif c == "Q":
            vals = [float(v) for v in tokens[i:i + 4]]; i += 4
            x, y = vals[-2], vals[-1]
            if rel:
                x, y = cx + x, cy + y
            pts.append((x, y)); cx, cy = x, y
        else:
            i += 1
    return pts


def col(el: ET.Element, key: str, default: str) -> str:
    v = el.get(key)
    if v is None:
        style = el.get("style") or ""
        m = re.search(rf"{key}\s*:\s*([^;]+)", style)
        v = m.group(1).strip() if m else default
    return v


def convert(src: Path, dst: Path) -> None:
    root = ET.parse(src).getroot()
    w = root.get("viewBox", "0 0 720 580").split()
    page_w, page_h = float(w[2]), float(w[3])

    cells: list[str] = []
    n = 0

    def nid() -> str:
        nonlocal n
        n += 1
        return f"c{n}"

    for el in root.iter():
        tag = el.tag.replace(NS, "")
        if tag == "rect":
            x, y = float(el.get("x", 0)), float(el.get("y", 0))
            rw, rh = float(el.get("width", 0)), float(el.get("height", 0))
            fill = el.get("fill", "#FFFFFF")
            stroke = el.get("stroke", "none")
            dash = el.get("stroke-dasharray")
            rx = float(el.get("rx", 0) or 0)
            style = (f"rounded={1 if rx else 0};whiteSpace=wrap;html=1;"
                     f"fillColor={fill};strokeColor={stroke};strokeWidth=1;")
            if rx:
                style += f"arcSize={int(min(50, rx / max(rw, rh) * 100))};"
            if dash:
                style += "dashed=1;"
            cells.append(
                f'<mxCell id="{nid()}" parent="1" style="{style}" value="" vertex="1">'
                f'<mxGeometry height="{rh}" width="{rw}" x="{x}" y="{y}" as="geometry"/>'
                f'</mxCell>')
        elif tag == "circle":
            cx, cy = float(el.get("cx", 0)), float(el.get("cy", 0))
            r = float(el.get("r", 0))
            fill = el.get("fill", "#FFFFFF")
            stroke = el.get("stroke", "none")
            style = (f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};"
                     f"strokeColor={stroke};strokeWidth=1;")
            cells.append(
                f'<mxCell id="{nid()}" parent="1" style="{style}" value="" vertex="1">'
                f'<mxGeometry height="{2 * r}" width="{2 * r}" x="{cx - r}" y="{cy - r}" '
                f'as="geometry"/></mxCell>')
        elif tag == "text":
            x, y = float(el.get("x", 0)), float(el.get("y", 0))
            size = float(el.get("font-size", 7))
            anchor = el.get("text-anchor", "start")
            fill = el.get("fill", "#1A1A1A")
            weight = el.get("font-weight", "normal")
            txt = (el.text or "").strip()
            if not txt:
                continue
            tw = max(size * 0.56 * len(txt), size * 2)
            th = size * 1.35
            left = x if anchor == "start" else (x - tw / 2 if anchor == "middle" else x - tw)
            bold = ";fontStyle=1" if str(weight) == "700" or weight == "bold" else ""
            style = (f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                     f"align={'center' if anchor == 'middle' else 'left'};"
                     f"verticalAlign=middle;fontSize={size};fontFamily=Helvetica;"
                     f"fontColor={fill}{bold};")
            cells.append(
                f'<mxCell id="{nid()}" parent="1" style="{style}" value="{txt}" vertex="1">'
                f'<mxGeometry height="{th:.1f}" width="{tw:.1f}" x="{left:.1f}" '
                f'y="{y - th * 0.72:.1f}" as="geometry"/></mxCell>')
        elif tag in ("path", "line"):
            if tag == "line":
                pts = [(float(el.get("x1", 0)), float(el.get("y1", 0))),
                       (float(el.get("x2", 0)), float(el.get("y2", 0)))]
                marker = el.get("marker-end")
                stroke = el.get("stroke", "#333333")
                dash = el.get("stroke-dasharray")
            else:
                d = el.get("d", "")
                if not d:
                    continue
                pts = parse_path(d)
                marker = el.get("marker-end")
                stroke = el.get("stroke", "#333333")
                dash = el.get("stroke-dasharray")
            if len(pts) < 2:
                continue
            arrow = "endArrow=block;endSize=5;" if marker else "endArrow=none;"
            dash_s = "dashed=1;" if dash else ""
            style = f"html=1;rounded=0;{arrow}strokeColor={stroke};strokeWidth=1;{dash_s}"
            way = "".join(f'<mxPoint x="{p[0]:.1f}" y="{p[1]:.1f}"/>' for p in pts[1:-1])
            arr = f'<Array as="points">{way}</Array>' if way else ""
            cells.append(
                f'<mxCell id="{nid()}" edge="1" parent="1" style="{style}" value="">'
                f'<mxGeometry relative="1" as="geometry"><mxPoint x="{pts[0][0]:.1f}" '
                f'y="{pts[0][1]:.1f}" as="sourcePoint"/><mxPoint x="{pts[-1][0]:.1f}" '
                f'y="{pts[-1][1]:.1f}" as="targetPoint"/>{arr}</mxGeometry></mxCell>')

    body = "\n                ".join(cells)
    xml = ('<mxfile host="65bd71144e">\n    <diagram id="conv" name="Page-1">\n'
           f'        <mxGraphModel dx="900" dy="700" grid="1" gridSize="10" guides="1" '
           f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
           f'pageWidth="{page_w:.0f}" pageHeight="{page_h:.0f}" math="0" shadow="0">\n'
           '            <root>\n                <mxCell id="0"/>\n'
           '                <mxCell id="1" parent="0"/>\n'
           f'                {body}\n            </root>\n'
           '        </mxGraphModel>\n    </diagram>\n</mxfile>\n')
    dst.write_text(xml, encoding="utf-8")
    print(f"{dst.name}: {len(cells)} cells")


if __name__ == "__main__":
    for stem in ("fig_environment", "fig_method"):
        convert(OUT / f"{stem}.svg", OUT / f"{stem}.drawio")