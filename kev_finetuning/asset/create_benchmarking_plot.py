#!/usr/bin/env python3
"""Create the Banking77 macro-F1 benchmarking graphic."""
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
SVG = OUT / "benchmarking.svg"
PNG = OUT / "benchmarking.png"
W, H = 1080, 1350


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, value, size, fill, weight=400, anchor="start", extra=""):
    return (f'<text x="{x}" y="{y}" font-family="Inter, Arial, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" {extra}>{esc(value)}</text>')


def rect(x, y, width, height, fill, rx=0, extra=""):
    return (f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{rx}" fill="{fill}" {extra}/>')


print("Building Banking77 benchmarking graphic…", flush=True)
parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <linearGradient id="accent" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ec9bd4"/><stop offset="1" stop-color="#c92ca2"/></linearGradient>
  <linearGradient id="tint" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fbecf7"/><stop offset="1" stop-color="#fff8fd"/></linearGradient>
</defs>
{rect(0, 0, W, H, "#f8f7f4")}
{rect(0, 0, W, 16, "url(#accent)")}
''']
parts += [
    rect(72, 68, 10, 10, "#c92ca2", 5),
    text(96, 80, "BANKING77  ·  FINE-TUNING EXPERIMENT", 16, "#706872", 700, extra='letter-spacing="1.1"'),
    text(72, 182, "Fine-tuning Kev-4B", 54, "#17151a", 750),
    text(72, 244, "raised Macro F1 by", 54, "#17151a", 750),
    text(72, 322, "+13.66 points", 72, "#c92ca2", 800),
    text(76, 370, "vs. the Jev reference score", 25, "#5e5961", 450),
    rect(72, 426, 936, 174, "url(#tint)", 24, 'stroke="#efd5e8" stroke-width="2"'),
    text(108, 473, "BEST KEV CHECKPOINT  ·  EPOCH 3", 16, "#8d3477", 750, extra='letter-spacing="1.2"'),
    text(108, 555, "93.36%", 62, "#211a20", 800),
    text(366, 548, "Macro F1", 23, "#5e5961", 600),
    text(972, 490, "FULL TEST SPLIT", 14, "#827985", 700, "end", 'letter-spacing="1.1"'),
    text(972, 519, "3,080 examples", 19, "#39343b", 600, "end"),
    text(972, 548, "77 intents", 19, "#39343b", 600, "end"),
    text(72, 660, "SCORE COMPARISON", 16, "#77717a", 750, extra='letter-spacing="1.5"'),
    text(1008, 660, "Macro F1 (%)", 16, "#77717a", 550, "end"),
]
rows = [
    ("Jev", "TypeSafe AI · reference", 79.70, "#adb4c1"),
    ("Kev-4B", "Before fine-tuning", 84.95, "#29272b"),
    ("Epoch 1", "Kev-4B · fine-tuned", 91.81, "#e8a4d4"),
    ("Epoch 2", "Kev-4B · fine-tuned", 93.08, "#d968b8"),
    ("Epoch 3", "Kev-4B · fine-tuned", 93.36, "url(#accent)"),
]
start_y, track_x, track_w = 728, 72, 810
# Keep the reference scale attached to the bars as one small axis.
axis_y = 703
for tick, label, anchor in [(0, "0", "start"), (50, "50", "middle"), (100, "100", "end")]:
    x = track_x + track_w * tick / 100
    parts.append(text(x, axis_y, label, 13, "#8b858d", 500, anchor))
    parts.append(f'<line x1="{x}" y1="710" x2="{x}" y2="1247" stroke="#dedbd8" stroke-width="1" stroke-dasharray="3 7"/>')
for idx, (name, detail, score, color) in enumerate(rows):
    y = start_y + idx * 111
    if idx == 4:
        parts.append(rect(56, y - 30, 968, 98, "#fff", 16, 'stroke="#efd5e8" stroke-width="1.5"'))
    parts.extend([
        text(88, y, name, 23, "#211e23", 700),
        text(88, y + 27, detail, 15, "#77717a", 450),
        rect(track_x, y + 43, track_w, 15, "#e9e7e5", 7),
        rect(track_x, y + 43, round(track_w * score / 100), 15, color, 7),
        text(988, y + 55, f"{score:.2f}%", 23, "#211e23", 750, "end"),
    ])
parts += [
    rect(72, 1300, 936, 1, "#dedbd8"),
    text(72, 1327, "Credits: TypeSafe AI (Jev reference) · Jared Palmer (Kev-4B) · Banking77 dataset", 13, "#716c73", 450),
    "</svg>"
]
SVG.write_text("\n".join(parts), encoding="utf-8")
print(f"Wrote editable SVG: {SVG.name}", flush=True)
print("Rendering 1080 × 1350 PNG…", flush=True)
subprocess.run(["rsvg-convert", "-w", str(W), "-h", str(H), "-o", str(PNG), str(SVG)], check=True)
print(f"PNG ready: {PNG.name}", flush=True)
