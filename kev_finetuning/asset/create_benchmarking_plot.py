#!/usr/bin/env python3
"""Create the Banking77 macro-F1 benchmarking graphic."""
from pathlib import Path
import argparse
import subprocess

OUT = Path(__file__).resolve().parent
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


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--summary", action="store_true", help="create a simplified three-model comparison")
args = parser.parse_args()
stem = "benchmarking_summary" if args.summary else "benchmarking"
SVG = OUT / f"{stem}.svg"
PNG = OUT / f"{stem}.png"
print(f"Building {stem.replace('_', ' ')} graphic…", flush=True)
parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <linearGradient id="accent" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ec9bd4"/><stop offset="1" stop-color="#c92ca2"/></linearGradient>
  <linearGradient id="tint" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fbecf7"/><stop offset="1" stop-color="#fff8fd"/></linearGradient>
</defs>
{rect(0, 0, W, H, "#f8f7f4")}
{rect(0, 0, W, 16, "url(#accent)")}
''']
parts += [
    text(72, 182, "Fine-tuning Kev-4B", 54, "#17151a", 750),
    text(72, 244, "raised Macro F1 by", 54, "#17151a", 750),
    text(72, 322, "+13.66 points", 72, "#c92ca2", 800),
    text(76, 370, "vs. the Jev reference score", 25, "#5e5961", 450),
    rect(72, 426, 936, 174, "url(#tint)", 24, 'stroke="#efd5e8" stroke-width="2"'),
    text(108, 473, "FINE-TUNED MODEL" if args.summary else "BEST KEV CHECKPOINT  ·  EPOCH 3", 16, "#8d3477", 750, extra='letter-spacing="1.2"'),
    text(108, 555, "93.36%", 62, "#211a20", 800),
    text(366, 548, "Macro F1", 23, "#5e5961", 600),
    text(972, 490, "FULL TEST SPLIT", 14, "#827985", 700, "end", 'letter-spacing="1.1"'),
    text(972, 519, "3,080 examples", 19, "#39343b", 600, "end"),
    text(972, 548, "77 intents", 19, "#39343b", 600, "end"),
    text(72, 660, "SCORE COMPARISON", 16, "#77717a", 750, extra='letter-spacing="1.5"'),
    text(1008, 660, "Macro F1 (%)", 16, "#77717a", 550, "end"),
]
if args.summary:
    parts.append(text(72, 706, "Banking77 customer queries classified into 77 banking intents", 19, "#5e5961", 450))
all_rows = [
    ("Jev", "TypeSafe AI · reference", 79.70, "#adb4c1"),
    ("Kev-4B", "Before fine-tuning", 84.95, "#29272b"),
    ("Epoch 1", "Kev-4B · fine-tuned", 91.81, "#e8a4d4"),
    ("Epoch 2", "Kev-4B · fine-tuned", 93.08, "#d968b8"),
    ("Epoch 3", "Kev-4B · fine-tuned", 93.36, "url(#accent)"),
]
rows = [all_rows[i] for i in (0, 1, 4)] if args.summary else all_rows
start_y, track_x, track_w = (790, 72, 810) if args.summary else (728, 72, 810)
row_gap = 150 if args.summary else 111
# Values sit beside each bar, so omit axis ticks and grid lines to keep the
# comparison clean and avoid detached scale labels in the portrait layout.
for idx, (name, detail, score, color) in enumerate(rows):
    y = start_y + idx * row_gap
    is_final = idx == (len(rows) - 1)
    if is_final:
        parts.append(rect(56, y - 30, 968, 98, "#fff", 16, 'stroke="#efd5e8" stroke-width="1.5"'))
    if args.summary and idx == 2:
        name, detail = "Fine-tuned model", "Kev-4B · fine-tuned"
    parts.extend([
        text(88, y, name, 23, "#211e23", 700),
        text(88, y + 27, detail, 15, "#77717a", 450),
        rect(track_x, y + 43, track_w, 15, "#e9e7e5", 7),
        rect(track_x, y + 43, round(track_w * score / 100), 15, color, 7),
        text(988, y + 55, f"{score:.2f}%", 23, "#211e23", 750, "end"),
    ])
parts += ["</svg>"]
SVG.write_text("\n".join(parts), encoding="utf-8")
print(f"Wrote editable SVG: {SVG.name}", flush=True)
print("Rendering 1080 × 1350 PNG…", flush=True)
subprocess.run(["rsvg-convert", "-w", str(W), "-h", str(H), "-o", str(PNG), str(SVG)], check=True)
print(f"PNG ready: {PNG.name}", flush=True)
