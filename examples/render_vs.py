"""Render ``runs/vs_recording.json`` (from ``record_vs.py``) into a split-screen GIF.

Left: the base model prompted for JSON, its tokens appearing at the recorded timestamps.
Right: the same weights after any2jev, all answers appearing at the recorded latency.
Timers show real elapsed milliseconds; playback is slowed by ``--slow`` so the eye can follow.

    python examples/render_vs.py --recording runs/vs_recording.json --out docs/vs.gif --slow 4
"""

from __future__ import annotations

import argparse
import json
import textwrap

from PIL import Image, ImageDraw, ImageFont

BG, PANEL, BORDER = "#0b0e14", "#12161f", "#262c3a"
TXT, DIM, MUTED = "#dde3ee", "#8b93a7", "#5a6275"
ORANGE, GREEN, RED, BLUE, YELLOW, PURPLE = "#ff8a4c", "#3ddc84", "#ff5c5c", "#5b8def", "#f2c14e", "#b58cff"
BAR_COLORS = [GREEN, BLUE, YELLOW, PURPLE, ORANGE, RED]


def font(size: int, bold: bool = False):
    names = (["consolab.ttf", "DejaVuSansMono-Bold.ttf", "Menlo-Bold.ttf"] if bold
             else ["consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE, F_SUB, F_BODY, F_TIMER, F_SMALL, F_BIG = font(17, True), font(13), font(15), font(30, True), font(12), font(20, True)


def invalid_reason(left: dict, questions: dict) -> str:
    obj = left.get("parsed")
    if obj is None:
        return "not a JSON object"
    for qid, q in questions.items():
        if qid not in obj:
            return f'missing key "{qid}"'
        v = obj[qid]
        if q["type"] == "choice" and v not in q["criteria"]:
            return f'{qid} = "{v}" is not an allowed option'
        if q["type"] == "noul" and not isinstance(v, bool):
            return f"{qid} = {v!r} is not a boolean"
        if q["type"] == "score" and not (isinstance(v, int) and not isinstance(v, bool) and 0 <= v < len(q["criteria"])):
            return f"{qid} = {v!r} is not a level 0..{len(q['criteria']) - 1}"
    return ""


def draw_panel(d, x, y, w, h, title, subtitle):
    d.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=PANEL, outline=BORDER)
    d.text((x + 16, y + 12), title, fill=TXT, font=F_TITLE)
    d.text((x + 16, y + 36), subtitle, fill=DIM, font=F_SUB)


def draw_timer(d, x_right, y, ms, color):
    s = f"{ms:,.0f} ms"
    w = d.textlength(s, font=F_TIMER)
    d.text((x_right - w, y), s, fill=color, font=F_TIMER)


def draw_bar(d, x, y, w, h, probs: dict, chosen: str | None):
    """Stacked probability bar with labels beneath."""
    cx = x
    for i, p in enumerate(probs.values()):
        seg = max(2, int(w * p))
        col = BAR_COLORS[i % len(BAR_COLORS)]
        d.rounded_rectangle([cx, y, cx + seg, y + h], radius=3, fill=col)
        cx += seg + 1
    lx, ly = x, y + h + 6
    for i, (k, p) in enumerate(probs.items()):
        col = BAR_COLORS[i % len(BAR_COLORS)]
        d.rectangle([lx, ly + 3, lx + 8, ly + 11], fill=col)
        label = f"{k} {p:.2f}"
        f = F_BODY if k == chosen else F_SMALL
        d.text((lx + 13, ly - (2 if k == chosen else 0)), label, fill=TXT if k == chosen else DIM, font=f)
        lx += 13 + d.textlength(label, font=f) + 16


def render(rec: dict, out: str, slow: float, fps: int, hold_ms: int):
    left, right, questions = rec["left"], rec["right"], rec["questions"]
    W, H = 1080, 500
    top, gutter, pw = 92, 24, (1080 - 3 * 24) // 2
    py, ph = top, H - top - 44
    lx, rx = gutter, gutter * 2 + pw
    t_end = left["total_ms"] + hold_ms
    frames = []
    step = 1000 / fps / slow  # real ms per frame
    t = 0.0
    char_w = F_BODY.getlength("M")
    cols = int((pw - 32) / char_w)
    reason = invalid_reason(left, questions)
    first_tok = left["first_token_ms"] or 0
    while t <= t_end:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        # header: the shared request
        d.text((gutter, 14), "same weights, same GPU, same request", fill=DIM, font=F_SUB)
        state = rec["state"]
        d.text((gutter, 34), "state:  " + (state if len(state) < 118 else state[:115] + "..."), fill=TXT, font=F_SUB)
        qs = " · ".join(f"{qid} ({q['type']})" for qid, q in questions.items())
        d.text((gutter, 54), f"questions:  {qs}", fill=TXT, font=F_SUB)
        # left panel
        draw_panel(d, lx, py, pw, ph, "LLM prompted for JSON", f"{left['model']} · transformers.generate() · greedy · {left['dtype'].replace('torch.', '')}")
        lt = min(t, left["total_ms"])
        done_l = t >= left["total_ms"]
        draw_timer(d, lx + pw - 16, py + 10, lt, ORANGE if done_l else TXT)
        text = "".join(p for ts, p in left["pieces"] if ts <= t)
        body_y = py + 70
        if t < first_tok:
            d.text((lx + 16, body_y), "| waiting for the first token...", fill=MUTED, font=F_BODY)
        else:
            lines = textwrap.wrap(text, cols) or [""]
            for i, line in enumerate(lines[:9]):
                d.text((lx + 16, body_y + i * 22), line, fill=TXT, font=F_BODY)
            if not done_l:
                cx = lx + 16 + F_BODY.getlength(lines[-1]) + 2
                d.rectangle([cx, body_y + (len(lines) - 1) * 22 + 2, cx + 9, body_y + (len(lines) - 1) * 22 + 19], fill=ORANGE)
        if done_l:
            sy = py + ph - 78
            d.text((lx + 16, sy), f"{left['output_tokens']} tokens decoded one by one · first token at {first_tok:.0f} ms", fill=DIM, font=F_SUB)
            if left["format_ok"]:
                d.text((lx + 16, sy + 24), "OK  valid JSON, but no probabilities and no confidence", fill=YELLOW, font=F_BODY)
            else:
                d.text((lx + 16, sy + 24), "X   invalid: " + reason, fill=RED, font=F_BODY)
                d.text((lx + 16, sy + 48), "    a text model can always emit a value your code never allowed", fill=DIM, font=F_SUB)
        # right panel
        draw_panel(d, rx, py, pw, ph, "any2jev · one forward pass", f"{right['model']} + LoRA + pointer head · {right['dtype'].replace('torch.', '')} · no generate()")
        rt = min(t, right["latency_ms"])
        done_r = t >= right["latency_ms"]
        draw_timer(d, rx + pw - 16, py + 10, rt, GREEN if done_r else TXT)
        if not done_r:
            d.text((rx + 16, body_y), "| encoding state + 3 questions in one pass...", fill=MUTED, font=F_BODY)
        else:
            yy = body_y
            for qid, ans in right["answers"].items():
                if ans["type"] == "noul":
                    val, probs, chosen = f"p(yes) = {ans['noul']:.2f}", {"no": 1 - ans["noul"], "yes": ans["noul"]}, "yes" if ans["noul"] >= 0.5 else "no"
                elif ans["type"] == "choice":
                    val, probs, chosen = f"{ans['choice']}   conf {ans['confidence']:.2f}", ans["probabilities"], ans["choice"]
                else:
                    legend = ans["legend"]
                    probs = {legend[k]: v for k, v in ans["probabilities"].items()}
                    chosen = legend[max(ans["probabilities"], key=ans["probabilities"].get)]
                    val = f"{ans['score']:.2f}   conf {ans['confidence']:.2f}"
                d.text((rx + 16, yy), qid, fill=DIM, font=F_SUB)
                d.text((rx + 16 + 120, yy - 3), val, fill=TXT, font=F_BIG)
                draw_bar(d, rx + 16, yy + 26, pw - 32, 10, probs, chosen)
                yy += 78
            sy = py + ph - 54
            d.text((rx + 16, sy), f"OK  3 typed answers · probabilities sum to 1 · {right['input_tokens']} input tokens", fill=GREEN, font=F_BODY)
            d.text((rx + 16, sy + 24), f"answered before the LLM's first token ({right['latency_ms']:.0f} ms < {first_tok:.0f} ms)" if right["latency_ms"] < first_tok
                   else f"{left['total_ms'] / right['latency_ms']:.0f}× faster than generating the JSON", fill=DIM, font=F_SUB)
        # footer
        foot = f"{rec['gpu']} · real timings replayed at 1/{slow:g} speed · any2jev"
        d.text((gutter, H - 30), foot, fill=MUTED, font=F_SUB)
        frames.append(img)
        t += step
    frames += [frames[-1]] * int(fps * 1.2)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"wrote {out}: {len(frames)} frames, {W}x{H}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="runs/vs_recording.json")
    ap.add_argument("--out", default="docs/vs.gif")
    ap.add_argument("--slow", type=float, default=4.0, help="playback slowdown factor")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--hold-ms", type=int, default=900, help="real-time ms to keep counting after the LLM finishes")
    a = ap.parse_args()
    rec = json.load(open(a.recording, encoding="utf-8"))
    render(rec, a.out, a.slow, a.fps, a.hold_ms)


if __name__ == "__main__":
    main()
