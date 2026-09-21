"""Snake, driven by an any2jev checkpoint: one Choice question per tick, no text generation.

    python examples/snake.py --generate 3000 --out data/snake/train.jsonl   # BFS teacher -> labelled requests
    any2jev train --base Qwen/Qwen3-0.6B --data data/snake/train.jsonl --val data/snake/val.jsonl --out runs/snake
    python examples/snake.py runs/snake                                       # play in the terminal
    python examples/snake.py runs/snake --gif snake.gif                       # also record a GIF (needs pillow)

The game loop owns the rules (walls, self-collision, legal moves); the model only picks between the
legal moves given a compact JSON state, the same split TypeSafe's own demos use.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
INSTRUCTIONS = "Which direction should the snake move next to reach the food without hitting a wall or itself?"


class Snake:
    def __init__(self, w=12, h=10, seed=0):
        self.w, self.h, self.rng = w, h, random.Random(seed)
        self.body = [(w // 2, h // 2), (w // 2 - 1, h // 2)]
        self.dir = "right"
        self.food = self._spawn()
        self.score, self.steps, self.alive = 0, 0, True

    def _spawn(self):
        free = [(x, y) for x in range(self.w) for y in range(self.h) if (x, y) not in self.body]
        return self.rng.choice(free)

    def legal(self):
        out = {}
        hx, hy = self.body[0]
        for d, (dx, dy) in DIRS.items():
            if d == OPPOSITE[self.dir]:
                continue
            nx, ny = hx + dx, hy + dy
            if 0 <= nx < self.w and 0 <= ny < self.h and (nx, ny) not in self.body[:-1]:
                out[d] = (nx, ny)
        return out

    def state(self):
        hx, hy = self.body[0]
        fx, fy = self.food
        return {
            "board": {"width": self.w, "height": self.h},
            "head": {"x": hx, "y": hy},
            "food": {"x": fx, "y": fy},
            "food_direction": {"horizontal": "right" if fx > hx else "left" if fx < hx else "same column",
                               "vertical": "down" if fy > hy else "up" if fy < hy else "same row"},
            "current_direction": self.dir,
            "length": len(self.body),
            "danger": {d: "blocked" for d in DIRS if d not in self.legal() and d != OPPOSITE[self.dir]},
        }

    def step(self, d):
        self.steps += 1
        moves = self.legal()
        if d not in moves:
            self.alive = False
            return
        self.dir = d
        nxt = moves[d]
        self.body.insert(0, nxt)
        if nxt == self.food:
            self.score += 1
            self.food = self._spawn()
        else:
            self.body.pop()

    def render(self):
        grid = [["." for _ in range(self.w)] for _ in range(self.h)]
        for x, y in self.body[1:]:
            grid[y][x] = "o"
        hx, hy = self.body[0]
        grid[hy][hx] = "O"
        fx, fy = self.food
        grid[fy][fx] = "*"
        return "\n".join("".join(r) for r in grid)

    def bfs_move(self):
        """Teacher policy: first step of the shortest safe path to the food (None if unreachable)."""
        blocked = set(self.body[:-1])
        start = self.body[0]
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == self.food:
                break
            for d, (dx, dy) in DIRS.items():
                nxt = (cur[0] + dx, cur[1] + dy)
                if 0 <= nxt[0] < self.w and 0 <= nxt[1] < self.h and nxt not in blocked and nxt not in prev:
                    prev[nxt] = (cur, d)
                    q.append(nxt)
        if self.food not in prev:
            return None
        node, move = self.food, None
        while prev[node] is not None:
            node, move = prev[node]
        return move if move in self.legal() else None


def request_for(game: Snake, legal: dict) -> dict:
    return {"state": game.state(), "questions": {"move": {
        "type": "choice", "instructions": INSTRUCTIONS, "criteria": {d: f"Move {d}" for d in legal}}}}


def generate(n_records: int, out: str, seed: int = 0):
    """Play games with the BFS teacher and write labelled requests (option order shuffled)."""
    from pathlib import Path

    rng = random.Random(seed)
    rows, game_seed = [], seed
    while len(rows) < n_records:
        game = Snake(w=rng.choice([8, 10, 12]), h=rng.choice([8, 10]), seed=game_seed)
        game_seed += 1
        while game.alive and game.steps < 300 and len(rows) < n_records:
            legal = game.legal()
            move = game.bfs_move()
            if not legal or move is None:
                break
            if len(legal) >= 2:
                order = list(legal)
                rng.shuffle(order)
                req = request_for(game, {d: legal[d] for d in order})
                req["questions"]["move"]["label"] = move
                rows.append(req)
            # explore a little so states are varied, but keep the label from the teacher
            game.step(move if rng.random() < 0.85 else rng.choice(list(legal)))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} labelled moves to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", nargs="?")
    ap.add_argument("--generate", type=int, default=0, help="write N labelled moves from the BFS teacher and exit")
    ap.add_argument("--out", default="data/snake/train.jsonl")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.08)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--dtype", default=None)
    a = ap.parse_args()

    if a.generate:
        generate(a.generate, a.out, a.seed)
        return
    if not a.model_dir:
        ap.error("model_dir is required unless --generate is given")

    from any2jev.model import DecisionModel

    model = DecisionModel.load(a.model_dir, dtype=a.dtype)
    game = Snake(seed=a.seed)
    frames, latencies = [], []
    while game.alive and game.steps < a.steps:
        legal = game.legal()
        if not legal:
            break
        req = request_for(game, legal)
        t0 = time.perf_counter()
        answers, _ = model.decide(req)
        latencies.append((time.perf_counter() - t0) * 1000)
        a_move = answers["move"]
        game.step(a_move["choice"])
        probs = " ".join(f"{k}:{v:.2f}" for k, v in a_move["probabilities"].items())
        frame = f"{game.render()}\nscore {game.score}  step {game.steps}  {latencies[-1]:.0f} ms   {probs}"
        frames.append(frame)
        sys.stdout.write("\x1b[2J\x1b[H" + frame + "\n")
        sys.stdout.flush()
        time.sleep(a.delay)
    print(f"\nfinal score {game.score} in {game.steps} steps | median latency {sorted(latencies)[len(latencies)//2]:.0f} ms")
    if a.gif:
        _save_gif(frames, a.gif)
        print(f"wrote {a.gif}")


def _save_gif(frames, path):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    imgs = []
    for f in frames:
        lines = f.split("\n")
        img = Image.new("RGB", (max(len(line) for line in lines) * 7 + 20, len(lines) * 13 + 20), "black")
        d = ImageDraw.Draw(img)
        for i, line in enumerate(lines):
            d.text((10, 10 + 13 * i), line, fill="white", font=font)
        imgs.append(img)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=120, loop=0)


if __name__ == "__main__":
    main()
