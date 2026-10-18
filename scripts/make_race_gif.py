"""Render figures/race.gif and figures/race-dark.gif from benchmark solves.

Every solver plays its solution of the same scramble at the same pace (one face turn
per two frames), so shorter solutions finish first.  The scrambles are the first of
the ``short-3x3`` and ``random-2x2`` sets; nothing is picked by hand.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rubiks_cube_solvers.cube import FACES, MOVE_NAMES, cube2, from_facelets, parse, to_facelets
from rubiks_cube_solvers.render import draw_cube, sticker_map

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
THEMES = {
    "light": {
        "bg": (252, 252, 251),
        "ink": (11, 11, 11),
        "ink2": (82, 81, 78),
        "muted": (137, 135, 129),
        "rule": (225, 224, 217),
    },
    "dark": {
        "bg": (26, 26, 25),
        "ink": (240, 239, 236),
        "ink2": (195, 194, 183),
        "muted": (137, 135, 129),
        "rule": (44, 44, 42),
    },
}
WIDTH = 1200
MARGIN = 16
CUBE = 236
FRAME_MS = 70
HOLD_START = 14
HOLD_END = 40


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def load():
    with open(ROOT / "results" / "solves.csv", newline="") as f:
        solves = list(csv.DictReader(f))
    with open(ROOT / "results" / "scrambles.csv", newline="") as f:
        scrambles = {(r["set"], r["id"]): r for r in csv.DictReader(f)}

    def solution(set_name, solver, config=None):
        for r in solves:
            if r["set"] == set_name and r["solver"] == solver and r["id"] == "0":
                if config is None or r["config"] == config:
                    return r
        raise KeyError((set_name, solver))

    short = scrambles[("short-3x3", "0")]
    two = scrambles[("random-2x2", "0")]
    rows3 = [
        ("Beginner's method", solution("short-3x3", "beginner")),
        ("Thistlethwaite", solution("short-3x3", "thistlethwaite")),
        ("Kociemba two-phase", solution("short-3x3", "kociemba", "target=20")),
        ("Korf IDA* (optimal)", solution("short-3x3", "korf")),
    ]
    rows2 = [
        ("Korf IDA* (optimal)", solution("random-2x2", "korf-2x2")),
        ("Learned cost-to-go + A*", solution("random-2x2", "learned-2x2")),
    ]
    start3 = from_facelets(short["state"])
    start2 = cube2.state_cube(int(two["state"]))
    return start3, short["scramble"], rows3, start2, rows2


class Panel:
    def __init__(self, title, row, start, n, x, y):
        self.title = title
        self.moves = parse(row["solution"])
        self.seconds = float(row["cpu_seconds"])
        self.n = n
        self.x, self.y = x, y
        self.states = [to_facelets(start)]
        cube = start
        for m in self.moves:
            cube = cube.apply([m])
            self.states.append(to_facelets(cube))
        self._cache = {}

    def image(self, step: int, half: bool, bg) -> tuple[Image.Image, int, str | None]:
        """Cube image after ``step`` moves (mid-turn of the next move if ``half``)."""
        done = min(step, len(self.moves))
        turn = None
        current = self.moves[done - 1] if done > 0 else None
        if half and step < len(self.moves):
            m = self.moves[step]
            power = m % 3 + 1
            angle = {1: np.pi / 4, 2: np.pi / 2, 3: -np.pi / 4}[power]
            turn = (FACES[m // 3], angle)
            current = m
            done = step
        key = (done, turn)
        if key not in self._cache:
            stickers = sticker_map(self.states[done], self.n)
            self._cache[key] = draw_cube(stickers, CUBE, n=self.n, turn=turn, background=bg)
        name = MOVE_NAMES[current] if current is not None else None
        return self._cache[key], done, name


def fmt_seconds(s: float) -> str:
    if s < 1e-3:
        return f"{s * 1e6:.0f} µs"
    if s < 1:
        return f"{s * 1e3:.0f} ms"
    return f"{s:.1f} s"


def render(theme_name: str, out: Path) -> None:
    theme = THEMES[theme_name]
    start3, scramble3, rows3, start2, rows2 = load()
    col = (WIDTH - 2 * MARGIN) // 4
    y1 = 58
    panel_h = 30 + CUBE + 56
    y2 = y1 + panel_h + 52
    height = y2 + panel_h + 8
    panels = [Panel(t, r, start3, 3, MARGIN + i * col, y1) for i, (t, r) in enumerate(rows3)]
    panels += [Panel(t, r, start2, 2, MARGIN + (i + 1) * col, y2) for i, (t, r) in enumerate(rows2)]
    longest = max(len(p.moves) for p in panels)

    f_title, f_name = font(19, True), font(15, True)
    f_count, f_move, f_small = font(13), font(22, True), font(12)

    rgb_frames = []
    schedule = [(0, False)] * HOLD_START
    for step in range(longest):
        schedule += [(step, True), (step + 1, False)]
    schedule += [(longest, False)] * HOLD_END
    for step, half in schedule:
        img = Image.new("RGB", (WIDTH, height), theme["bg"])
        d = ImageDraw.Draw(img)
        d.text(
            (MARGIN, 14),
            "Same scramble, every solver, one face turn per tick",
            font=f_title,
            fill=theme["ink"],
        )
        d.text(
            (MARGIN, y1 - 6),
            f"3x3x3, 14-move scramble:  {scramble3}",
            font=f_small,
            fill=theme["muted"],
            anchor="lb",
        )
        d.line([(MARGIN, y2 - 26), (WIDTH - MARGIN, y2 - 26)], fill=theme["rule"], width=1)
        d.text(
            (MARGIN, y2 - 6),
            "2x2x2, random state: optimal search against the learned heuristic",
            font=f_small,
            fill=theme["muted"],
            anchor="lb",
        )
        for p in panels:
            cube_img, done, name = p.image(step, half, theme["bg"])
            cx = p.x + (col - CUBE) // 2
            d.text((p.x + col // 2, p.y + 14), p.title, font=f_name, fill=theme["ink"], anchor="mm")
            img.paste(cube_img, (cx, p.y + 26))
            base = p.y + 26 + CUBE + 8
            finished = done == len(p.moves) and not (half and step < len(p.moves))
            if finished and step > 0:
                d.text(
                    (p.x + col // 2, base + 10),
                    f"solved: {len(p.moves)} moves",
                    font=f_name,
                    fill=theme["ink"],
                    anchor="mm",
                )
                d.text(
                    (p.x + col // 2, base + 32),
                    f"CPU time {fmt_seconds(p.seconds)}",
                    font=f_count,
                    fill=theme["ink2"],
                    anchor="mm",
                )
            else:
                if name is None:
                    d.text(
                        (p.x + col // 2, base + 10),
                        "scrambled",
                        font=f_count,
                        fill=theme["muted"],
                        anchor="mm",
                    )
                else:
                    d.text((p.x + col // 2, base + 10), name, font=f_move, fill=theme["ink"], anchor="mm")
                d.text(
                    (p.x + col // 2, base + 32),
                    f"move {done + (1 if half and step < len(p.moves) else 0)} of {len(p.moves)}",
                    font=f_count,
                    fill=theme["ink2"],
                    anchor="mm",
                )
        rgb_frames.append(img)
    # One palette for every frame, computed from a strip of sample frames (scrambled,
    # mid-turn and solved) so text greys and sticker colours all get their own entries.
    samples = [rgb_frames[i] for i in (0, HOLD_START, HOLD_START + 1, len(rgb_frames) // 2, -1)]
    strip = Image.new("RGB", (WIDTH, height * len(samples)))
    for k, sample in enumerate(samples):
        strip.paste(sample, (0, k * height))
    palette = strip.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in rgb_frames]
    frames[0].save(
        out, save_all=True, append_images=frames[1:], duration=FRAME_MS, loop=0, optimize=True, disposal=1
    )
    print(f"wrote {out.relative_to(ROOT)}: {len(frames)} frames, {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    render("light", ROOT / "figures" / "race.gif")
    render("dark", ROOT / "figures" / "race-dark.gif")
