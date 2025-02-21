"""
Any live cell with fewer than two live neighbours dies, as if by underpopulation.
Any live cell with two or three live neighbours lives on to the next generation.
Any live cell with more than three live neighbours dies, as if by overpopulation.
Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
"""

import argparse
from typing import NamedTuple, Set

import cv2
import numpy as np

from postgrad.video_writer import VideoWriter


class Cell(NamedTuple("Cell", [("row", int), ("column", int)])):
    def neighbors(self):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                yield Cell(self.row + dr, self.column + dc)

    def step(self, live_cells: Set["Cell"]) -> bool:
        count = sum(n in live_cells for n in self.neighbors())
        if count == 3:
            return True

        return count == 2 and self in live_cells


class GameOfLife:
    def __init__(self, live_cells: Set[Cell]):
        self.live_cells = live_cells

    def step(self):
        new_live_cells = set()
        for cell in self.live_cells:
            if cell.step(self.live_cells):
                new_live_cells.add(cell)
            for n in cell.neighbors():
                if n.step(self.live_cells):
                    new_live_cells.add(n)

        self.live_cells = new_live_cells

    def draw(self, image: np.ndarray):
        rows, columns = image.shape
        image[:, :] = 255
        for cell in self.live_cells:
            if 0 <= cell.row < rows and 0 <= cell.column < columns:
                image[cell.row, cell.column] = 0


Pulsar = """
..###...###..
.............
#....#.#....#
#....#.#....#
#....#.#....#
..###...###..
.............
..###...###..
#....#.#....#
#....#.#....#
#....#.#....#
.............
..###...###..
"""


GosperGliderGun = """
........................#...........
......................#.#...........
............##......##............##
...........#...#....##............##
##........#.....#...##..............
##........#...#.##....#.#...........
..........#.....#.......#...........
...........#...#....................
............##......................
"""


Acorn = """
..#.....
....#...
.##..###
"""


def parse_pattern(pattern: str, center: Cell) -> Set[Cell]:
    cells = set()
    rows = pattern.strip().split("\n")
    dr, dc = center
    dr -= len(rows) // 2
    dc -= len(rows[0]) // 2
    for r, row in enumerate(pattern.strip().split("\n")):
        for c, cell in enumerate(row):
            if cell == "#":
                cells.add(Cell(r + dr, c + dc))

    return cells


def main(rows: int, columns: int, steps: int, video_path: str, framerate: float):
    image = np.zeros((3*rows + 2, columns), dtype=np.uint8)
    image[rows, :] = 120
    image[2*rows + 1, :] = 120
    center = Cell(rows // 2, columns // 2)
    pulsar = GameOfLife(parse_pattern(Pulsar, center))
    glider_gun = GameOfLife(parse_pattern(GosperGliderGun, center))
    acorn = GameOfLife(parse_pattern(Acorn, center))
    scale = 8
    frame = np.zeros((image.shape[0]*scale, image.shape[1]*scale), dtype=np.uint8)
    kron = np.ones((scale, scale))

    if video_path is not None:
        video = VideoWriter(video_path, (frame.shape[1], frame.shape[0]), framerate=framerate, quality=17)
        video.text_color = (0, 0, 0)
        video.start()
    else:
        video = None

    for _ in range(steps):
        pulsar.draw(image[:rows])
        glider_gun.draw(image[rows + 1:2*rows + 1])
        acorn.draw(image[-rows:])
        pulsar.step()
        glider_gun.step()
        acorn.step()

        frame[:] = np.kron(image, kron)
        for r in range(0, rows):
            frame[r * scale, :] = 240
            frame[(2 * r + 1) * scale, :] = 240
            frame[-r * scale, :] = 240

        for c in range(0, frame.shape[1], scale):
            r1 = rows * scale
            frame[:r1, c] = 240
            r0 = r1 + scale
            r1 = r0 + (rows * scale)
            frame[r0:r1, c] = 240
            r0 = r1 + scale
            frame[r0:, c] = 240

        cv2.imshow("Game of Life", frame)
        cv2.waitKey(1)

        if video is not None:
            video.frame[:] = frame[:, :, np.newaxis]
            video.write_frame()

    if video is not None:
        video.stop()

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser("Conway's Game of Life")
    parser.add_argument("--rows", "-r", type=int, default=30)
    parser.add_argument("--columns", "-c", type=int, default=80)
    parser.add_argument("--steps", "-s", type=int, default=1200)
    parser.add_argument("--video-path", "-v", type=str, default=None)
    parser.add_argument("--framerate", "-f", type=float, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.rows, args.columns, args.steps, args.video_path, args.framerate)
