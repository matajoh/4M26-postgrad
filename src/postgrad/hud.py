"""Module providing Heads Up Display (HUD) classes for the PostGrad game.

NB out of scope for Tripos.
"""

import os
from typing import NamedTuple

import numpy as np

from .drawing import GlyphSheet, SpriteSheet
from .geometry import Direction, Vec2
from .sprite import Sprite, SpriteKind


class Scoreboard(NamedTuple("Scoreboard", [("score", int),
                                           ("high_score", int),
                                           ("text_pixels", np.ndarray),
                                           ("score_pixels", np.ndarray),
                                           ("highscore_pixels", np.ndarray),
                                           ("glyphs", GlyphSheet)])):
    @staticmethod
    def create(score: int, high_score: int, glyphs: GlyphSheet) -> "Scoreboard":
        text_pixels = glyphs.draw("    1UP   HIGH SCORE")[..., :3]
        score_pixels = glyphs.draw(f"{score:02}")[..., :3]
        high_score_pixels = glyphs.draw(f"{high_score:02}")[..., :3]
        return Scoreboard(score, high_score, text_pixels, score_pixels, high_score_pixels, glyphs)

    @property
    def height(self) -> int:
        return 2 * self.glyphs.size

    @property
    def width(self) -> int:
        return 20 * self.glyphs.size

    def add_points(self, points: int) -> "Scoreboard":
        score = self.score + points
        pixels = self.glyphs.draw(f"{score:02}")[..., :3]
        if score < self.high_score:
            return self._replace(score=score, score_pixels=pixels)

        high_score = score
        highscore_pixels = pixels
        return self._replace(score=score, score_pixels=pixels,
                             high_score=high_score, highscore_pixels=highscore_pixels)

    def draw(self, image: np.ndarray, row: int, column: int):
        c = column
        r = row
        h, w = self.text_pixels.shape[:2]
        image[r:r + h, c:c + w] = self.text_pixels
        r += self.glyphs.size
        h, w = self.score_pixels.shape[:2]
        c = column + (8 * self.glyphs.size) - w
        image[r:r + h, c:c + w] = self.score_pixels
        h, w = self.highscore_pixels.shape[:2]
        c = column + (18 * self.glyphs.size) - w
        image[r:r + h, c:c + w] = self.highscore_pixels


class TriesBoard(NamedTuple("TriesBoard", [("count", int), ("max_count", int), ("pixels", np.ndarray)])):
    @staticmethod
    def create(icon: np.ndarray, count: int, max_count: int) -> "TriesBoard":
        h, w = icon.shape[:2]
        pixels = np.zeros((h, w * max_count, 3), dtype=np.uint8)
        for i in range(max_count):
            pixels[:, i * w:(i + 1) * w] = icon[..., :3]

        return TriesBoard(count, max_count, pixels)

    def with_icon(self, icon: np.ndarray) -> "TriesBoard":
        h, w = icon.shape[:2]
        pixels = np.zeros((h, w * self.max_count, 3), dtype=np.uint8)
        for i in range(self.max_count):
            pixels[:, i * w:(i + 1) * w] = icon[..., :3]

        return self._replace(pixels=pixels)

    @property
    def height(self) -> int:
        return self.pixels.shape[0]

    @property
    def width(self) -> int:
        return self.height * self.count

    def add_try(self) -> "TriesBoard":
        return self._replace(count=min(self.count + 1, self.max_count))

    def remove_try(self) -> "TriesBoard":
        return self._replace(count=max(self.count - 1, 0))

    def draw(self, image: np.ndarray, row: int, column: int):
        h = self.pixels.shape[0]
        w = h * self.count
        r = row
        c = column
        image[r:r + h, c:c + w] = self.pixels[:, :w]


class FruitBoard(NamedTuple("FruitBoard", [("pixels", np.ndarray), ("level", int)])):
    @staticmethod
    def create(level: int, spritesheet: SpriteSheet) -> "FruitBoard":
        pixels = np.concatenate(
            [spritesheet.read_sprite("food", name, True)
             for name in ["banana", "pear", "apple_red", "pretzel",
                          "orange", "strawberry", "cherries"]],
            axis=1)
        return FruitBoard(pixels, level)

    def next_level(self) -> "FruitBoard":
        return self._replace(level=self.level + 1)

    def draw(self, image: np.ndarray, row: int, column: int):
        h = self.pixels.shape[0]
        w = h * self.level
        r = row
        c = column - w
        image[r:r + h, c:c + w] = self.pixels[:, -w:, :3]

    def item(self) -> Sprite:
        item = SpriteKind.CHERRY.add(min(self.level, 7) - 1)
        match item:
            case SpriteKind.CHERRY:
                index = 6
            case SpriteKind.STRAWBERRY:
                index = 5
            case SpriteKind.ORANGE:
                index = 4
            case SpriteKind.PRETZEL:
                index = 3
            case SpriteKind.APPLE:
                index = 2
            case SpriteKind.PEAR:
                index = 1
            case SpriteKind.BANANA:
                index = 0
            case _:
                raise ValueError(f"not a fruit: {item}")

        h = self.pixels.shape[0]
        c0 = index * h
        c1 = c0 + h
        return Sprite.create(item, Vec2(-100, -100), h, Direction.RIGHT, self.pixels[:, c0:c1])


class HUD(NamedTuple("HUD", [("scoreboard", Scoreboard),
                             ("triesboard", TriesBoard),
                             ("fruitboard", FruitBoard),
                             ("extra_tries", int),
                             ("points_per_try", int)])):
    def add_try(self) -> "HUD":
        return self._replace(triesboard=self.triesboard.add_try())

    def remove_try(self) -> "HUD":
        return self._replace(triesboard=self.triesboard.remove_try())

    @property
    def tries(self) -> int:
        return self.triesboard.count

    def with_try_icon(self, icon: np.ndarray) -> "HUD":
        return self._replace(triesboard=self.triesboard.with_icon(icon))

    def add_points(self, points: int) -> "HUD":
        scoreboard = self.scoreboard.add_points(points)
        tries = scoreboard.score // self.points_per_try
        if tries <= self.extra_tries:
            return self._replace(scoreboard=scoreboard)

        triesboard = self.triesboard.add_try()
        extra_tries = tries
        return self._replace(scoreboard=scoreboard, triesboard=triesboard, extra_tries=extra_tries)

    def next_level(self) -> "HUD":
        return self._replace(fruitboard=self.fruitboard.next_level())

    def fruit(self) -> Sprite:
        return self.fruitboard.item()

    def level(self) -> int:
        return self.fruitboard.level

    def draw(self, frame: np.ndarray, image: np.ndarray):
        rows, columns = frame.shape[:2]
        h, w = image.shape[:2]
        c0 = (columns - w) // 2
        c1 = c0 + w
        r0 = (rows - h) // 2
        r1 = r0 + h
        frame[r0:r1, c0:c1] = image[..., :3]

        border = (rows - h) // 2

        h = self.scoreboard.height
        r = (border - h) // 2
        self.scoreboard.draw(frame, r, c0)

        h = self.triesboard.height
        r = r1 + (border - h) // 2
        self.triesboard.draw(frame, r, c0)

        h = self.fruitboard.pixels.shape[0]
        r = r1 + (border - h) // 2
        self.fruitboard.draw(frame, r, c1)

    def lose(self):
        high_score_path = os.path.expanduser("~/.postgrad_highscore")
        with open(high_score_path, "w") as f:
            f.write(str(self.scoreboard.high_score))

    @staticmethod
    def create(grad: int, glyphs: GlyphSheet, spritesheet: SpriteSheet,
               num_tries=3, max_tries=5, points_per_try=10000) -> "HUD":
        high_score_path = os.path.expanduser("~/.postgrad_highscore")
        if not os.path.exists(high_score_path):
            with open(high_score_path, "w") as f:
                f.write("0")

            high_score = 0
        else:
            with open(high_score_path, "r") as f:
                try:
                    high_score = int(f.read())
                except ValueError:
                    high_score = 0

        scoreboard = Scoreboard.create(0, high_score, glyphs)
        icon = spritesheet.read_sprite("grad", str(grad), True)
        tries = TriesBoard.create(icon, num_tries, max_tries)
        fruitboard = FruitBoard.create(1, spritesheet)
        return HUD(scoreboard, tries, fruitboard, 0, points_per_try)
