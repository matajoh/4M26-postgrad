"""Module providing game screens.

NB out of scope for Tripos.
"""

import json
from typing import List, NamedTuple, Tuple

import numpy as np

from .drawing import GlyphSheet, Tilesheet, asset_path, blend, read_rgba
from .geometry import Rect2
from .sprite import Action


class SelectInfo(NamedTuple("SelectInfo", [("name", str), ("color", Tuple[int, int, int]),
                                           ("box", Rect2)])):
    @staticmethod
    def from_dict(data: dict) -> "SelectInfo":
        return SelectInfo(data["name"], tuple(data["color"]), Rect2(**data["bb"]))


Fade = NamedTuple("Fade", [("selected", int), ("ticks", int)])

FADE_TICKS = 30
BLINK_ON_TICKS = 50
BLINK_OFF_TICKS = 10


class SelectScreen(NamedTuple("SelectScreen", [("pixels", np.ndarray),
                                               ("gray_pixels", np.ndarray),
                                               ("background", np.ndarray),
                                               ("info", List[SelectInfo]),
                                               ("selected", int),
                                               ("ticks", int),
                                               ("fade", Fade),
                                               ("ready", bool)])):
    def with_selected(self, selected: int) -> "SelectScreen":
        return self._replace(selected=selected % len(self.info))

    def with_ready(self, ready: bool) -> "SelectScreen":
        return self._replace(ready=ready)

    def create(tilesheet: Tilesheet, frame_size: Tuple[int, int],
               selected: int) -> "SelectScreen":
        info_path = asset_path("select_screen.json")
        with open(info_path, "r") as file:
            data = json.load(file)

        pixels = read_rgba(data["image_path"])
        info = [SelectInfo.from_dict(i) for i in data["select_info"]]

        background = np.zeros((*frame_size, 4), dtype=np.uint8)
        tilesize = tilesheet.tile_size
        background[:tilesize, :tilesize] = tilesheet.read_tile("wall", "n/w/e/s/nw/ne/sw")
        background[:tilesize, -tilesize:] = tilesheet.read_tile("wall", "n/w/e/s/nw/ne/se")
        background[-tilesize:, :tilesize] = tilesheet.read_tile("wall", "n/w/e/s/nw/sw/se")
        background[-tilesize:, -tilesize:] = tilesheet.read_tile("wall", "n/w/e/s/ne/sw/se")
        rows, columns = frame_size
        for c in range(tilesize, columns - tilesize, tilesize):
            background[:tilesize, c:c + tilesize] = tilesheet.read_tile("wall", "n/w/e/nw/ne")
            background[-tilesize:, c:c + tilesize] = tilesheet.read_tile("wall", "w/e/s/sw/se")

        for r in range(tilesize, rows-tilesize, tilesize):
            background[r:r + tilesize, :tilesize] = tilesheet.read_tile("wall", "n/w/s/nw/sw")
            background[r:r + tilesize, -tilesize:] = tilesheet.read_tile("wall", "n/e/s/ne/se")

        for r in range(tilesize, rows - tilesize, tilesize):
            for c in range(tilesize, columns - tilesize, tilesize):
                background[r:r + tilesize, c:c + tilesize] = tilesheet.read_tile("floor", "0")

        gray = pixels[..., :3].astype(np.float32)
        gray *= [0.3, 0.59, 0.11]
        gray = gray.sum(axis=-1, keepdims=True)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        gray_pixels = pixels.copy()
        gray_pixels[..., :3] = gray

        return SelectScreen(pixels, gray_pixels, background, info, selected, 0, None, False)

    def draw(self, frame: np.ndarray):
        pixels = self.gray_pixels.copy()

        if self.fade is not None:
            rect = self.info[self.fade.selected].box
            c0, r0, c1, r1 = rect
            fade = self.pixels[r0:r1, c0:c1].copy()
            alpha = fade[..., 3].astype(np.int32)
            alpha = alpha * self.fade.ticks // FADE_TICKS
            fade[..., 3] = alpha.astype(np.uint8)
            pixels[r0:r1, c0:c1] = blend(pixels[r0:r1, c0:c1], fade)

        color = self.info[self.selected].color + (255,)
        rect = self.info[self.selected].box
        c0, r0, c1, r1 = rect

        if self.ticks < BLINK_ON_TICKS:
            pixels[r0-2:r1+2, c0:c1] = color
            pixels[r0:r1, c0-2:c1+2] = color
            pixels[r0-1:r1+1, c0-1:c1+1] = color

        pixels[r0:r1, c0:c1] = self.pixels[r0:r1, c0:c1]
        image = self.background.copy()
        h, w = pixels.shape[:2]
        rows, columns = frame.shape[:2]
        r0 = (rows - h) // 2
        c0 = (columns - w) // 2
        r1 = r0 + h
        c1 = c0 + w
        image[r0:r1, c0:c1] = blend(image[r0:r1, c0:c1], pixels)
        frame[:] = image[..., :3]

    def step(self, action: Action) -> "SelectScreen":
        if action == Action.MOVE_UP:
            return self._replace(ready=True)

        selected = self.selected
        fade = self.fade
        if action == Action.MOVE_LEFT:
            selected = max(self.selected - 1, 0)
        elif action == Action.MOVE_RIGHT:
            selected = min(self.selected + 1, len(self.info) - 1)

        if selected != self.selected:
            ticks = 0
            fade = Fade(self.selected, FADE_TICKS)
        elif fade:
            ticks = (fade.ticks - 1) % FADE_TICKS
            if ticks == 0:
                fade = None
            else:
                fade = fade._replace(ticks=ticks)

        ticks = (self.ticks + 1) % (BLINK_ON_TICKS + BLINK_OFF_TICKS)
        return self._replace(ticks=ticks, selected=selected, fade=fade)


class InfoScreen(NamedTuple("InfoScreen", [("pixels", np.ndarray)])):
    @staticmethod
    def create(tilesheet: Tilesheet, frame_size: Tuple[int, int]) -> "InfoScreen":
        image_path = asset_path("faculty_info.png")
        pixels = read_rgba(image_path)

        image = np.zeros((*frame_size, 4), dtype=np.uint8)
        tilesize = tilesheet.tile_size
        rows, columns = frame_size
        image[:tilesize, :tilesize] = tilesheet.read_tile("faculty_room", "e/s/se")
        image[:tilesize, -tilesize:] = tilesheet.read_tile("faculty_room", "w/s/sw")
        image[-tilesize:, :tilesize] = tilesheet.read_tile("faculty_room", "n/e/ne")
        image[-tilesize:, -tilesize:] = tilesheet.read_tile("faculty_room", "n/w/nw")
        rows, columns = frame_size
        for c in range(tilesize, columns - tilesize, tilesize):
            image[:tilesize, c:c + tilesize] = tilesheet.read_tile("faculty_room", "w/e/s/sw/se")
            image[-tilesize:, c:c + tilesize] = tilesheet.read_tile("faculty_room", "n/w/e/nw/ne")

        for r in range(tilesize, rows-tilesize, tilesize):
            image[r:r + tilesize, :tilesize] = tilesheet.read_tile("faculty_room", "n/e/s/ne/se")
            image[r:r + tilesize, -tilesize:] = tilesheet.read_tile("faculty_room", "n/w/s/nw/sw")

        for r in range(tilesize, rows - tilesize, tilesize):
            for c in range(tilesize, columns - tilesize, tilesize):
                image[r:r + tilesize, c:c + tilesize] = tilesheet.read_tile("faculty_room", "n/w/e/s/nw/ne/sw/se")

        h, w = pixels.shape[:2]
        r0 = (rows - h) // 2
        c0 = (columns - w) // 2
        r1 = r0 + h
        c1 = c0 + w
        image[r0:r1, c0:c1] = blend(image[r0:r1, c0:c1], pixels)
        return InfoScreen(image)

    def draw(self, frame: np.ndarray):
        frame[:] = self.pixels[..., :3]


class GameOverScreen(NamedTuple("GameOverScreen", [("go_pixels", np.ndarray),
                                                   ("yn_pixels", np.ndarray),
                                                   ("spacing", int)])):
    @staticmethod
    def create(glyphsheet: GlyphSheet) -> "GameOverScreen":
        go_pixels = glyphsheet.draw("GAME OVER")[..., :3]
        scale = np.ones((4, 4, 1))
        go_pixels = np.kron(go_pixels, scale)

        yn_pixels = glyphsheet.draw("Continue? [y]/[n]")[..., :3]
        scale = np.ones((2, 2, 1))
        yn_pixels = np.kron(yn_pixels, scale)

        spacing = glyphsheet.size * 4

        return GameOverScreen(go_pixels, yn_pixels, spacing)

    def draw(self, frame: np.ndarray):
        rows, columns = frame.shape[:2]
        h0, w0 = self.go_pixels.shape[:2]
        h1, w1 = self.yn_pixels.shape[:2]
        r0 = (rows - h0 - h1 - self.spacing) // 2
        c0 = (columns - w0) // 2
        r1 = r0 + h0
        c1 = c0 + w0
        frame[r0:r1, c0:c1] = self.go_pixels
        r0 = r1 + self.spacing
        r1 = r0 + h1
        c0 = (columns - w1) // 2
        c1 = c0 + w1
        frame[r0:r1, c0:c1] = self.yn_pixels


class PauseScreen(NamedTuple("PauseScreen", [("pixels", np.ndarray)])):
    @staticmethod
    def create(glyphsheet: GlyphSheet) -> "PauseScreen":
        pixels = glyphsheet.draw("PAUSED")
        scale = np.ones((4, 4, 1))
        pixels = np.kron(pixels, scale)
        return PauseScreen(pixels)

    def draw(self, frame: np.ndarray):
        frame[..., :3] = frame[..., :3] // 2
        rows, columns = frame.shape[:2]
        h, w = self.pixels.shape[:2]
        r0 = (rows - h) // 2
        c0 = (columns - w) // 2
        r1 = r0 + h
        c1 = c0 + w
        alpha1 = self.pixels[..., 3:].astype(np.uint16)
        alpha0 = 255 - alpha1
        pixels0 = alpha0 * frame[r0:r1, c0:c1].astype(np.uint16)
        pixels1 = alpha1 * self.pixels[..., :3].astype(np.uint16)
        frame[r0:r1, c0:c1] = ((pixels0 + pixels1) >> 8).astype(np.uint8)
