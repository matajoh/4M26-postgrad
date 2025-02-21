"""Custom 2D graphics functions using NumPy arrays.

NB: Nothing in this module is in scope for the Tripos. Students do not
need to understand how this works, but may find it interesting.
"""

import json
import os
from typing import Mapping, NamedTuple, Tuple

import numpy as np
from PIL import Image

from .geometry import Rect2, Vec2


def asset_path(*args) -> str:
    return os.path.join(os.path.dirname(__file__), "assets", *args)


def blend(*images) -> np.ndarray:
    result = images[0].astype(np.uint16)
    for image in images[1:]:
        alpha1 = image[:, :, 3:].astype(np.uint16)
        alpha0 = 255 - alpha1
        result[:, :, :3] = (alpha1 * image[:, :, :3] +
                            alpha0 * result[:, :, :3]) >> 8
        result[:, :, 3:] = np.minimum(255, image[:, :, 3:] + result[:, :, 3:])

    return result.astype(np.uint8)


def blend_into_safe(image: np.ndarray, pos: Vec2, pixels: np.ndarray):
    h, w = pixels.shape[:2]
    c0, r0 = pos - Vec2(w // 2, h // 2)
    c1, r1 = c0 + w, r0 + h
    src_c0 = src_r0 = 0
    src_c1 = w
    src_r1 = h
    rows, columns = image.shape[:2]
    if c0 < 0:
        src_c0 = -c0
        c0 = 0
    if c1 > columns:
        src_c1 -= c1 - columns
        c1 = columns
    if r0 < 0:
        src_r0 = -r0
        r0 = 0
    if r1 > rows:
        src_r1 -= r1 - rows
        r1 = rows
    image[r0:r1, c0:c1] = blend(image[r0:r1, c0:c1], pixels[src_r0:src_r1, src_c0:src_c1])


def blend_into(image: np.ndarray, pos: Vec2, pixels: np.ndarray):
    h, w = pixels.shape[:2]
    c0, r0 = pos - Vec2(w // 2, h // 2)
    c1, r1 = c0 + w, r0 + h
    image[r0:r1, c0:c1] = blend(image[r0:r1, c0:c1], pixels)


def erase(image: np.ndarray, pos: Vec2, size: int):
    c0, r0 = pos - Vec2(size // 2, size // 2)
    c1, r1 = c0 + size, r0 + size
    image[r0:r1, c0:c1] = 0


class TileInfo(NamedTuple("TileInfo", [("row", int), ("column", int)])):
    @staticmethod
    def create(info: dict) -> "TileInfo":
        return TileInfo(info["row"], info["column"])


TileNames = set(["wall", "floor", "faculty_room", "faculty_room_door"])


def read_rgba(*name: str) -> np.ndarray:
    path = asset_path(*name)
    image = Image.open(path).convert("RGBA")
    return np.array(image)


class Tilesheet(NamedTuple("Tilesheet", [("pixels", np.ndarray),
                                         ("tile_size", int),
                                         ("rows", int),
                                         ("columns", int),
                                         ("tiles", Mapping[Tuple[str, str], TileInfo])])):
    @staticmethod
    def load(info_path: str = None) -> "Tilesheet":
        if info_path is None:
            info_path = asset_path("assets.json")
        else:
            assert os.path.exists(info_path) and info_path.endswith(".json")

        with open(info_path, "r") as f:
            info = json.load(f)

        image_path = os.path.join(
            os.path.dirname(info_path), info["image_path"])
        image = Image.open(image_path).convert("RGBA")
        pixels = np.array(image)
        tiles = {(i["name"], i["type"]): TileInfo.create(i) for i in info["tiles"]
                 if i["name"] in TileNames}
        return Tilesheet(pixels, info["tile_size"], info["rows"], info["columns"], tiles)

    def read_tile(self, name: str, type: str) -> np.ndarray:
        tile = self.tiles[(name, type)]
        r = tile.row * self.tile_size
        c = tile.column * self.tile_size
        return self.pixels[r:r + self.tile_size, c:c + self.tile_size]


class SpriteInfo(NamedTuple("SpriteInfo", [("row", int), ("column", int), ("bb", Rect2)])):
    @staticmethod
    def create(info: dict) -> "SpriteInfo":
        bb = info["bb"]
        left = bb["x"]
        top = bb["y"]
        right = left + bb["width"]
        bottom = top + bb["height"]
        return SpriteInfo(info["row"], info["column"], Rect2(left, top, right, bottom))


class SpriteSheet(NamedTuple("SpriteSheet", [("pixels", np.ndarray),
                                             ("item_size", int),
                                             ("sprites", Mapping[Tuple[str, str], SpriteInfo])])):
    @staticmethod
    def load(info_path: str = None) -> "SpriteSheet":
        if info_path is None:
            info_path = asset_path("assets.json")
        else:
            assert os.path.exists(info_path) and info_path.endswith(".json")

        with open(info_path, "r") as f:
            info = json.load(f)

        image_path = os.path.join(
            os.path.dirname(info_path), info["image_path"])
        image = Image.open(image_path).convert("RGBA")
        pixels = np.array(image)
        sprites = {(i["name"], i["type"]): SpriteInfo.create(i) for i in info["tiles"]
                   if i["name"] not in TileNames}

        return SpriteSheet(pixels, info["tile_size"], sprites)

    def read_sprite(self, name: str, type: str, no_crop=False, no_alpha=False) -> np.ndarray:
        sprite = self.sprites[(name, type)]
        r = sprite.row * self.item_size
        c = sprite.column * self.item_size
        image = self.pixels[r:r + self.item_size, c:c + self.item_size]
        if no_alpha:
            image = image[..., :3]

        if no_crop:
            return image

        top = sprite.bb.top
        left = sprite.bb.left
        right = sprite.bb.right + 1
        bottom = sprite.bb.bottom + 1
        return image[top:bottom, left:right]


class GlyphInfo(NamedTuple("GlyphInfo", [("row", int), ("column", int)])):
    @staticmethod
    def create(info: dict) -> "GlyphInfo":
        return GlyphInfo(info["row"], info["column"])


class GlyphSheet(NamedTuple("GlyphSheet", [("pixels", np.ndarray), ("size", int),
                                           ("glyphs", Mapping[str, GlyphInfo])])):
    @staticmethod
    def load(info_path: str = None) -> "GlyphSheet":
        if info_path is None:
            info_path = asset_path("glyphs.json")
        else:
            assert os.path.exists(info_path) and info_path.endswith(".json")

        with open(info_path, "r") as f:
            info = json.load(f)

        image_path = os.path.join(
            os.path.dirname(info_path), info["image_path"])
        image = Image.open(image_path).convert("RGBA")
        pixels = np.array(image)
        glyphs = {i["type"]: GlyphInfo.create(i) for i in info["tiles"]}
        return GlyphSheet(pixels, info["tile_size"], glyphs)

    def draw(self, text: str, color=None, spacing=1) -> "np.ndarray":
        w = (self.size + spacing) * len(text) - spacing
        image = np.zeros((self.size, w, 4), dtype=np.uint8)
        left = 0
        for _, c in enumerate(text):
            glyph = self.glyphs[c]
            r = glyph.row * self.size
            c = glyph.column * self.size
            right = left + self.size
            pixels = self.pixels[r:r + self.size, c:c + self.size]
            if color:
                pixels[:, :, :3] = color

            image[:, left:right] = pixels
            left = right + spacing

        return image

    def blend(self, image: np.ndarray, pos: Vec2,
              text: str, color=None, spacing=1, scaling=1) -> None:
        pixels = self.draw(text, color, spacing)
        if scaling > 1:
            scale = np.ones((scaling, scaling, 1))
            pixels = np.kron(pixels, scale)

        h, w = pixels.shape[:2]
        c, r = pos - Vec2(w // 2, h // 2)
        image[r:r + h, c:c + w] = blend(image[r:r + h, c:c + w], pixels)
