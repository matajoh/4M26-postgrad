"""Asset creation script.

This script takes the raw asset files and creates a single tilesheet
with associated information for use by the game. As such, there are some
classes which appear here which will have corresponding classes which
are almost identical in the game code. At first glance this duplication
of code may seem like a bad idea, but it allows the game code to change
its internal representations without having to keep the importing
code in sync. Thus, the import code can stay fairly simple and is
independent of the game code.

One way to create your own "skin" of the game is to change the raw
assets and re-run this script.
"""

from colorsys import hls_to_rgb, rgb_to_hls
from enum import IntFlag
import json
import os
import shutil
from typing import Any, List, Mapping, NamedTuple, Tuple

import numpy as np
from PIL import Image


WallGuide = NamedTuple(
    "WallGuide", [("row", int), ("column", int), ("guide", str)])


class WallType(IntFlag):
    Single = 0
    North = 1
    NorthEast = 2
    East = 4
    SouthEast = 8
    South = 16
    SouthWest = 32
    West = 64
    NorthWest = 128
    All = 255

    @staticmethod
    def from_guide(w: WallGuide) -> "WallType":
        val = WallType.Single
        if w.guide[0] == '.':
            val |= WallType.NorthWest
        if w.guide[1] == '.':
            val |= WallType.North
        if w.guide[2] == '.':
            val |= WallType.NorthEast
        if w.guide[3] == '.':
            val |= WallType.West
        if w.guide[5] == '.':
            val |= WallType.East
        if w.guide[6] == '.':
            val |= WallType.SouthWest
        if w.guide[7] == '.':
            val |= WallType.South
        if w.guide[8] == '.':
            val |= WallType.SouthEast

        return val

    @staticmethod
    def from_string(s: str) -> "WallType":
        val = WallType.Single
        dir_set = set(s.split("/"))
        if "nw" in dir_set:
            val |= WallType.NorthWest
        if "n" in dir_set:
            val |= WallType.North
        if "ne" in dir_set:
            val |= WallType.NorthEast
        if "w" in dir_set:
            val |= WallType.West
        if "e" in dir_set:
            val |= WallType.East
        if "sw" in dir_set:
            val |= WallType.SouthWest
        if "s" in dir_set:
            val |= WallType.South
        if "se" in dir_set:
            val |= WallType.SouthEast

        return val

    def __str__(self) -> str:
        dirs = []
        if self & WallType.North:
            dirs.append("n")
        if self & WallType.West:
            dirs.append("w")
        if self & WallType.East:
            dirs.append("e")
        if self & WallType.South:
            dirs.append("s")

        if self & WallType.NorthWest:
            dirs.append("nw")
        if self & WallType.NorthEast:
            dirs.append("ne")
        if self & WallType.SouthWest:
            dirs.append("sw")
        if self & WallType.SouthEast:
            dirs.append("se")

        return "/".join(dirs)


def read_rgba(path: str) -> np.ndarray:
    image = Image.open(path)
    image = image.convert("RGBA")
    pixels = np.array(image)
    return pixels


def crop_alpha(pixels: np.ndarray) -> np.ndarray:
    y, x = np.where(pixels[:, :, 3] > 0.5)
    r0 = y.min()
    r1 = y.max() + 1
    c0 = x.min()
    c1 = x.max() + 1
    rows = r1 - r0
    columns = c1 - c0
    if rows > columns:
        c0 = (c0 + c1 - rows) // 2
        c1 = c0 + rows
    else:
        r0 = (r0 + r1 - columns) // 2
        r1 = r0 + columns

    return pixels[r0:r1, c0:c1]


class Tilesheet(NamedTuple("Tilesheet", [("pixels", np.ndarray),
                                         ("tile_size", int),
                                         ("rows", int),
                                         ("columns", int),
                                         ("padding", int),
                                         ("info", List[dict])])):
    @staticmethod
    def from_image(path: str, tile_size: int, padding: int) -> "Tilesheet":
        pixels = read_rgba(path)
        rows = pixels.shape[0] // tile_size
        columns = pixels.shape[1] // tile_size
        return Tilesheet(pixels, tile_size, rows, columns, padding, [])

    @staticmethod
    def create(rows: int, columns: int, tile_size: int, padding=0) -> "Tilesheet":
        pixels = np.zeros((rows * tile_size, columns *
                          tile_size, 4), dtype=np.uint8)
        pixels[:, :] = 0
        return Tilesheet(pixels, tile_size, rows, columns, padding, [])

    def read_tile(self, row: int, column: int) -> np.ndarray:
        r = row * self.tile_size
        c = column * self.tile_size
        return self.pixels[r:r + self.tile_size - self.padding, c:c + self.tile_size - self.padding]

    def write_tile(self, row: int, column: int, pixels: np.ndarray):
        r = row * self.tile_size
        c = column * self.tile_size
        pixels = pixels[:self.tile_size, :self.tile_size]
        self.pixels[r:r + self.tile_size,
                    c:c + self.tile_size] = pixels

    def add_tile(self, pixels: np.ndarray, info: Any = None) -> int:
        index = len(self.info)
        if info is None:
            info = {"name": f"tile_{index}"}
        elif isinstance(info, str):
            info = {"name": info, "type": info}
        else:
            info = info.copy()

        row = index // self.columns
        column = index % self.columns
        info["row"] = row
        info["column"] = column
        self.info.append(info)
        self.write_tile(row, column, pixels)
        return index

    def save(self, info_path: str):
        assert info_path.endswith(".json")
        png_path = info_path.replace(".json", ".png")
        Image.fromarray(self.pixels).save(png_path)

        metadata = {
            "rows": self.rows,
            "columns": self.columns,
            "tile_size": self.tile_size,
            "tiles": self.info,
            "image_path": os.path.basename(png_path)
        }
        with open(info_path, "w") as f:
            json.dump(metadata, f, indent=4)


class Wall(NamedTuple("Wall", [("name", str), ("type", WallType), ("pixels", np.ndarray)])):
    @staticmethod
    def load_walls(tilesheet: Tilesheet,
                   row: int, column: int,
                   name: str,
                   guides: List[WallGuide]) -> List["Wall"]:
        walls = []
        for g in guides:
            pixels = tilesheet.read_tile(row + g.row, column + g.column)
            walls.append(Wall(name, WallType.from_guide(g), pixels))

        return walls

    @property
    def info(self) -> dict:
        return {
            "name": self.name,
            "type": str(self.type)
        }


def read_guides(path: str) -> List[WallGuide]:
    guides = []
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    rows = len(lines) // 3
    columns = len(lines[0]) // 3

    for r in range(rows):
        for c in range(columns):
            if c * 3 >= len(lines[r * 3]):
                continue

            guide = lines[r * 3][c * 3:c * 3 + 3] + \
                lines[r * 3 + 1][c * 3:c * 3 + 3] + \
                lines[r * 3 + 2][c * 3:c * 3 + 3]
            guides.append(WallGuide(r, c, guide))

    return guides


class Floor(NamedTuple("Floor", [("type", int), ("pixels", np.ndarray)])):
    @property
    def info(self) -> dict:
        return {
            "name": "floor",
            "type": str(self.type)
        }


FacultyTiles = ["e/s/se", "w/s/sw", "n/e/ne", "n/w/nw/",
                "w/e/s/sw/se", "n/w/e/nw/ne", "n/w/s/nw/sw", "n/e/s/ne/se",
                "n/w/e/s/nw/ne/se/sw"]


class Rectangle(NamedTuple("Rectangle", [("x", int), ("y", int), ("width", int), ("height", int)])):
    @staticmethod
    def from_alpha(pixels: np.ndarray) -> "Rectangle":
        y, x = np.where(pixels[:, :, 3] > 0.5)
        return Rectangle(int(x.min()),
                         int(y.min()),
                         int(x.max() - x.min()),
                         int(y.max() - y.min()))


class Sprite(NamedTuple("Sprite", [("name", str), ("type", str), ("bb", Rectangle), ("pixels", np.ndarray)])):
    @staticmethod
    def from_png(path: str, name: str, type: str) -> "Sprite":
        pixels = read_rgba(path)
        bb = Rectangle.from_alpha(pixels)
        return Sprite(name, type, bb, pixels)

    @staticmethod
    def from_tilesheet(ts: Tilesheet, row: int, column: int, name: str, type: str) -> "Sprite":
        pixels = ts.read_tile(row, column)
        bb = Rectangle.from_alpha(pixels)
        return Sprite(name, type, bb, pixels)

    @property
    def info(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "bb": self.bb._asdict(),
        }

    @staticmethod
    def load_food(path: str) -> List["Sprite"]:
        foods = []
        for file in os.listdir(path):
            if file.endswith(".png"):
                foods.append(Sprite.from_png(
                    os.path.join(path, file), "food", file[:-4]))

        return foods


class Glyphs(NamedTuple("Glyphs", [("pixels", np.ndarray),
                                   ("size", int),
                                   ("charmap", Mapping[str, Tuple[int, int]])])):
    @staticmethod
    def create(path: str, charmap: dict) -> "Glyphs":
        pixels = read_rgba(path)
        tile_size = charmap["tile_size"]
        charmap = {t["type"]: (t["row"], t["column"])
                   for t in charmap["tiles"]}
        return Glyphs(pixels, tile_size, charmap)

    def measure(self, text: str) -> Tuple[int, int]:
        w = len(text) * self.size + len(text) - 1
        return self.size, w

    def glyph(self, g: str) -> np.ndarray:
        r0, c0 = self.charmap[g]
        r0 *= self.size
        c0 *= self.size
        r1 = r0 + self.size
        c1 = c0 + self.size
        return self.pixels[r0:r1, c0:c1]

    def draw(self, text: str):
        w = len(text) * self.size + len(text) - 1
        image = np.zeros((self.size, w, 4), dtype=np.uint8)
        c0 = 0
        for g in text:
            c1 = c0 + self.size
            image[:, c0:c1] = self.glyph(g)
            c0 += self.size + 1

        return image


def load_charmap(assets_path: str, name: str) -> dict:
    charmap_path = os.path.join(assets_path, f"{name}.txt")
    with open(charmap_path, "r", encoding="utf-8") as f:
        lines = [line.strip('\n') for line in f.readlines()]

    lines = lines[1:]
    charmap = {
        "rows": len(lines),
        "columns": len(lines[0]),
        "image_path": "glyphs.png",
        "tile_size": 8,
        "tiles": []
    }

    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            charmap["tiles"].append({
                "name": "glyph",
                "type": char,
                "row": r,
                "column": c
            })

    return charmap


def create_power_pellet(input_path: str, output_path: str):
    pixels = crop_alpha(read_rgba(input_path))
    h, w = pixels.shape[:2]
    anim = np.zeros((h, 256*w, 4), dtype=np.uint8)
    for i in range(256):
        hue = (i // 32) / 8
        sat = 1
        seq = i % 32
        lig = 1 - (seq if seq < 16 else 32 - seq) / 64
        rgba = list(hls_to_rgb(hue, lig, sat)) + [1]
        anim[:, i*w:(i+1)*w] = (pixels * rgba).astype(np.uint8)

    Image.fromarray(anim).quantize().save(output_path, optimize=True)


def create_player(pixels: np.ndarray, assets_path: str, name: str):
    pixels = crop_alpha(pixels)
    h, w = pixels.shape[:2]
    anim = np.zeros((4*h, w, 4), dtype=np.uint8)
    for i in range(4):
        anim[i*h:(i+1)*h] = pixels
        pixels = np.rot90(pixels, k=-1, axes=(0, 1))

    assets_path = os.path.join(assets_path, "grads")
    if not os.path.exists(assets_path):
        os.makedirs(assets_path)

    Image.fromarray(anim).quantize(16).save(os.path.join(assets_path, f"{name}.png"), optimize=True)


def create_professor(pixels: np.ndarray, assets_path: str, name: str):
    pixels = crop_alpha(pixels)
    h, w = pixels.shape[:2]
    image = np.zeros((h * 4, w * 3, 4), dtype=np.uint8)
    for i in range(4):
        image[i*h:(i+1)*h, :w] = pixels
        frightened = pixels.copy()
        eaten = pixels.copy()
        for r in range(h):
            for c in range(w):
                rgba = list(pixels[r, c]/255)
                if rgba[-1] == 0:
                    continue

                hls = list(rgb_to_hls(*rgba[:3]))
                frightened[r, c, :3] = [int(255 * x) for x in hls_to_rgb(240 / 360,
                                                                         hls[1],
                                                                         max(hls[2], 0.5))]
                eaten[r, c, :3] = [int(255 * x) for x in hls_to_rgb(hls[0],
                                                                    (1 + hls[1]) * 0.5,
                                                                    0)]

        image[i*h:(i+1)*h, w:2*w] = frightened
        image[i*h:(i+1)*h, 2*w:] = eaten

        pixels = np.rot90(pixels, k=-1, axes=(0, 1))

    pixels0 = image[:, w:2*w]
    pixels1 = image[:, :w]
    flash = np.zeros((4*h, 60*w, 4), dtype=np.uint8)
    svals = np.zeros(60)
    svals[0:30] = np.linspace(1, 0, 30)
    svals[30:] = np.linspace(0, 1, 30)
    for i in range(60):
        s0 = svals[i]
        s1 = 1 - s0
        flash[:, i*w:(i+1)*w] = (pixels0 * s0 + pixels1 * s1).astype(np.uint8)

    assets_path = os.path.join(assets_path, "faculty")
    if not os.path.exists(assets_path):
        os.makedirs(assets_path)

    Image.fromarray(image).quantize(64).save(os.path.join(assets_path, f"{name}.png"), optimize=True)
    Image.fromarray(flash).quantize().save(os.path.join(assets_path, f"{name}_flash.png"), optimize=True)


GRAD_NAMES = ["imogen", "iggy", "james", "john"]


def create_select_screen(glyphs: Glyphs, grad_sprites: List[Sprite], assets_path: str):
    avatar_size = glyphs.size * len(max(GRAD_NAMES, key=len))
    title = glyphs.draw("POSTGRAD")
    pick = glyphs.draw("[<] Select Your Grad [>]")
    start = glyphs.draw("[^] to Start!")
    scale = np.ones((5, 5, 1))
    title = np.kron(title, scale)
    players = [p.pixels for p in grad_sprites]
    scale = avatar_size // players[0].shape[0] + 1
    scale = np.ones((scale, scale, 1))
    players = [np.kron(p, scale) for p in players]
    names = [glyphs.draw(name.capitalize()) for name in GRAD_NAMES]

    title_height, title_width = title.shape[:2]
    pick_height, pick_width = pick.shape[:2]
    start_height, start_width = start.shape[:2]
    name_height = names[0].shape[0]
    avatar_size = players[0].shape[0]
    spacing = 2 * glyphs.size
    avavar_width = len(players) * (avatar_size + spacing) - spacing
    width = max(avavar_width, title_width)
    height = title_height + 3 * spacing + pick_height + spacing + avatar_size + name_height + spacing + start_height
    pixels = np.zeros((height, width, 4), dtype=np.uint8)

    r = 0
    c = (width - title_width) // 2
    pixels[:title_height, c:c + title_width] = title
    r += title_height + 3 * spacing
    c = (width - pick_width) // 2
    pixels[r:r + pick_height, c:c + pick_width] = pick
    r += pick_height + spacing
    c = (width - avavar_width) // 2
    info = []
    for i, (grad, name) in enumerate(zip(players, names)):
        info.append({
            "name": GRAD_NAMES[i],
            "color": tuple(grad[12, 24, :3]),
            "bb": {
                "left": c,
                "top": r,
                "right": c + avatar_size,
                "bottom": r + avatar_size + name_height + spacing // 2
            }
        })
        pixels[r:r + avatar_size, c:c + avatar_size] = grad
        nr = r + avatar_size
        name_width = name.shape[1]
        nc = c + (avatar_size - name_width) // 2
        pixels[nr:nr + name_height, nc:nc + name_width] = name

        c += avatar_size + spacing

    r += avatar_size + name_height + spacing
    c = (width - start_width) // 2
    pixels[r:r + start_height, c:c + start_width] = start

    info = {
        "image_path": "select_screen.png",
        "select_info": info
    }
    image = Image.fromarray(pixels).quantize(64)
    image.save(os.path.join(assets_path, "select_screen.png"), optimize=True)
    with open(os.path.join(assets_path, "select_screen.json"), "w") as f:
        json.dump(info, f, indent=4)


PROF_INFO = [
    ["Professor Blink",
     "Blink's a junior professor with lots",
     "of energy and a grant to write. He  ",
     "has an uncanny ability to find you  ",
     "anywhere in the building, because   ",
     "ten years ago he was in your shoes. "
     ],
    ["Professor Pink",
     "Pink is your fashionable advisor.   ",
     "He is always dressed to impress and ",
     "you seem to run into him all the    ",
     "time. He wants to read the latest   ",
     "draft of your thesis."
     ],
    ["Professor Ink",
     "Professor Sir Ebonus Ink is the head",
     "of division and he is trying to get ",
     "you to attend a seminar, but is also",
     "trying to avoid a conversation with ",
     "Blink about funding."
     ],
    ["Professor Sue",
     "Sue is the much beloved emeritus    ",
     "chair, who wants you to proofread   ",
     "her memoirs. She knows every nook   ",
     "and cranny but luckily for you she  ",
     "keeps forgetting her glasses."
     ]
]


def create_faculty_info_screen(glyphs: Glyphs, faculty: List[Sprite], assets_path: str):
    line_length = max(len(line) for line in PROF_INFO[0])
    line_height, line_width = glyphs.measure("A" * line_length)
    avatar_size = faculty[0].pixels.shape[0] * 4
    spacing = glyphs.size
    height = len(faculty) * (2 * spacing + avatar_size + spacing) - spacing
    width = avatar_size + spacing + line_width
    image = np.zeros((height, width, 4), dtype=np.uint8)
    r = 0
    for info, sprite in zip(PROF_INFO, faculty):
        avatar = sprite.pixels
        scale = np.ones((4, 4, 1))
        avatar = np.kron(avatar, scale)
        name = glyphs.draw(info[0])
        scale = np.ones((2, 2, 1))
        name = np.kron(name, scale)
        text = [glyphs.draw(line) for line in info[1:]]
        name_height, name_width = name.shape[:2]
        text_height = line_height * (len(info) - 1) + len(info) - 2

        image[r:r + name_height, 4:name_width+4] = name
        r += name_height
        image[r:r + avatar_size, :avatar_size] = avatar
        c = avatar_size + spacing // 2
        rr = r + (avatar_size - text_height) // 2
        for t in text:
            h, w = t.shape[:2]
            image[rr:rr + h, c:c + w] = t
            rr += h + 1
        r += avatar_size + spacing

    image = Image.fromarray(image).quantize(64)
    image.save(os.path.join(assets_path, "faculty_info.png"), optimize=True)


def create_assets():
    dirname = os.path.dirname(__file__)
    tilesheet_path = os.path.join(
        dirname, "topdown_shooter_pixel", "tilesheet_transparent.png")
    guides_path = os.path.join(dirname, "wall_guides.txt")
    food_path = os.path.join(dirname, "food")
    scores_path = os.path.join(dirname, "scores.png")
    assets_path = os.path.join(dirname, "..", "src", "postgrad", "assets")
    assets_path = os.path.abspath(assets_path)

    tilesheet = Tilesheet.from_image(tilesheet_path, 17, 1)
    guides = read_guides(guides_path)
    walls = Wall.load_walls(tilesheet, 10, 0, "wall", guides)
    floors = [Floor(0, tilesheet.read_tile(0, 11))]
    door = tilesheet.read_tile(17, 10)
    scoresheet = Tilesheet.from_image(scores_path, 16, 0)
    players = [
        Sprite.from_tilesheet(tilesheet, 3, -5, "grad", "0"),
        Sprite.from_tilesheet(tilesheet, 9, -5, "grad", "1"),
        Sprite.from_tilesheet(tilesheet, 10, -5, "grad", "2"),
        Sprite.from_tilesheet(tilesheet, 15, -5, "grad", "3"),
    ]
    faculty = [
        Sprite.from_tilesheet(tilesheet, 1, -5, "faculty", "0"),
        Sprite.from_tilesheet(tilesheet, 5, -5, "faculty", "1"),
        Sprite.from_tilesheet(tilesheet, 14, -5, "faculty", "2"),
        Sprite.from_tilesheet(tilesheet, 4, -5, "faculty", "3"),
    ]
    faculty_types = set(WallType.from_string(t) for t in FacultyTiles)
    faculty_room = [w for w in Wall.load_walls(tilesheet, 4, 0, "faculty_room", guides)
                    if w.type in faculty_types]

    food = Sprite.load_food(food_path)

    scores = [
        Sprite.from_tilesheet(scoresheet, 0, 0, "score", "100"),
        Sprite.from_tilesheet(scoresheet, 0, 1, "score", "200"),
        Sprite.from_tilesheet(scoresheet, 0, 2, "score", "400"),
        Sprite.from_tilesheet(scoresheet, 0, 3, "score", "500"),
        Sprite.from_tilesheet(scoresheet, 0, 4, "score", "700"),
        Sprite.from_tilesheet(scoresheet, 0, 5, "score", "800"),
        Sprite.from_tilesheet(scoresheet, 0, 6, "score", "1000"),
        Sprite.from_tilesheet(scoresheet, 0, 7, "score", "1600"),
        Sprite.from_tilesheet(scoresheet, 0, 8, "score", "2000"),
        Sprite.from_tilesheet(scoresheet, 0, 9, "score", "5000")
    ]

    pellet = Sprite.from_png(os.path.join(
        dirname, "pellet.png"), "pellet", "pellet")

    total_tiles = len(walls) + len(floors) + len(faculty_room) + \
        1 + len(food) + len(scores) + len(players)
    rows = 8
    columns = total_tiles // rows
    if total_tiles % rows != 0:
        columns += 1

    assets = Tilesheet.create(rows, columns, tilesheet.tile_size - 1)

    for w in walls:
        assets.add_tile(w.pixels, w.info)

    for f in floors:
        assets.add_tile(f.pixels, f.info)

    for f in faculty_room:
        assets.add_tile(f.pixels, f.info)

    assets.add_tile(door, "faculty_room_door")

    for p in players:
        assets.add_tile(p.pixels, p.info)

    create_player(players[0].pixels, assets_path, "0")
    create_player(players[1].pixels, assets_path, "1")
    create_player(players[2].pixels, assets_path, "2")
    create_player(players[3].pixels, assets_path, "3")

    create_professor(faculty[0].pixels, assets_path, "blink")
    create_professor(faculty[1].pixels, assets_path, "pink")
    create_professor(faculty[2].pixels, assets_path, "ink")
    create_professor(faculty[3].pixels, assets_path, "sue")

    for f in food:
        assets.add_tile(f.pixels, f.info)

    for s in scores:
        assets.add_tile(s.pixels, s.info)

    assets.add_tile(pellet.pixels, pellet.info)
    assets.save(os.path.join(assets_path, "assets.json"))

    create_power_pellet(os.path.join(dirname, "power_pellet.png"),
                        os.path.join(assets_path, "power_pellet.png"))

    charmap = load_charmap(dirname, "outrunner")
    glyphs_path = os.path.join(dirname, "outrunner.png")
    shutil.copyfile(glyphs_path,
                    os.path.join(assets_path, "glyphs.png"))
    with open(os.path.join(assets_path, "glyphs.json"), "w") as f:
        json.dump(charmap, f, indent=4)

    glyphs = Glyphs.create(glyphs_path, charmap)
    create_select_screen(glyphs, players, assets_path)
    create_faculty_info_screen(glyphs, faculty, assets_path)


if __name__ == "__main__":
    create_assets()
