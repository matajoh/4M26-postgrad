"""Module providing logic for game levels."""

from enum import IntFlag
from typing import List, Mapping, NamedTuple, FrozenSet, Set

import numpy as np

from .drawing import asset_path, blend, SpriteSheet, Tilesheet, erase
from .geometry import Direction, Vec2
from .sprite import Sprite


class Connectivity(IntFlag):
    """This is used in drawing the level and can be ignored."""
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
    def from_string(s: str) -> "Connectivity":
        val = Connectivity.Single
        dir_set = set(s.split("/"))
        if "nw" in dir_set:
            val |= Connectivity.NorthWest
        if "n" in dir_set:
            val |= Connectivity.North
        if "ne" in dir_set:
            val |= Connectivity.NorthEast
        if "w" in dir_set:
            val |= Connectivity.West
        if "e" in dir_set:
            val |= Connectivity.East
        if "sw" in dir_set:
            val |= Connectivity.SouthWest
        if "s" in dir_set:
            val |= Connectivity.South
        if "se" in dir_set:
            val |= Connectivity.SouthEast

        return val

    def __str__(self) -> str:
        dirs = []
        n = self & Connectivity.North
        w = self & Connectivity.West
        e = self & Connectivity.East
        s = self & Connectivity.South
        if n:
            dirs.append("n")
        if w:
            dirs.append("w")
        if e:
            dirs.append("e")
        if s:
            dirs.append("s")

        if n and w and self & Connectivity.NorthWest:
            dirs.append("nw")

        if n and e and self & Connectivity.NorthEast:
            dirs.append("ne")

        if s and w and self & Connectivity.SouthWest:
            dirs.append("sw")

        if s and e and self & Connectivity.SouthEast:
            dirs.append("se")

        return "/".join(dirs)


class Tile(NamedTuple("Tile", [("row", int), ("column", int)])):
    """A tile in the level grid.
    
    Each level is made of a tiles, but the grad and professors can move
    within a tile. As such, we often will have to move between tiles
    and `Vec2` instances.
    """
    def __sub__(self, other: "Tile") -> "Tile":
        return Tile(self.row - other.row, self.column - other.column)

    def l1norm(self) -> int:
        return abs(self.row) + abs(self.column)

    def clip(self, rmin: int, cmin: int, rmax: int, cmax: int) -> "Tile":
        row = min(max(self.row, rmin), rmax)
        column = min(max(self.column, cmin), cmax)
        return Tile(row, column)

    def connectivity(self, tiles: Set["Tile"]) -> Connectivity:
        connectivity = Connectivity.Single
        if Tile(self.row - 1, self.column) in tiles:
            connectivity |= Connectivity.North
        if Tile(self.row + 1, self.column) in tiles:
            connectivity |= Connectivity.South
        if Tile(self.row, self.column - 1) in tiles:
            connectivity |= Connectivity.West
        if Tile(self.row, self.column + 1) in tiles:
            connectivity |= Connectivity.East
        if Tile(self.row - 1, self.column - 1) in tiles:
            connectivity |= Connectivity.NorthWest
        if Tile(self.row - 1, self.column + 1) in tiles:
            connectivity |= Connectivity.NorthEast
        if Tile(self.row + 1, self.column - 1) in tiles:
            connectivity |= Connectivity.SouthWest
        if Tile(self.row + 1, self.column + 1) in tiles:
            connectivity |= Connectivity.SouthEast

        return connectivity

    def neighbor(self, d: Direction) -> "Tile":
        match d:
            case Direction.UP:
                return Tile(self.row - 1, self.column)
            case Direction.RIGHT:
                return Tile(self.row, self.column + 1)
            case Direction.DOWN:
                return Tile(self.row + 1, self.column)
            case Direction.LEFT:
                return Tile(self.row, self.column - 1)

    def neighbors(self) -> List["Tile"]:
        return [(self.neighbor(d), d) for d in Direction]

    def to_vec(self, size: int) -> Vec2:
        """Convert this tile to a Vec2 corresponding to its centre."""
        half_size = size // 2
        return Vec2(self.column * size + half_size, self.row * size + half_size)


StartTiles = NamedTuple("StartTiles", [("grad", Tile),
                                       ("blink", Tile),
                                       ("pink", Tile),
                                       ("ink", Tile),
                                       ("sue", Tile)])


EdgeMap = Mapping[Direction, Tile]


class TileGraph(NamedTuple("Graph", [("tiles", FrozenSet[Tile]),
                                     ("edges", Mapping[Tile, EdgeMap])])):
    """This is a graph representation for a level.
    
    Note how we both store a (frozen) set of tiles and also an edge list.
    The edge list is keyed by direction, which is helpful for facing-restricted
    pathfinding and provides extra structure.
    """
    @staticmethod
    def create(tiles: FrozenSet[Tile],
               warp_left: Tile,
               warp_right: Tile) -> "TileGraph":
        edges = {}
        for v in tiles:
            edges[v] = {}
            for n, d in v.neighbors():
                if n in tiles:
                    edges[v][d] = n

        edges[warp_left][Direction.LEFT] = warp_right
        edges[warp_right][Direction.RIGHT] = warp_left
        return TileGraph(tiles, edges)


PelletCounts = NamedTuple("PelletCounts", [("pellets", int), ("power_pellets", int)])


class Level(NamedTuple("Level", [
    ("rows", int),
    ("columns", int),
    ("player_graph", TileGraph),
    ("faculty_graph", TileGraph),
    ("power_pellets", FrozenSet[Vec2]),
    ("warp_left", Tile),
    ("warp_right", Tile),
    ("start", StartTiles),
    ("item", Tile),
    ("pellets", FrozenSet[Vec2]),
    ("pellet_counts", PelletCounts),
    ("image", np.ndarray),
    ("pellets_image", np.ndarray),
    ("tilesize", int)
])):
    """Class representing a level in the game.

    The logic in this class is largely graphics related and can be ignored.
    """
    @staticmethod
    def create(tilesheet: Tilesheet, s: str = None) -> "Level":
        if s is None:
            path = asset_path("default_level.txt")
            with open(path, "r") as f:
                s = f.read()

        lines = s.split("\n")
        rows = len(lines)
        columns = len(lines[0])
        walls: Set[Tile] = set()
        faculty_room: Set[Tile] = set()
        warp_left: Tile = None
        warp_right: Tile = None
        start: Tile = [None] * 5
        item: Tile = None
        power_pellets: Set[Tile] = set()
        floors: List[Tile] = []
        player_tiles: Set[Vec2] = set()
        faculty_tiles: Set[Tile] = set()
        for row, line in enumerate(lines):
            for column, char in enumerate(line):
                t = Tile(row, column)
                match char:
                    case "#":
                        walls.add(t)
                    case "F":
                        faculty_room.add(t)
                    case "1":
                        door = t
                        start[1] = t
                        faculty_tiles.add(t)
                    case "2":
                        start[2] = t
                        faculty_room.add(t)
                        faculty_tiles.add(t)
                    case "3":
                        start[3] = t
                        faculty_room.add(t)
                        faculty_tiles.add(t)
                    case "4":
                        start[4] = t
                        faculty_room.add(t)
                        faculty_tiles.add(t)
                    case "L":
                        warp_left = t
                        floors.append(t)
                        player_tiles.add(t)
                        player_tiles.add(Tile(row, column - 1))
                    case "R":
                        warp_right = t
                        floors.append(t)
                        player_tiles.add(t)
                        player_tiles.add(Tile(row, column + 1))
                    case "S":
                        start[0] = t
                        floors.append(t)
                        player_tiles.add(t)
                    case "I":
                        item = t
                        floors.append(t)
                        player_tiles.add(t)
                    case "P":
                        power_pellets.add(t.to_vec(tilesheet.tile_size))
                        floors.append(t)
                        player_tiles.add(t)
                    case ".":
                        floors.append(t)
                        player_tiles.add(t)

        assert warp_left is not None
        assert warp_right is not None
        assert start is not None
        assert faculty_room

        image = np.zeros((rows * tilesheet.tile_size,
                          columns * tilesheet.tile_size, 4),
                         dtype=np.uint8)
        for tile in walls:
            connectivity = tile.connectivity(walls)
            r = tile.row * tilesheet.tile_size
            c = tile.column * tilesheet.tile_size
            image[r:r + tilesheet.tile_size,
                  c:c + tilesheet.tile_size] = tilesheet.read_tile("wall", str(connectivity))

        floor_tiles = [t for t in tilesheet.tiles if t[0] == "floor"]
        for floor in floors:
            r = floor.row * tilesheet.tile_size
            c = floor.column * tilesheet.tile_size
            if c < 0 or c >= image.shape[1]:
                continue

            tile = floor_tiles[(floor.row * columns +
                                floor.column) % len(floor_tiles)]
            image[r:r + tilesheet.tile_size,
                  c:c + tilesheet.tile_size] = tilesheet.read_tile(*tile)

        no_pellets = set(start)
        faculty_room = faculty_room.union([door])
        for tile in faculty_room:
            connectivity = tile.connectivity(faculty_room)
            r = tile.row * tilesheet.tile_size
            c = tile.column * tilesheet.tile_size
            image[r:r + tilesheet.tile_size,
                  c:c + tilesheet.tile_size] = tilesheet.read_tile("faculty_room", str(connectivity))

        left = min(faculty_room, key=lambda t: t.column)
        right = max(faculty_room, key=lambda t: t.column)
        top = min(faculty_room, key=lambda t: t.row)
        bottom = max(faculty_room, key=lambda t: t.row)
        for r in range(top.row - 2, bottom.row + 3):
            for c in range(left.column - 2, right.column + 3):
                no_pellets.add(Tile(r, c))

        for r in range(-3, 4):
            for c in range(-3, 4):
                no_pellets.add(Tile(warp_left.row + r, warp_left.column + c))
                no_pellets.add(Tile(warp_right.row + r, warp_right.column + c))

        r = door.row * tilesheet.tile_size
        c = door.column * tilesheet.tile_size
        tile = blend(tilesheet.read_tile("faculty_room", "n/w/e/s/nw/ne/sw/se"),
                     tilesheet.read_tile("faculty_room_door", "faculty_room_door"))
        image[r:r + tilesheet.tile_size,
              c:c + tilesheet.tile_size] = tile
        no_pellets.add(door)

        spritesheet = SpriteSheet.load()
        pellet_image = spritesheet.read_sprite("pellet", "pellet")

        pellets = set()
        for floor in floors:
            if floor in no_pellets:
                continue

            pos = floor.to_vec(tilesheet.tile_size)
            if pos not in power_pellets:
                pellets.add(pos)

            for n, _ in floor.neighbors():
                if n in no_pellets:
                    continue

                if n in floors:
                    pellets.add((pos + n.to_vec(tilesheet.tile_size)) // 2)

        pellets_image = np.zeros_like(image)
        h, w = pellet_image.shape[:2]
        for p in pellets:
            c0, r0 = p - Vec2(w//2, h//2)
            c1, r1 = c0 + w, r0 + h
            pellets_image[r0:r1, c0:c1] = pellet_image

        pellets = frozenset(pellets)
        player_graph = TileGraph.create(frozenset(player_tiles), warp_left, warp_right)
        faculty_graph = TileGraph.create(frozenset(faculty_tiles).union(player_tiles), warp_left, warp_right)
        power_pellets = frozenset(power_pellets)
        start_tiles = StartTiles(*start)
        pellet_counts = PelletCounts(len(pellets), len(power_pellets))
        return Level(rows, columns, player_graph, faculty_graph, power_pellets, warp_left,
                     warp_right, start_tiles, item, pellets, pellet_counts, image, pellets_image,
                     tilesheet.tile_size)

    def is_outside(self, pos: Vec2) -> bool:
        return pos.x < 0 or pos.x >= self.columns * self.tilesize or pos.y < 0 or pos.y >= self.rows * self.tilesize

    def is_inside(self, pos: Vec2) -> bool:
        return not self.is_outside(pos)

    def tile_to_vec(self, tile: Tile) -> Vec2:
        return tile.to_vec(self.tilesize)

    def draw(self):
        image = blend(self.image, self.pellets_image)
        return image

    def adjust(self, sprite: Sprite) -> Sprite:
        pos = sprite.pos
        facing = sprite.facing
        tile = self.vec_to_tile(pos)
        center = tile.to_vec(self.tilesize)
        if facing in {Direction.UP, Direction.DOWN}:
            if pos.x == center.x and facing in self.faculty_graph.edges[tile]:
                pos = Vec2(center.x, pos.y)
            else:
                if ((facing == Direction.UP and pos.y < center.y) or
                        (facing == Direction.DOWN and pos.y > center.y)):
                    pos = Vec2(pos.x, center.y)
        else:
            if pos.y == center.y and facing in self.faculty_graph.edges[tile]:
                pos = Vec2(pos.x, center.y)
            else:
                if ((facing == Direction.LEFT and pos.x < center.x) or
                        (facing == Direction.RIGHT and pos.x > center.x)):
                    pos = Vec2(center.x, pos.y)

        return sprite._replace(pos=pos)

    def check_warp(self, sprite: Sprite) -> Sprite:
        tile = self.vec_to_tile(sprite.pos)
        if tile == self.warp_left.neighbor(Direction.LEFT):
            pos = self.tile_to_vec(self.warp_right)
            return sprite.teleport(pos + Vec2(self.tilesize // 2 - 1, 0))

        if tile == self.warp_right.neighbor(Direction.RIGHT):
            pos = self.tile_to_vec(self.warp_left)
            return sprite.teleport(pos - Vec2(self.tilesize // 2 - 1, 0))

        return sprite

    def vec_to_tile(self, pos: Vec2) -> Tile:
        r = int(pos.y // self.tilesize)
        c = int(pos.x // self.tilesize)
        return Tile(r, c)

    def snap_to_grid(self, pos: Vec2) -> Vec2:
        return self.vec_to_tile(pos).to_vec(self.tilesize)

    def find_nearest_tile(self, paths: FrozenSet[Tile], v: Vec2) -> Tile:
        t = self.vec_to_tile(v).clip(0, 0, self.rows-1, self.columns-1)
        return min(paths, key=lambda p: (p - t).l1norm())

    def eat_pellets(self, sprite: Sprite) -> "Level":
        half_size = self.tilesize // 2
        pos = sprite.pos

        eaten = set([pos])
        for i in range(-half_size // 2, half_size // 2 + 1):
            eaten.update([pos + Vec2(0, i), pos + Vec2(i, 0)])

        pellets = self.pellets
        power_pellets = self.power_pellets
        pellets_eaten = pellets.intersection(eaten)
        power_pellets_eaten = self.power_pellets.intersection(eaten)

        if not pellets_eaten and not power_pellets_eaten:
            return self

        pellets_image = np.copy(self.pellets_image)
        if pellets_eaten:
            erase_size = half_size // 2
            pellets = self.pellets - pellets_eaten
            for p in pellets_eaten:
                erase(pellets_image, p, erase_size)

        if power_pellets_eaten:
            erase_size = half_size
            power_pellets = self.power_pellets - power_pellets_eaten
            for p in power_pellets_eaten:
                erase(pellets_image, p, erase_size)

        return self._replace(pellets=pellets,
                             power_pellets=power_pellets,
                             pellets_image=pellets_image)

    @property
    def remaining_pellets(self) -> int:
        return len(self.pellets) + len(self.power_pellets)

    @property
    def pellets_eaten(self) -> int:
        return self.pellet_counts.pellets - len(self.pellets)

    @property
    def power_pellets_eaten(self) -> int:
        return self.pellet_counts.power_pellets - len(self.power_pellets)
