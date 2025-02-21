from typing import List, NamedTuple, Set
from random import choice, shuffle

import numpy as np

from postgrad.video_writer import VideoWriter


class Tile(NamedTuple("Tile", [("row", int), ("column", int)])):
    def distance(self, other: "Tile") -> int:
        return abs(self.row - other.row) + abs(self.column - other.column)

    def between(self, other: "Tile") -> "Tile":
        return Tile((self.row + other.row) // 2,
                    (self.column + other.column) // 2)

    def up(self, d=1) -> "Tile":
        return Tile(self.row - d, self.column)

    def down(self, d=1) -> "Tile":
        return Tile(self.row + d, self.column)

    def left(self, d=1) -> "Tile":
        return Tile(self.row, self.column - d)

    def right(self, d=1) -> "Tile":
        return Tile(self.row, self.column + d)

    def num_walls(self, walls: Set["Tile"]) -> int:
        return sum(w in walls for w in (self.up(), self.down(),
                                        self.left(), self.right()))

    def room(self, floors: Set["Tile"]) -> Set["Tile"]:
        tiles = set()
        frontier = [self]
        while frontier:
            tile = frontier.pop()
            if tile in tiles:
                continue

            tiles.add(tile)
            for n in (tile.up(), tile.down(), tile.left(), tile.right()):
                if n in floors:
                    frontier.append(n)

        return tiles

    def walls(self, walls: Set["Tile"],
              floors: Set["Tile"]) -> List["Tile"]:
        result = []
        for n in (self.up(2), self.down(2),
                  self.left(2), self.right(2)):
            wall = self.between(n)
            if n in floors and wall in walls:
                result.append(wall)

        return result


def init_maze(rows: int, columns: int, floors: Set[Tile], walls: Set[Tile]):
    assert rows % 2 == 1 and columns % 2 == 1
    for r in range(rows):
        for c in range(columns):
            walls.add(Tile(r, c))

    for r in range(1, rows, 2):
        for c in range(1, columns, 2):
            t = Tile(r, c)
            floors.add(Tile(r, c))
            yield t

    walls.difference_update(floors)


def remove_walls(walls: Set[Tile], floors: Set[Tile]):
    frontier: List[Tile] = list(floors)
    shuffle(frontier)

    for tile in frontier:
        to_remove = tile.walls(walls, floors)
        if len(to_remove) == 0:
            continue

        wall = choice(to_remove)
        walls.remove(wall)
        floors.add(wall)
        yield wall


def connect_rooms(walls: Set[Tile], floors: Set[Tile]):
    frontier: List[Tile] = list(floors)
    for tile in frontier:
        room = tile.room(floors)
        if room == floors:
            break

        roomDividers: Set[Tile] = set()
        for t in room:
            for n in (t.up(2), t.down(2), t.left(2), t.right(2)):
                if n in floors and n not in room:
                    roomDividers.add(t.between(n))

        if not roomDividers:
            continue

        wall = choice(list(roomDividers))
        walls.remove(wall)
        floors.add(wall)
        yield room, wall


def draw_maze(image: np.ndarray, floors: Set[Tile], tilesize: int):
    image[:] = 60
    for r, c in floors:
        image[r * tilesize:(r + 1) * tilesize,
              c * tilesize:(c + 1) * tilesize] = 255


def animate(path: str, rows: int, columns: int, tilesize=64, framerate=8):
    with VideoWriter(path, (columns * tilesize, rows * tilesize),
                     framerate=framerate) as video:
        floors = set()
        walls = set()
        draw_maze(video.frame, floors, tilesize)
        video.write_frame()
        video.write_frame()
        for t in init_maze(rows, columns, floors, walls):
            draw_maze(video.frame, floors, tilesize)
            r, c = t
            video.frame[r * tilesize:(r + 1) * tilesize,
                        c * tilesize:(c + 1) * tilesize] = [255, 0, 0]
            video.write_frame()

        for t in remove_walls(walls, floors):
            draw_maze(video.frame, floors, tilesize)
            r, c = t
            video.frame[r * tilesize:(r + 1) * tilesize,
                        c * tilesize:(c + 1) * tilesize] = [255, 0, 0]
            video.write_frame()
            video.write_frame()

        for room, t in connect_rooms(walls, floors):
            draw_maze(video.frame, floors, tilesize)
            for r, c in room:
                video.frame[r * tilesize:(r + 1) * tilesize,
                            c * tilesize:(c + 1) * tilesize] = [255, 128, 128]

            r, c = t
            video.frame[r * tilesize:(r + 1) * tilesize,
                        c * tilesize:(c + 1) * tilesize] = [255, 0, 0]
            video.write_frame()
            video.write_frame()
            video.write_frame()
            video.write_frame()

        draw_maze(video.frame, floors, tilesize)
        video.write_frame()
        video.write_frame()


if __name__ == "__main__":
    animate("maze.mp4", 21, 21)
