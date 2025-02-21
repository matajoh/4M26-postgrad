from heapq import heappop, heappush
from random import choice, shuffle
from typing import List, NamedTuple, Set

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


def draw_square(image: np.ndarray, tile: Tile, color: List[int], tilesize: int):
    half = tilesize // 2
    r = tile.row * tilesize + half - tilesize // 4
    c = tile.column * tilesize + half - tilesize // 4
    image[r:r+tilesize // 2, c:c+tilesize // 2] = color


def draw_line(image: np.ndarray, lhs: Tile, rhs: Tile, color: List[int], tilesize: int):
    half = tilesize // 2
    r0, c0 = lhs
    r1, c1 = rhs
    if r1 < r0:
        r0, r1 = r1, r0

    if c1 < c0:
        c0, c1 = c1, c0

    r0 = (r0 * tilesize) + half - (tilesize // 16)
    c0 = (c0 * tilesize) + half - (tilesize // 16)
    r1 = (r1 * tilesize) + half + (tilesize // 16)
    c1 = (c1 * tilesize) + half + (tilesize // 16)
    image[r0:r1, c0:c1] = color


class AStar:
    def __init__(self, start: Tile, goal: Tile, graph: Set[Tile]):
        self.frontier = []
        heappush(self.frontier, (0, 0, start))
        self.came_from = {start: None}
        self.cost_so_far = {start: 0}
        self.current = start
        self.start = start
        self.goal = goal
        self.graph = graph

    def distance(self, lhs: Tile, rhs: Tile):
        return lhs.distance(rhs)

    def heuristic(self, t: Tile):
        return t.distance(self.goal)

    def neighbors(self, t: Tile):
        for n in (t.up(), t.down(), t.left(), t.right()):
            if n in self.graph:
                yield n

    def step(self) -> bool:
        _, _, x = heappop(self.frontier)
        self.current = x

        if x == self.goal:
            return True

        for y in self.neighbors(x):
            new_cost = self.cost_so_far[x] + self.distance(x, y)
            if new_cost < self.cost_so_far.get(y, float("inf")):
                self.cost_so_far[y] = new_cost
                h = self.heuristic(y)
                priority = new_cost + self.heuristic(y)
                heappush(self.frontier, (priority, h, y))
                self.came_from[y] = x

        return False

    def draw(self, image: np.ndarray, tilesize: int):
        max_cost = self.distance(self.start, self.goal) * 3 // 2
        image[:] = 60
        for tile in self.graph:
            r = tile.row * tilesize
            c = tile.column * tilesize
            image[r:r+tilesize, c:c+tilesize] = 255

        for tile in self.came_from:
            if self.came_from[tile] is None:
                continue

            cost = self.cost_so_far[tile]
            color = int(128 * cost / max_cost)
            draw_line(image, tile, self.came_from[tile], color, tilesize)

        for tile in self.came_from:
            cost = self.cost_so_far[tile]
            color = int(128 * cost / max_cost)
            draw_square(image, tile, color, tilesize)

        current = self.current
        while True:
            prev = self.came_from[current]
            if prev is None:
                break

            draw_line(image, current, prev, [0, 0, 255], tilesize)
            current = prev

        current = self.current
        while True:
            prev = self.came_from[current]
            if prev is None:
                break

            draw_square(image, current, [0, 0, 255], tilesize)
            current = prev

        draw_square(image, self.start, [0, 255, 0], tilesize)
        draw_square(image, self.goal, [255, 0, 0], tilesize)


def generate_maze(rows: int, columns: int) -> Set[Tile]:
    assert rows % 2 == 1 and columns % 2 == 1
    walls = set()
    for r in range(rows):
        for c in range(columns):
            walls.add(Tile(r, c))

    floors: Set[Tile] = set()
    
    for r in range(1, rows, 2):
        for c in range(1, columns, 2):
            floors.add(Tile(r, c))

    walls.difference_update(floors)

    frontier: List[Tile] = list(floors)
    shuffle(frontier)

    for tile in frontier:
        to_remove = tile.walls(walls, floors)
        if len(to_remove) == 0:
            continue

        wall = choice(to_remove)
        walls.remove(wall)
        floors.add(wall)

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

    return floors


def animate(path: str, rows: int, columns: int, tilesize=64, framerate=4):
    graph = generate_maze(rows, columns)
    start = choice([t for t in graph if t.row < rows // 4 and t.column < columns // 4])
    goal = choice([t for t in graph if t.row > 3 * rows // 4 and t.column > 3 * columns // 4])

    astar = AStar(start, goal, graph)

    with VideoWriter(path, (columns * tilesize, rows * tilesize), framerate=framerate) as video:
        astar.draw(video.frame, tilesize)
        video.write_frame()

        while not astar.step():
            astar.draw(video.frame, tilesize)
            video.write_frame()

        astar.draw(video.frame, tilesize)
        video.write_frame()


if __name__ == "__main__":
    animate("astar.mp4", 21, 21)
