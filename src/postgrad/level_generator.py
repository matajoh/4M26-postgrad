"""Module providing code for generating random levels."""

import random
from typing import List, Mapping, NamedTuple, Set

from .geometry import Direction


class Cell(NamedTuple("Cell", [("row", int), ("column", int)])):
    """A cell in the maze."""
    def potential_neighbors(self,
                            valid_floors: Set["Cell"],
                            invalid_walls: Set["Cell"]) -> List["Cell"]:
        neighbors = []
        for d in Direction:
            w = self.neighbor(d)
            if w in invalid_walls:
                continue
            n = w.neighbor(d)
            if n in valid_floors:
                neighbors.append(n)

        return neighbors

    def up(self) -> "Cell":
        return Cell(self.row - 1, self.column)

    def down(self) -> "Cell":
        return Cell(self.row + 1, self.column)

    def left(self) -> "Cell":
        return Cell(self.row, self.column - 1)

    def right(self) -> "Cell":
        return Cell(self.row, self.column + 1)

    def neighbor(self, direction: Direction) -> "Cell":
        match direction:
            case Direction.UP:
                return self.up()
            case Direction.DOWN:
                return self.down()
            case Direction.LEFT:
                return self.left()
            case Direction.RIGHT:
                return self.right()

    def wall(self, other: "Cell") -> "Cell":
        return Cell((self.row + other.row) // 2, (self.column + other.column) // 2)


def generate_maze(floors: Set[Cell], walls: Set[Cell], max_wall_length: int) -> Set[Cell]:
    """Maze generation algorithm.
    
    This maze generation algorithm is modified from the one shown in
    lecture to reduce the rejection rate when sampling. The first change
    is that instead of starting with all walls and adding floor tiles,
    we start with all floors and add walls. PostGrad requires lots of movement
    options for the player, and so this biases the samples helpfully.
    Other changes are noted in the code.
    """
    added: Set[Cell] = set()
    wall_cells: Mapping[Cell, Set[Cell]] = {cell: set([cell]) for cell in walls}

    frontier: List[Cell] = list(walls)
    # this adds the random element
    random.shuffle(frontier)
    while frontier:
        cell = frontier.pop()
        if len(wall_cells[cell]) > 1:
            continue

        to_add = []
        for d in Direction:
            w = cell.neighbor(d)
            # try adding this wall
            test = floors.difference(added, [w])
            deadend = False
            # we don't want to add walls that create dead ends
            for d2 in Direction:
                n = w.neighbor(d2)
                if n not in test:
                    continue

                count = 0
                for d3 in Direction:
                    if n.neighbor(d3) in test:
                        count += 1

                if count <= 1:
                    # this wall has created a dead end
                    deadend = True
                    break

            if deadend:
                continue

            n = w.neighbor(d)
            if n in wall_cells and len(wall_cells[n]) + 2 > max_wall_length:
                # adding this wall would create a wall that is too long
                continue

            to_add.append(d)

        if not to_add:
            # reject this sample
            raise RuntimeError("Cannot continue")

        # choose a wall to add at random
        d = random.choice(to_add)
        w = cell.neighbor(d)
        n = w.neighbor(d)
        if n in wall_cells:
            wall_cells[n].update([cell, w])
            wall_cells[cell] = wall_cells[n]
        else:
            wall_cells[cell].add(w)

        added.add(w)

    return added


def add_faculty_room(center: Cell, floors: Set[Cell], walls: Set[Cell]):
    r0, c0 = center
    faculty_room = set()
    # add the faculty room cells
    for r in range(r0 - 1, r0 + 2):
        for c in range(c0 - 2, c0 + 3):
            faculty_room.add(Cell(r, c))

    # add a corridor around the faculty room
    for r in range(r0 - 2, r0 + 3):
        floors.update([Cell(r, c0 - 3), Cell(r, c0 + 3)])

    for c in range(c0 - 3, c0 + 4):
        floors.update([Cell(r0 - 2, c), Cell(r0 + 2, c)])

    # add the walls around the corridor
    for r in range(r0 - 3, r0 + 4):
        walls.update([Cell(r, c0 - 4), Cell(r, c0 + 4)])

    for c in range(c0-4, c0+5):
        walls.update([Cell(r0-3, c), Cell(r0+3, c)])

    # poke four holes in the walls
    c = random.randint(1, 3)
    floors.update([Cell(r0-3, c0-c), Cell(r0-3, c0+c)])
    c = random.randint(1, 3)
    floors.update([Cell(r0+3, c0-c), Cell(r0+3, c0+c)])

    return faculty_room


def add_warp(row: int, columns: int, floors: Set[Cell], walls: Set[Cell]):
    # add the walls and floors around the warp
    cols = [1, 2, 3, columns-4, columns-3, columns-2]
    for r in range(row-3, row+4):
        for c in cols:
            cell = Cell(r, c)
            if r == row:
                floors.add(cell)
            else:
                walls.add(cell)

        floors.add(Cell(r, 4))
        floors.add(Cell(r, columns-5))

    # punch a hole through to the faculty corridor
    floors.add(Cell(row, 5))
    floors.add(Cell(row, columns-6))

    return Cell(row, 0), Cell(row, columns-1)


def remove_dead_ends(floors: Set[Cell], walls: Set[Cell], perm_walls: Set[Cell]):
    """Remove dead ends from the level.
    
    These rare, inadvertently created dead ends are caused by mazing
    interacting awkwardly with the fixed features of the maze. This
    function removes them by removing walls until the dead end is
    removed.
    """
    def num_neighbors(cell: Cell) -> List[Cell]:
        return sum(1 for d in Direction
                   if cell.neighbor(d) in floors)

    def valid_walls(cell: Cell) -> Set[Cell]:
        return set([cell.neighbor(d) for d in Direction
                    if cell.neighbor(d) in walls])

    dead_ends: List[Cell] = []
    for cell in floors:
        if num_neighbors(cell) == 1:
            dead_ends.append(cell)

    while dead_ends:
        current = dead_ends.pop()
        if num_neighbors(current) > 1:
            # no longer a dead end
            continue

        to_remove = valid_walls(current) - perm_walls
        if to_remove:
            # remove a random wall
            w = random.choice(list(to_remove))
            walls.remove(w)
            floors.add(w)
        else:
            # neighbors of the warps
            continue


def generate_level(rows: int, columns: int, max_wall_length: int) -> str:
    """Generate a level for PostGrad.
    
    This function uses a form of rejection sampling to sample level layouts.
    """
    assert rows % 2 == 1 and columns % 2 == 1
    floors = set()
    walls = set()
    power_pellets = []

    # add the borders
    for r in range(rows):
        walls.add(Cell(r, 0))
        walls.add(Cell(r, columns-1))
    for c in range(columns):
        walls.add(Cell(0, c))
        walls.add(Cell(rows-1, c))

    # add the warps and the faculty room
    center = Cell(rows // 2 - 1, columns // 2)
    warp_left, warp_right = add_warp(rows // 2 - 1, columns, floors, walls)
    faculty_room = add_faculty_room(center, floors, walls)

    item = center.down().down()
    start = Cell((rows * 3) // 4, columns // 2)
    if start in walls:
        floors.add(start)
        walls.remove(start)

    # starting points
    starts = [start, center.up(), center.left(), center, center.right()]

    # these walls cannot be removed
    perm_walls = walls.copy()

    # at this point, we have added the fixed features of the level, which
    # will act as constraints to the mazing.

    # seed the maze by making every even square a wall
    maze_walls = set()
    maze_floors = set()
    for r in range(rows):
        for c in range(columns):
            cell = Cell(r, c)
            if cell.row % 2 == 1 or cell.column % 2 == 1:
                maze_floors.add(cell)
            else:
                maze_walls.add(cell)

    # the wall squares cannot participate in the maze
    maze_walls -= perm_walls
    maze_floors -= perm_walls

    added = None
    for i in range(1000):
        # we will try 1000 times to generate a maze. In practice we will
        # not need anywhere near this many samples to achieve a valid
        # level layout, due to the work done to reduce the support of the
        # sampling distribution. The reason for this high number is that
        # we want a high confidence that this stage will complete, because
        # otherwise the game will crash.
        try:
            added = generate_maze(maze_floors.union(floors), maze_walls, max_wall_length)
            break
        except RuntimeError:
            pass

    if added is None:
        raise RuntimeError("Cannot generate maze")

    floors.update(maze_floors.difference(added))
    walls.update(maze_walls.union(added))

    # remove any lingering dead ends
    remove_dead_ends(floors, walls, perm_walls)

    # add the power pellets to the four corners.
    top_pp = random.choice([1, 3, 5])
    bottom_pp = random.choice([rows - 2, rows - 4, rows - 6])
    power_pellets = [Cell(top_pp, 1), Cell(top_pp, columns - 2),
                     Cell(bottom_pp, 1), Cell(bottom_pp, columns - 2)]

    # generate text
    lines = [[" " for _ in range(columns)] for _ in range(rows)]
    for cell in walls:
        lines[cell.row][cell.column] = "#"
    for cell in floors:
        lines[cell.row][cell.column] = "."
    for cell in faculty_room:
        lines[cell.row][cell.column] = "F"
    lines[warp_left.row][warp_left.column] = "L"
    lines[warp_right.row][warp_right.column] = "R"
    lines[starts[0].row][starts[0].column] = "S"
    for i in range(1, 5):
        lines[starts[i].row][starts[i].column] = str(i)
    lines[item.row][item.column] = "I"
    for cell in power_pellets:
        lines[cell.row][cell.column] = "P"

    return "\n".join("".join(row) for row in lines)
