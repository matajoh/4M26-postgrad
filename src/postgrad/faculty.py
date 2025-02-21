"""Module containing logic for the professor agents."""

from typing import Callable, List, NamedTuple, Tuple
from random import choice

import numpy as np

from .astar import astar
from .core import GameConfig, PhaseTimer, ProfessorState, SpriteKind
from .drawing import read_rgba
from .geometry import Direction, Vec2
from .level import TileGraph, Level, Tile
from .sprite import Action, Sprite


class Locations(NamedTuple("Locations", [("grad", Vec2),
                                         ("blink", Vec2),
                                         ("pink", Vec2),
                                         ("ink", Vec2),
                                         ("sue", Vec2)])):
    GRAD_INDEX = 0
    BLINK_INDEX = 1
    PINK_INDEX = 2
    INK_INDEX = 3
    SUE_INDEX = 4

    @staticmethod
    def from_level(level: Level):
        return Locations(level.tile_to_vec(level.start[Locations.GRAD_INDEX]),
                         level.tile_to_vec(level.start[Locations.BLINK_INDEX]),
                         level.tile_to_vec(level.start[Locations.PINK_INDEX]),
                         level.tile_to_vec(level.start[Locations.INK_INDEX]),
                         level.tile_to_vec(level.start[Locations.SUE_INDEX]))


AStarState = NamedTuple("State", [("tile", Tile), ("facing", Direction)])


def valid_neighbors(graph: TileGraph, tile: Tile, facing: Direction) -> List[AStarState]:
    edges = graph.edges[tile]
    return [AStarState(edges[f], f) for f in Direction
            if f in edges and f != facing.opposite()]


def shortest_path_facing(graph: TileGraph, start: AStarState, goal: Tile) -> List[AStarState]:
    """Shortest path search that incorporates facing direction."""
    def distance(a: AStarState, b: AStarState):
        return a.facing.distance(b.facing) + (a.tile - b.tile).l1norm()

    def heuristic(a: AStarState):
        return (a.tile - goal).l1norm()

    def neighbors(a: AStarState):
        return valid_neighbors(graph, a.tile, a.facing)

    def is_goal(a: AStarState):
        return a.tile == goal

    path = astar(distance, heuristic, neighbors, is_goal, start)
    return path


def shortest_path(graph: TileGraph, start: Tile, goal: Tile) -> List[Tile]:
    """Basic shortest path search."""
    def distance(a: Tile, b: Tile):
        return (a - b).l1norm()

    def heuristic(a: Tile):
        return (a - goal).l1norm()

    def neighbors(a: Tile):
        return valid_neighbors(graph, a)

    def is_goal(a: Tile):
        return a == goal

    path = astar(distance, heuristic, neighbors, is_goal, start)
    return path


GameInfo = NamedTuple("GameInfo", [("grad", Sprite), ("faculty", "Professors"),
                                   ("scatter", Locations), ("start", Locations)])

ChooseGoal = Callable[[Level, ProfessorState, GameInfo], Vec2]


class Target(NamedTuple("Target", [("pos", Vec2), ("action", Action)])):
    @staticmethod
    def create(level: Level, state: AStarState) -> "Target":
        return Target(level.tile_to_vec(state.tile),
                      Action.from_direction(state.facing))


def choose_random_turn(level: Level, prof: "Professor", default: Vec2) -> Vec2:
    tile = level.vec_to_tile(prof.pos)
    if tile in level.player_graph.tiles:
        neighbors = valid_neighbors(level.player_graph, tile, prof.facing)
    else:
        return default

    if neighbors:
        tile, _ = choice(neighbors)
        return level.tile_to_vec(tile)

    return default


def blink(level: Level, phase: ProfessorState, info: GameInfo) -> Vec2:
    match phase:
        case ProfessorState.IN_ROOM:
            return info.faculty.blink.pos
        case ProfessorState.SCATTER:
            return info.scatter.blink
        case ProfessorState.CHASE:
            # go directly for the grad
            return info.grad.pos
        case ProfessorState.FRIGHTENED:
            return choose_random_turn(level,
                                      info.faculty.blink,
                                      info.scatter.blink)
        case ProfessorState.ASKED:
            return info.start.blink


def pink(level: Level, phase: ProfessorState, info: GameInfo) -> Vec2:
    match phase:
        case ProfessorState.IN_ROOM:
            return info.faculty.pink.pos
        case ProfessorState.SCATTER:
            return info.scatter.pink
        case ProfessorState.CHASE:
            # go two tiles ahead of the grad
            return info.grad.pos.move(info.grad.facing, 2 * level.tilesize)
        case ProfessorState.FRIGHTENED:
            return choose_random_turn(level,
                                      info.faculty.pink,
                                      info.scatter.pink)
        case ProfessorState.ASKED:
            return info.start.pink


def ink(level: Level, phase: ProfessorState, info: GameInfo) -> Vec2:
    match phase:
        case ProfessorState.IN_ROOM:
            return info.faculty.ink.pos
        case ProfessorState.SCATTER:
            return info.scatter.ink
        case ProfessorState.CHASE:
            # complex behaviour combining the grad's position and the
            # Professor Blink's position
            pos = info.grad.pos.move(info.grad.facing, level.tilesize)
            return pos + (pos - info.faculty.blink.pos)
        case ProfessorState.FRIGHTENED:
            return choose_random_turn(level, info.faculty.ink,
                                      info.scatter.ink)
        case ProfessorState.ASKED:
            return info.start.ink


def sue(level: Level, phase: ProfessorState, info: GameInfo) -> Vec2:
    match phase:
        case ProfessorState.IN_ROOM:
            return info.faculty.sue.pos
        case ProfessorState.SCATTER:
            return info.scatter.sue
        case ProfessorState.CHASE:
            # Either go directly for the grad, or avoid them if you get
            # too close
            dist = (info.grad.pos - info.faculty.sue.pos).l1norm()
            if dist > 4 * level.tilesize:
                return info.grad.pos

            return info.scatter.sue
        case ProfessorState.FRIGHTENED:
            return choose_random_turn(level, info.faculty.sue,
                                      info.scatter.sue)
        case ProfessorState.ASKED:
            return info.start.sue


class Professor(NamedTuple("Professor", [("choose_goal", ChooseGoal), ("outer_timer", PhaseTimer),
                                         ("timer", PhaseTimer), ("phases", Tuple[PhaseTimer]),
                                         ("pellet_count", int), ("sprite", Sprite),
                                         ("target", Target), ("goal", Vec2),
                                         ("show_phase", bool)])):
    @property
    def phase(self) -> ProfessorState:
        return self.timer.phase

    def frightened_sprite(self) -> Sprite:
        return self.sprite.with_image(0, 1).with_velocity(1, 2).spin().spin()

    def flashing_sprite(self) -> Sprite:
        return self.sprite.with_image(1, 0)

    def eaten_sprite(self) -> Sprite:
        return self.sprite.with_image(0, 2).with_velocity(2, 1)

    def normal_sprite(self) -> Sprite:
        return self.sprite.with_image(0, 0).with_velocity(19, 20)

    def update_path(self, level: Level, info: GameInfo) -> "Professor":
        """Update the path of the professor, if needed."""
        graph = level.player_graph
        tile = level.vec_to_tile(self.pos)
        if self.phase == ProfessorState.ASKED or tile not in graph.tiles:
            graph = level.faculty_graph

        # our target is the "next tile" in our path to the current goal
        target = self.target
        if target is None:
            # we have lost our target tile, usually as a result of a phase switch.
            action = Action.from_direction(self.facing)
            grid_pos = level.snap_to_grid(self.pos)
            if grid_pos == self.pos:
                # we are at a junction so set this as our target and proceed
                return self._replace(target=Target(grid_pos, action)).update_path(level, info)

            # set our target as the tile we are moving towards
            f0 = (grid_pos - self.pos).normalize()
            f1 = Vec2(0, 0).move(self.facing, 1)
            if f0.dot(f1) >= 0:
                target = Target(grid_pos, action)
            else:
                target = Target(grid_pos.move(self.facing, level.tilesize), action)

            return self._replace(target=target).update_path(level, info)

        if self.pos != target.pos:
            # continue moving towards our target
            return self

        # pick a new goal
        goal = self.choose_goal(level, self.phase, info)

        if goal is None:
            # this should never happen, so make it really obvious
            # something is wrong
            return self._replace(sprite=self.sprite.spin())

        # we have a goal, so we compute the shortest path
        start_t = level.find_nearest_tile(graph.tiles, self.pos)
        goal_t = level.find_nearest_tile(graph.tiles, goal)
        if start_t == goal_t:
            # congratulations! we're already there
            if self.phase == ProfessorState.ASKED:
                return self._replace(sprite=self.normal_sprite(),
                                     timer=self.timer.complete())

            neighbors = valid_neighbors(graph, level.vec_to_tile(self.pos), self.facing)
            if len(neighbors) == 0:
                return self._replace(sprite=self.sprite.spin())

            # wander around aimlessly
            target = Target.create(level, neighbors[0])
        else:
            # compute the path
            path = shortest_path_facing(graph, AStarState(start_t, self.facing), goal_t)
            if path is None:
                # again, something has gone very wrong
                return self._replace(sprite=self.sprite.spin())

            # set our target to the next tile in the path
            target = Target.create(level, path[1])

        return self._replace(target=target, goal=goal)

    @property
    def kind(self) -> SpriteKind:
        return self.sprite.kind

    @property
    def pos(self) -> Vec2:
        return self.sprite.pos

    @property
    def facing(self) -> Direction:
        return self.sprite.facing

    def ask(self) -> "Professor":
        if self.show_phase:
            print(f"{self.kind} {ProfessorState.ASKED} INF")

        return self._replace(
            sprite=self.eaten_sprite(),
            timer=self.phases[ProfessorState.ASKED.index()]
        )

    def frighten(self) -> "Professor":
        if self.phase in {ProfessorState.FRIGHTENED, ProfessorState.ASKED, ProfessorState.IN_ROOM}:
            return self

        return self._replace(
            target=None,
            sprite=self.frightened_sprite(),
            timer=self.phases[ProfessorState.FRIGHTENED.index()],
            outer_timer=self.timer
        )

    def step(self, level: Level, info: GameInfo) -> "Professor":
        """Move the professor to the next state."""
        prof = self.update_path(level, info)
        sprite = prof.sprite
        if sprite.pos != prof.target.pos:
            sprite = prof.sprite.step(prof.target.action)

        pos = sprite.pos
        sprite = level.check_warp(sprite)
        if sprite.pos != pos:
            # sprite warped, force pathing refresh
            prof = prof._replace(target=Target(sprite.pos, sprite.facing))

        if self.show_phase and self.timer.ticks % 60 == 0:
            print(f"{self.kind}: {self.phase} {self.timer.ticks // 60}")

        outer_timer = prof.outer_timer
        timer = prof.timer.tick()
        if timer.is_done():
            # the current timer is done
            if outer_timer:
                # if there is an outer timer, we should return to it
                if timer.phase == ProfessorState.FRIGHTENED:
                    sprite = self.normal_sprite()

                timer = outer_timer
                outer_timer = None
            else:
                # otherwise, we proceed to the next phase timer
                timer = prof.phases[timer.index + 1]

            if self.show_phase and timer.ticks < 0:
                print(f"{self.kind} {timer.phase} INF")
        elif timer.phase == ProfessorState.IN_ROOM and level.pellets_eaten >= prof.pellet_count:
            # the grad has "eaten" enough pellets that we can leave the room
            timer = prof.phases[0]

        if timer.phase == ProfessorState.FRIGHTENED:
            flash_ticks = prof.phases[ProfessorState.FRIGHTENED.index()].ticks // 2
            if timer.ticks == flash_ticks:
                sprite = sprite.with_image(1, 0)
            elif timer.ticks < flash_ticks:
                sprite = sprite.tick()
        elif self.kind == SpriteKind.PROF_BLINK:
            if level.remaining_pellets == 10:
                sprite = sprite.with_velocity(21, 20)
            if level.remaining_pellets == 20:
                sprite = sprite.with_velocity(20, 20)

        return prof._replace(sprite=sprite, timer=timer, outer_timer=outer_timer)

    def teleport(self, pos: Vec2) -> "Professor":
        return self._replace(sprite=self.sprite.teleport(pos),
                             target=Target(pos, self.facing))

    @staticmethod
    def create(kind: SpriteKind, level: Level,
               config: GameConfig,
               facing=Direction.RIGHT,
               show_phase=False) -> "Professor":
        locations = Locations.from_level(level)
        match kind:
            case SpriteKind.PROF_BLINK:
                choose_goal = blink
                name = "blink"
                pos = locations.blink
            case SpriteKind.PROF_PINK:
                choose_goal = pink
                name = "pink"
                pos = locations.pink
            case SpriteKind.PROF_INK:
                choose_goal = ink
                name = "ink"
                pos = locations.ink
            case SpriteKind.PROF_SUE:
                choose_goal = sue
                name = "sue"
                pos = locations.sue
            case _:
                raise ValueError(f"unknown professor kind: {kind}")

        pellet_count = config.prof_pellets[kind]
        image = read_rgba("faculty", f"{name}.png")
        flash = read_rgba("faculty", f"{name}_flash.png")
        size = image.shape[0] // 4
        sprite = Sprite.create(kind, pos, size, facing, image, flash)
        sprite = sprite.with_velocity(19, 20)
        phases = config.phases
        timer = phases[ProfessorState.IN_ROOM.index()]
        return Professor(choose_goal, None, timer, phases, pellet_count,
                         sprite, Target(pos, facing), pos, show_phase)


class Professors(NamedTuple("Professors", [("blink", Professor), ("pink", Professor),
                                           ("ink", Professor), ("sue", Professor)])):
    def step(self, level: Level, info: GameInfo) -> "Professors":
        return Professors(
            self.blink.step(level, info),
            self.pink.step(level, info),
            self.ink.step(level, info),
            self.sue.step(level, info))

    def frighten(self) -> "Professors":
        return Professors(
            self.blink.frighten(),
            self.pink.frighten(),
            self.ink.frighten(),
            self.sue.frighten())

    def ask(self, p: Professor) -> "Professors":
        match p.sprite.kind:
            case SpriteKind.PROF_BLINK:
                return self._replace(blink=self.blink.ask())
            case SpriteKind.PROF_PINK:
                return self._replace(pink=self.pink.ask())
            case SpriteKind.PROF_INK:
                return self._replace(ink=self.ink.ask())
            case SpriteKind.PROF_SUE:
                return self._replace(sue=self.sue.ask())
            case _:
                raise ValueError(f"unknown professor kind: {p.sprite.kind}")


class Faculty(NamedTuple("Faculty", [("professors", Professors),
                                     ("scatter", Locations),
                                     ("start", Locations)])):
    def create(level: Level, config: GameConfig) -> "Faculty":
        professors = Professors(
            Professor.create(SpriteKind.PROF_BLINK, level, config, Direction.UP, config.show_state),
            Professor.create(SpriteKind.PROF_PINK, level, config, Direction.RIGHT, config.show_state),
            Professor.create(SpriteKind.PROF_INK, level, config, Direction.UP, config.show_state),
            Professor.create(SpriteKind.PROF_SUE, level, config, Direction.LEFT, config.show_state))
        scatter = Locations(
            None,
            level.tile_to_vec(Tile(0, level.columns-1)),
            level.tile_to_vec(Tile(0, 0)),
            level.tile_to_vec(Tile(level.rows-1, level.columns-1)),
            level.tile_to_vec(Tile(level.rows-1, 0)))
        start = Locations.from_level(level)
        return Faculty(professors, scatter, start)

    def step(self, level: Level, grad: Sprite, energized: bool) -> "Faculty":
        info = GameInfo(grad, self.professors, self.scatter, self.start)
        profs = self.professors.step(level, info)
        if energized:
            profs = profs.frighten()

        return self._replace(professors=profs)

    def ask(self, p: Professor) -> "Faculty":
        return self._replace(professors=self.professors.ask(p))

    def frighten(self) -> "Faculty":
        profs = [p.frighten() for p in self.professors]
        return self._replace(professors=Professors(*profs))

    def intersect(self, level: Level, sprite: Sprite) -> List[Professor]:
        result = []
        for p in self.professors:
            dist = (p.pos - sprite.pos).l1norm()
            if dist < level.tilesize // 2:
                result.append(p)

        return result

    def draw(self, image: np.ndarray, show_goals: bool):
        for p in self.professors:
            p.sprite.draw_clipped(image)

            if show_goals:
                r = p.goal.y - 1
                c = p.goal.x - 1
                match p.kind:
                    case SpriteKind.PROF_BLINK:
                        color = (255, 0, 0)
                    case SpriteKind.PROF_PINK:
                        color = (255, 0, 255)
                    case SpriteKind.PROF_INK:
                        color = (0, 255, 255)
                    case SpriteKind.PROF_SUE:
                        color = (135, 118, 212)

                image[r:r + 3, c:c + 3, :3] = color
