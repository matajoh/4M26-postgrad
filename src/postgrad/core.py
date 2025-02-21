"""Shared structures and constants for the game."""

from enum import Enum, auto
from typing import Mapping, NamedTuple, Tuple

from .drawing import GlyphSheet, SpriteSheet, Tilesheet


class SpriteKind(Enum):
    """Different kinds of drawn sprites."""
    PELLET = auto()
    POWER_PELLET = auto()
    CHERRY = auto()
    STRAWBERRY = auto()
    ORANGE = auto()
    PRETZEL = auto()
    APPLE = auto()
    PEAR = auto()
    BANANA = auto()
    GRAD = auto()
    PROF_BLINK = auto()
    PROF_PINK = auto()
    PROF_INK = auto()
    PROF_SUE = auto()

    def add(self, value: int) -> "SpriteKind":
        return SpriteKind(self.value + value)


class ProfessorState(Enum):
    START = auto()
    IN_ROOM = auto()
    SCATTER = auto()
    CHASE = auto()
    FRIGHTENED = auto()
    ASKED = auto()
    END = auto()

    def index(self):
        match self:
            case ProfessorState.ASKED:
                return -3
            case ProfessorState.FRIGHTENED:
                return -2
            case ProfessorState.IN_ROOM:
                return -1
            case _:
                raise ValueError(f"Phase {self} does not have an index")


class GameState(Enum):
    START = auto()
    PAUSED = auto()
    PLAYER_SELECT = auto()
    IN_LEVEL = auto()
    BETWEEN_LEVELS = auto()
    GAME_OVER = auto()
    END = auto()


class PostGradState(Enum):
    START = auto()
    STARTING = auto()
    PLAYING = auto()
    STRESSING = auto()
    CAUGHT = auto()
    WINNING = auto()
    WIN = auto()
    LOSE = auto()


class PhaseTimer(NamedTuple("PhaseTimer", [("index", int),
                                           ("phase", ProfessorState),
                                           ("ticks", int)])):
    """A timer for use in the Professor phase cycle."""

    def tick(self) -> "PhaseTimer":
        if self.ticks > 0:
            return self._replace(ticks=self.ticks - 1)

        return self

    def is_done(self) -> bool:
        return self.ticks == 0

    def complete(self) -> "PhaseTimer":
        return self._replace(ticks=0)


class GameConfig(NamedTuple("GameConfig", [("points", Mapping[SpriteKind, int]),
                                           ("timings", Mapping[PostGradState, int]),
                                           ("phases", Tuple[PhaseTimer]),
                                           ("prof_pellets", Mapping[SpriteKind, int]),
                                           ("glyphsheet", GlyphSheet),
                                           ("spritesheet", SpriteSheet),
                                           ("tilesheet", Tilesheet),
                                           ("start_tries", int),
                                           ("max_tries", int),
                                           ("points_per_try", int),
                                           ("invulnerable", bool),
                                           ("show_goals", bool),
                                           ("show_state", bool),
                                           ("level_rows", int),
                                           ("level_columns", int),
                                           ("level_max_wall_length", int),
                                           ("frame_size", Tuple[int, int]),
                                           ("grad", int)])):
    """Configuration for the game.

    Configuration classes like this are a very useful design pattern for
    simulations of all kinds, games included. By providing a single place
    where all the configuration is stored, it makes it easy to experiment
    with different values to tweak the game. Further, it avoids the issue
    of having magic constants scattered throughout the codebase.    
    """
    @staticmethod
    def create(points=None,
               timings=None,
               phases=None,
               prof_pellets=None,
               glyphsheet: str = None,
               spritesheet: str = None,
               tilesheet: str = None,
               start_tries=3,
               max_tries=5,
               points_per_try=10000,
               invulnerable=False,
               show_goals=False,
               show_state=False,
               level_rows=21,
               level_columns=19,
               level_max_wall_length=7,
               frame_size=(400, 400),
               grad=0) -> "GameConfig":
        if points is None:
            points = {
                SpriteKind.PELLET: 10,
                SpriteKind.POWER_PELLET: 50,
                SpriteKind.CHERRY: 100,
                SpriteKind.STRAWBERRY: 200,
                SpriteKind.ORANGE: 500,
                SpriteKind.PRETZEL: 700,
                SpriteKind.APPLE: 1000,
                SpriteKind.PEAR: 2000,
                SpriteKind.BANANA: 5000,
                SpriteKind.PROF_BLINK: 200,  # these are not specific
                SpriteKind.PROF_PINK: 400,  # but instead represent the scaling
                SpriteKind.PROF_INK: 800,  # points for each professor
                SpriteKind.PROF_SUE: 1600}

        if timings is None:
            timings = {
                PostGradState.STARTING: 2 * 60,
                PostGradState.PLAYING: -1,
                PostGradState.STRESSING: 1 * 60,
                PostGradState.CAUGHT: 2 * 60,
                PostGradState.WINNING: 4 * 60,
                PostGradState.WIN: -1,
                PostGradState.LOSE: -1}

        if phases is None:
            phases = (
                PhaseTimer(0, ProfessorState.SCATTER, 7 * 60),
                PhaseTimer(1, ProfessorState.CHASE, 20 * 60),
                PhaseTimer(2, ProfessorState.SCATTER, 7 * 60),
                PhaseTimer(3, ProfessorState.CHASE, 20 * 60),
                PhaseTimer(4, ProfessorState.SCATTER, 5 * 60),
                PhaseTimer(5, ProfessorState.CHASE, 20 * 60),
                PhaseTimer(6, ProfessorState.SCATTER, 5 * 60),
                PhaseTimer(7, ProfessorState.CHASE, -1),
                PhaseTimer(9, ProfessorState.ASKED, -1),
                PhaseTimer(11, ProfessorState.FRIGHTENED, 10 * 60),
                PhaseTimer(13, ProfessorState.IN_ROOM, -1)
            )

        assert phases[ProfessorState.ASKED.index()].phase == ProfessorState.ASKED
        assert phases[ProfessorState.FRIGHTENED.index()].phase == ProfessorState.FRIGHTENED
        assert phases[ProfessorState.IN_ROOM.index()].phase == ProfessorState.IN_ROOM

        if prof_pellets is None:
            prof_pellets = {
                SpriteKind.PROF_BLINK: 0,
                SpriteKind.PROF_PINK: 0,
                SpriteKind.PROF_INK: 33,
                SpriteKind.PROF_SUE: 100}

        glyphsheet = GlyphSheet.load(glyphsheet)
        spritesheet = SpriteSheet.load(spritesheet)
        tilesheet = Tilesheet.load(tilesheet)

        return GameConfig(points, timings, phases, prof_pellets,
                          glyphsheet, spritesheet, tilesheet,
                          start_tries, max_tries, points_per_try,
                          invulnerable, show_goals, show_state,
                          level_rows, level_columns, level_max_wall_length,
                          frame_size, grad)
