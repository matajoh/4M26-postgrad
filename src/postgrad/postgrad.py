from typing import Mapping, NamedTuple, Tuple

import numpy as np

from .core import GameConfig, PostGradState, SpriteKind
from .drawing import blend_into_safe, read_rgba
from .faculty import Faculty, ProfessorState
from .geometry import Direction, Vec2
from .hud import HUD
from .level import Level
from .level_generator import generate_level
from .sprite import Action, Sprite


class Score(NamedTuple("Score", [("pos", Vec2), ("value", int), ("ticks", int), ("pixels", Mapping[int, np.ndarray])])):
    def draw(self, image: np.ndarray):
        if self.ticks <= 0:
            return

        blend_into_safe(image, self.pos, self.pixels[self.value])

    def step(self) -> "Score":
        if self.ticks <= 0:
            return self

        return self._replace(ticks=self.ticks - 1)

    def update(self, value: int, pos: Vec2, ticks: int) -> "Score":
        return Score(pos, value, ticks, self.pixels)


class PostGrad(NamedTuple("Game", [("level", Level),
                                   ("state", PostGradState),
                                   ("ticks", int),
                                   ("num_steps", int),
                                   ("grad", Sprite),
                                   ("faculty", Faculty),
                                   ("fruit", Sprite),
                                   ("fruit_eaten", int),
                                   ("faculty_eaten", int),
                                   ("config", GameConfig),
                                   ("hud", HUD),
                                   ("score", Score),
                                   ("power_pellet", Sprite)])):
    def play_fruit(self) -> "PostGrad":
        fruit = self.fruit
        if self.level.pellets_eaten >= 64 and self.fruit_eaten == 0:
            fruit = fruit.teleport(self.level.tile_to_vec(self.level.item))
        elif self.level.pellets_eaten >= 176 and self.fruit_eaten == 1:
            fruit = fruit.teleport(self.level.tile_to_vec(self.level.item))

        if fruit.bb.intersects(self.grad.bb):
            points = self.config.points[fruit.kind]
            return self._replace(
                hud=self.hud.add_points(points),
                score=self.score.update(points, fruit.pos, 60),
                fruit=fruit.teleport(Vec2(-100, -100)),
                fruit_eaten=self.fruit_eaten + 1)

        return self._replace(fruit=fruit)

    def play_faculty(self, energized: bool) -> "PostGrad":
        faculty = self.faculty.step(self.level, self.grad, energized)
        if energized:
            faculty_eaten = 0
        else:
            faculty_eaten = self.faculty_eaten

        hud = self.hud
        score = self.score
        for p in self.faculty.intersect(self.level, self.grad):
            if p.phase == ProfessorState.FRIGHTENED:
                faculty = faculty.ask(p)
                points = self.config.points[SpriteKind.PROF_BLINK.add(faculty_eaten)]
                faculty_eaten += 1
                hud = hud.add_points(points)
                score = score.update(points, p.pos, 60)
            elif not self.config.invulnerable and p.phase != ProfessorState.ASKED:
                return self._replace(faculty_eaten=0).to_state(PostGradState.STRESSING)

        return self._replace(faculty=faculty, faculty_eaten=faculty_eaten,
                             hud=hud, score=score)

    def play(self, action: Action) -> "PostGrad":
        grad = self.level.adjust(self.grad.step(action))
        grad = self.level.check_warp(grad)
        level = self.level.eat_pellets(grad)
        pellets_eaten = level.pellets_eaten - self.level.pellets_eaten
        power_pellets_eaten = level.power_pellets_eaten - self.level.power_pellets_eaten
        points = self.config.points[SpriteKind.PELLET] * pellets_eaten
        points += self.config.points[SpriteKind.POWER_PELLET] * power_pellets_eaten
        hud = self.hud.add_points(points)
        game = self._replace(grad=grad, level=level, hud=hud)
        energized = power_pellets_eaten

        game = game.play_fruit()
        game = game.play_faculty(energized)
        return game

    def tick(self) -> "PostGrad":
        if self.ticks == 0:
            return self

        if self.config.show_state and self.ticks % 60 == 0:
            print(self.state, self.ticks // 60)

        return self._replace(ticks=self.ticks - 1)

    def animate(self) -> "PostGrad":
        return self._replace(score=self.score.step(),
                             power_pellet=self.power_pellet.tick())

    @property
    def player_score(self) -> int:
        return self.hud.scoreboard.score

    def transition(self) -> "PostGrad":
        match self.state:
            case PostGradState.STARTING:
                return self.to_state(PostGradState.PLAYING)
            case PostGradState.STRESSING:
                return self.to_state(PostGradState.CAUGHT)
            case PostGradState.CAUGHT:
                hud = self.hud.remove_try()
                if hud.tries == 0:
                    self.hud.lose()
                    return self.to_state(PostGradState.LOSE)
                else:
                    return self._replace(hud=hud).reset_level()
            case PostGradState.WINNING:
                return self.to_state(PostGradState.WIN)
            case _:
                raise ValueError(f"invalid state: {self.state}")

    def step(self, action: Action) -> "PostGrad":
        game = self.animate()
        if game.ticks == 0:
            game = game.transition()
            if self.config.show_state and game.ticks < 0:
                print(game.state, "INF")
        else:
            game = game.tick()
            match game.state:
                case PostGradState.PLAYING:
                    game = game.play(action)
                    if game.level.remaining_pellets == 0:
                        game = game.to_state(PostGradState.WINNING)
                case PostGradState.STRESSING:
                    game = game._replace(grad=game.grad.spin())

        return game._replace(num_steps=self.num_steps + 1)

    def render(self, frame_size: Tuple[int, int]) -> np.ndarray:
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        self.draw(frame)
        return frame

    def draw(self, frame: np.ndarray):
        image = self.level.draw()
        for pp in self.level.power_pellets:
            power_pellet = self.power_pellet.teleport(pp)
            power_pellet.draw(image)

        if self.level.is_inside(self.fruit.pos):
            self.fruit.draw(image)

        self.grad.draw_clipped(image)
        self.faculty.draw(image, self.config.show_goals)
        self.score.draw(image)

        match self.state:
            case PostGradState.STARTING:
                self.config.glyphsheet.blend(image,
                                             self.level.tile_to_vec(self.level.item),
                                             "READY!")
            case PostGradState.WINNING:
                if self.ticks < self.config.timings[PostGradState.WINNING] // 2:
                    # flash the screen
                    image[:, :, :3] += self.ticks
            case PostGradState.CAUGHT:
                # caught animation
                final_ticks = self.level.tilesize // 2
                crop_ticks = final_ticks * 4
                if self.ticks < final_ticks:
                    c0, r0 = self.grad.pos - self.ticks
                    c1, r1 = self.grad.pos + self.ticks
                elif self.ticks < crop_ticks:
                    c0, r0 = self.grad.pos - final_ticks
                    c1, r1 = self.grad.pos + final_ticks
                else:
                    h, w = image.shape[:2]
                    tl = self.grad.pos - final_ticks
                    br = Vec2(w, h) - self.grad.pos + final_ticks
                    total = self.config.timings[PostGradState.CAUGHT] - crop_ticks
                    c0, r0 = (total - self.ticks + crop_ticks) * tl // total
                    br = (self.ticks - crop_ticks) * br // total
                    c1, r1 = self.grad.pos + final_ticks + br

                image[:r0, :] = 0
                image[r1:, :] = 0
                image[:, :c0] = 0
                image[:, c1:] = 0

        self.hud.draw(frame, image)

    def start_game(self, grad: int) -> "PostGrad":
        state = PostGradState.STARTING
        ticks = self.config.timings[state]
        hud = HUD.create(grad, self.config.glyphsheet, self.config.spritesheet,
                         self.config.start_tries, self.config.max_tries, self.config.points_per_try)
        pixels = read_rgba("grads", f"{grad}.png")
        size = pixels.shape[0] // 4
        grad = Sprite.create(SpriteKind.GRAD,
                             self.level.tile_to_vec(self.level.start[0]),
                             size, Direction.UP, pixels)
        return self._replace(hud=hud, fruit=hud.fruit(), num_steps=0,
                             grad=grad, state=state, ticks=ticks)

    def next_level(self) -> "PostGrad":
        level_map = generate_level(self.config.level_rows,
                                   self.config.level_columns,
                                   self.config.level_max_wall_length)
        level = Level.create(self.config.tilesheet, level_map)
        hud = self.hud.next_level()
        return self._replace(
            level=level, state=PostGradState.STARTING,
            ticks=self.config.timings[PostGradState.STARTING],
            grad=self.grad.teleport(level.tile_to_vec(level.start.grad)),
            faculty=Faculty.create(level, self.config), hud=hud,
            fruit_eaten=0, faculty_eaten=0, fruit=hud.fruit())

    def reset_level(self) -> "PostGrad":
        pos = self.level.tile_to_vec(self.level.start.grad)
        return self.to_state(PostGradState.STARTING)._replace(
            grad=self.grad.teleport(pos),
            faculty=Faculty.create(self.level, self.config))

    def to_state(self, state: PostGradState) -> "PostGrad":
        return self._replace(state=state,
                             ticks=self.config.timings[state])

    @staticmethod
    def create(config: GameConfig = None) -> "PostGrad":
        if config is None:
            config = GameConfig.create()

        level = Level.create(config.tilesheet)
        faculty = Faculty.create(level, config)
        score_sprites = {v: config.spritesheet.read_sprite("score", str(v))
                         for k, v in config.points.items()
                         if k not in {SpriteKind.PELLET, SpriteKind.POWER_PELLET}}
        score = Score(Vec2(0, 0), 0, 0, score_sprites)
        pixels = read_rgba("power_pellet.png")
        size = pixels.shape[0]
        power_pellet = Sprite.create(SpriteKind.POWER_PELLET, Vec2(0, 0),
                                     size, Direction.RIGHT, pixels).with_velocity(0, 1)
        return PostGrad(level, PostGradState.START, -1, -1, None, faculty,
                        None, 0, 0, config, None, score, power_pellet)
