"""Module providing sprite logic.

Sprites are an essential component of 2D games, and contain some interesting
design patterns. Much of the logic is graphics related and out of scope,
but students should be familiar with a few key concepts.
"""

from enum import Enum
from typing import NamedTuple, Tuple

import numpy as np

from .core import SpriteKind
from .drawing import blend_into_safe
from .geometry import Vec2, Direction, Rect2, Velocity


class Action(Enum):
    """These are the actions that a sprite can take.

    Note that these actions can come from player input or from the
    faculty AI.
    """
    NONE = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4

    def from_direction(d: Direction) -> "Action":
        match d:
            case Direction.UP:
                return Action.MOVE_UP
            case Direction.DOWN:
                return Action.MOVE_DOWN
            case Direction.LEFT:
                return Action.MOVE_LEFT
            case Direction.RIGHT:
                return Action.MOVE_RIGHT

    def opposite(self) -> "Action":
        match self:
            case Action.MOVE_UP:
                return Action.MOVE_DOWN
            case Action.MOVE_DOWN:
                return Action.MOVE_UP
            case Action.MOVE_LEFT:
                return Action.MOVE_RIGHT
            case Action.MOVE_RIGHT:
                return Action.MOVE_LEFT
            case Action.NONE:
                return Action.NONE


class SpriteImage(NamedTuple("SpriteImage", [("pixels", np.ndarray),
                                             ("size", int),
                                             ("num_frames", int),
                                             ("num_facings", int)])):
    @staticmethod
    def create(pixels: np.ndarray, size: int) -> "SpriteImage":
        assert size > 0, size
        h, w = pixels.shape[:2]
        num_facings = h // size
        num_frames = w // size
        return SpriteImage(pixels, size, num_frames, num_facings)

    def select(self, facing: Direction, frame: int) -> np.ndarray:
        r = facing.value % self.num_facings
        c = frame * self.size
        r0 = r * self.size
        r1 = r0 + self.size
        c0 = c
        c1 = c0 + self.size
        return self.pixels[r0:r1, c0:c1]


class Sprite(NamedTuple("Sprite", [("kind", SpriteKind),
                                   ("pos", Vec2),
                                   ("velocity", Velocity),
                                   ("size", int),
                                   ("facing", Direction),
                                   ("images", Tuple[SpriteImage, ...]),
                                   ("image", int),
                                   ("frame", int)])):
    """Class representing the state of a Sprite.

    Note again that this is a NamedTuple. The reason here is to make
    state transitions on sprites very clear.
    """
    @property
    def bb(self) -> Rect2:
        return Rect2.create(self.pos, self.size, self.size)

    def tick(self) -> "Sprite":
        """Animation ticks."""
        return self.with_frame(self.frame + 1)

    def with_images(self, *images: SpriteImage) -> "Sprite":
        return self._replace(images=list(images))

    def with_frame(self, frame: int) -> "Sprite":
        frame = frame % self.images[self.image].num_frames
        return self._replace(frame=frame)

    def with_image(self, image: int, frame=0) -> "Sprite":
        return self._replace(image=image, frame=frame)

    @property
    def pixels(self) -> np.ndarray:
        return self.images[self.image].select(self.facing, self.frame)

    def draw(self, image: np.ndarray):
        blend_into_safe(image, self.pos, self.pixels)

    def draw_clipped(self, image: np.ndarray):
        rect = Rect2(0, 0, image.shape[1], image.shape[0])

        if self.bb.left < rect.left:
            blend_into_safe(image, self.pos, self.pixels)
            blend_into_safe(image, self.pos + Vec2(rect.width, 0), self.pixels)
        elif self.bb.right > rect.right:
            blend_into_safe(image, self.pos, self.pixels)
            blend_into_safe(image, self.pos - Vec2(rect.width, 0), self.pixels)
        elif self.bb.top < rect.top:
            blend_into_safe(image, self.pos, self.pixels)
            blend_into_safe(image, self.pos + Vec2(0, rect.height), self.pixels)
        elif self.bb.bottom > rect.bottom:
            blend_into_safe(image, self.pos, self.pixels)
            blend_into_safe(image, self.pos - Vec2(0, rect.height), self.pixels)
        else:
            self.draw(image)

    def step(self, action: Action) -> "Sprite":
        """Moves the sprite to the next state."""
        if action == Action.NONE:
            return self

        match action:
            case Action.MOVE_UP:
                d = Direction.UP
            case Action.MOVE_DOWN:
                d = Direction.DOWN
            case Action.MOVE_LEFT:
                d = Direction.LEFT
            case Action.MOVE_RIGHT:
                d = Direction.RIGHT
            case _:
                d = None

        if d:
            # NB note the use of velocity, position, and direction
            pos = self.velocity.move(self.pos, d)
            velocity = self.velocity.tick()
            facing = d
        else:
            # Even though no action has been taken we need to tick
            # the velocity as it works on time, not movement.
            pos = self.pos
            velocity = self.velocity.tick()
            facing = self.facing

        return self._replace(pos=pos, velocity=velocity, facing=facing)

    def with_facing(self, facing: Direction) -> "Sprite":
        return self._replace(facing=facing)

    def spin(self, clockwise=False) -> "Sprite":
        return self.with_facing(self.facing.rotate(clockwise))

    def teleport(self, pos: Vec2) -> "Sprite":
        return self._replace(pos=pos)

    def with_velocity(self, num: int, den: int) -> "Sprite":
        return self._replace(velocity=Velocity.create(num, den))

    def create(kind: SpriteKind, pos: Vec2,
               size: int, facing: Direction, *pixels) -> "Sprite":
        images = []
        for p in pixels:
            images.append(SpriteImage.create(p, size))

        images = tuple(images)
        return Sprite(kind, pos, Velocity.create(1, 1), size, facing, images, 0, 0)
