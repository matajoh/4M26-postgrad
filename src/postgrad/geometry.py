"""Module providing integer geometry classes.

The gameplay is implemented without any floating point values. This allows
the game to be deterministic and reproducible. The classes below give us
the key maths we need without violating this principle.
"""


from enum import Enum
from fractions import Fraction
from typing import List, NamedTuple, Union


class Direction(Enum):
    """The game operates on a 4-way connection system."""
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    def __lt__(self, other: "Direction") -> bool:
        return self.value < other.value

    def opposite(self) -> "Direction":
        return Direction((self.value + 2) % 4)

    def rotate(self, clockwise: bool) -> "Direction":
        return Direction((self.value + (1 if clockwise else -1)) % 4)

    def forward(self) -> List["Direction"]:
        """Returns valid directions for this facing."""
        return [self, self.rotate(True), self.rotate(False)]

    def distance(self, other: "Direction") -> int:
        """Returns the number of turns needed to go between two facings."""
        diff = abs(self.value - other.value)
        return min(diff, 4 - diff)


class Vec2(NamedTuple("Vec2", [("x", int), ("y", int)])):
    """An integer-based 2D vector class.
    
    First, note that this class is a NamedTuple. This means that it is
    immutable, which is a hugely useful property. Since its properties
    are also immutable, we can use this class as a key in dictionaries or
    store it in a set.

    Secondly, note all of these special methods, like `__add__` and `__sub__`.
    These methods indicate to Python that we want to overload various operators,
    like `+` and `-` respectively. This enables our code to read more clearly
    when performing maths with vector objects.
    """
    def __add__(self, other: Union["Vec2", int]) -> "Vec2":
        if isinstance(other, Vec2):
            return Vec2(self.x + other.x, self.y + other.y)

        return Vec2(self.x + other, self.y + other)

    def __radd__(self, other: Union["Vec2", int]) -> "Vec2":
        return self.__add__(other)

    def __sub__(self, other: Union["Vec2", int]) -> "Vec2":
        if isinstance(other, Vec2):
            return Vec2(self.x - other.x, self.y - other.y)

        return Vec2(self.x - other, self.y - other)

    def __mul__(self, other: Union[int, "Vec2"]) -> "Vec2":
        if isinstance(other, Vec2):
            return Vec2(self.x * other.x, self.y * other.y)
        return Vec2(other * self.x, other * self.y)

    def __rmul__(self, scalar: Union[int, "Vec2"]) -> "Vec2":
        return self.__mul__(scalar)

    def __floordiv__(self, scalar: int) -> "Vec2":
        return Vec2(self.x // scalar, self.y // scalar)

    def is_zero(self) -> bool:
        return not (self.x or self.y)

    def l1norm(self) -> int:
        return abs(self.x) + abs(self.y)

    def normalize(self) -> "Vec2":
        norm = self.l1norm()
        return Vec2(self.x // norm, self.y // norm)

    def dot(self, other: "Vec2") -> int:
        return self.x * other.x + self.y * other.y

    def move(self, d: Direction, distance: int) -> "Vec2":
        """This method allows us to easily combine positions and facings."""
        if d == Direction.UP:
            return Vec2(self.x, self.y - distance)
        if d == Direction.RIGHT:
            return Vec2(self.x + distance, self.y)
        if d == Direction.DOWN:
            return Vec2(self.x, self.y + distance)
        if d == Direction.LEFT:
            return Vec2(self.x - distance, self.y)


class Rect2(NamedTuple("Rect2", [("left", int), ("top", int), ("right", int), ("bottom", int)])):
    """This is mostly used by the grahpics code and can be ignored."""
    def create(pos: Vec2, width: int, height: int = None) -> "Rect2":
        if height is None:
            height = width

        return Rect2(pos.x - width // 2, pos.y - height // 2,
                     pos.x + width // 2, pos.y + height // 2)

    @property
    def center(self) -> Vec2:
        return Vec2((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def is_outside(self, point: Vec2) -> bool:
        return point.x < self.left or point.x > self.right or point.y < self.top or point.y > self.bottom

    def is_inside(self, point: Vec2) -> bool:
        return not self.is_outside(point)

    def is_disjoint(self, other: "Rect2") -> bool:
        return self.right < other.left or self.left > other.right or self.bottom < other.top or self.top > other.bottom

    def intersects(self, other: "Rect2") -> bool:
        return not self.is_disjoint(other)

    def contains(self, other: "Rect2") -> bool:
        return (self.left <= other.left and self.right >= other.right
                and self.top <= other.top and self.bottom >= other.bottom)

    def place(self, width: int, height: int) -> "Rect2":
        left = self.left + (self.width - width) // 2
        top = self.top + (self.height - height) // 2
        right = left + width
        bottom = top + height
        return Rect2(left, top, right, bottom)

    @property
    def half_size(self) -> Vec2:
        return Vec2((self.right - self.left) // 2, (self.top - self.bottom) // 2)


class Velocity(NamedTuple("Velocity", [("count", int), ("adjust", int), ("step", int), ("speed", int)])):
    """This class provides a way to set fractional rates of change.
    
    The main problem with an integer-based system of movement is that it
    becomes difficult to make it such that, for example, the grad moves
    1.1 times as fast as a professor. However, as long as long as we are
    dealing with rational numbers, we can use this class to occassionally
    move a little bit further than normal, or a little bit less than normal.
    """
    def tick(self) -> "Velocity":
        step = (self.step + 1) % self.count
        if self.adjust > 0 and step < self.adjust:
            # we need to move a bit more every so often
            speed = 2
        elif self.adjust < 0 and step < -self.adjust:
            # we need to move a bit less every so often
            speed = 0
        else:
            speed = 1

        return self._replace(step=step, speed=speed)

    def move(self, pos: Vec2, facing: Direction) -> Vec2:
        if self.speed:
            pos = pos.move(facing, self.speed)
            while pos.x % self.speed or pos.y % self.speed:
                # this largely occurs when the speed has been changed
                # mid movement. Various checks elsewhere require the
                # sprite position to be a multiple of its speed.
                pos = pos.move(facing, -1)

        return pos

    @staticmethod
    def create(numerator: int, denominator: int) -> "Velocity":
        f = Fraction(numerator, denominator)
        numerator = f.numerator
        denominator = f.denominator
        if numerator > denominator:
            # we need the numerator and denominator to be
            # multiples of two when moving fractionally faster
            numerator *= 2
            denominator *= 2

        adjust = numerator - denominator
        assert f <= 2
        return Velocity(denominator, adjust, 0, 1)
