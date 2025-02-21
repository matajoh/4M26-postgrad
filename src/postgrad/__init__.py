"""PostGrad video game.

These files, in addition with the README, act as lecture notes and a
revision aid for the Tripos. Some files are not in scope for the Tripos
and will be marked as such in the module comments. Other files will have
commented methods or functions which correspond to those discussed in
lecture, and students should understand those thoroughly.

As a general note, anything to do with graphics is not in scope for the
Tripos. This includes the `drawing` module and all of the sprite logic
including the `Tilesheet`, `SpriteSheet`, and `GlyphSheet` classes.
"""

import argparse
import random

__version__ = "1.0.0"

from .core import GameConfig
from .postgrad import PostGrad
from .level import Level
from .level_generator import generate_level
from .replay import Replay

__all__ = ["Level", "PostGrad"]


try:
    from .game import Game
except ImportError:
    class Game:
        """Dummy class to avoid import errors."""

        def __init__(self, *args, **kwargs):
            pass


def main():
    """Run the game."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--version", action="version", version=__version__)
    parser.add_argument("-f", "--fullscreen", action="store_true", help="Run the game in fullscreen mode")
    parser.add_argument("-i", "--invulnerable", action="store_true", help="Make the grad invulnerable")
    parser.add_argument("-g", "--show-goals", action="store_true", help="Show the goal tiles of the profs")
    parser.add_argument("-s", "--show-state", action="store_true", help="Show the state of the game")
    parser.add_argument("-r", "--replay", help="Path to save the replay")
    parser.add_argument("-l", "--replay-length", default=20*60*60, type=int, help="Maximum length of the replay")
    args = parser.parse_args()
    config = GameConfig.create(invulnerable=args.invulnerable,
                               show_goals=args.show_goals,
                               show_state=args.show_state)
    game = Game((1200, 800), args.fullscreen, config, args.replay, args.replay_length)
    game.run()


def level_main():
    """Generate a map."""
    parser = argparse.ArgumentParser(description="Postgrad level generator")
    parser.add_argument("-v", "--version", action="version", version=__version__)
    parser.add_argument("--rows", default=21, type=int, help="Number of rows in the level")
    parser.add_argument("--columns", default=19, type=int, help="Number of columns in the level")
    parser.add_argument("--max-wall-length", default=7, type=int, help="Maximum length of a wall")
    parser.add_argument("--seed", type=int, help="Random seed")
    args = parser.parse_args()
    random.seed(args.seed)
    print(generate_level(args.rows, args.columns, args.max_wall_length))


def replay_main():
    """Replay a game."""
    parser = argparse.ArgumentParser(description="Replay a game")
    parser.add_argument("replay_path", help="Path to the replay file")
    parser.add_argument("-v", "--video_path", help="Path to the video file")
    parser.add_argument("-i", "--invulnerable", action="store_true",
                        help="Make the grad invulnerable")
    args = parser.parse_args()
    replay = Replay(args.replay_path, args.video_path, args.invulnerable)
    replay.play()
