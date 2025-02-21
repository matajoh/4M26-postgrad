"""Module providing replay capabilities.

NB out of scope for Tripos.
"""

from collections import deque
import random

import cv2
import numpy as np

from .core import GameConfig, PostGradState
from .postgrad import PostGrad
from .sprite import Action
from .video_writer import VideoWriter


class Replay:
    def __init__(self, replay_path: str, video_path: str = None, invulerable: bool = False):
        replay = np.load(replay_path)
        self.grad = int(replay["grad"])
        self.seed = int(replay["seed"])
        config = GameConfig.create()
        self.frame = np.zeros(config.frame_size + (3,), dtype=np.uint8)
        self.game: PostGrad = PostGrad.create(config._replace(invulnerable=invulerable))
        self.actions = deque()
        if video_path is not None:
            width, height = config.frame_size
            self.frame_scale = np.full((2, 2, 1), 1, np.uint8)
            self.video_writer = VideoWriter(video_path, (width * 2, height * 2), framerate=60, quality=17)
        else:
            self.video_writer = None

        actions = replay["actions"]
        for s, a in actions:
            self.actions.append((s, Action(int(a))))

    def play(self):
        random.seed(self.seed)
        game = self.game.start_game(self.grad)
        if self.video_writer is not None:
            self.video_writer.start()

        while game.state != PostGradState.LOSE and self.actions:
            if game.state == PostGradState.WIN:
                game = game.next_level()

            s, a = self.actions[0]
            if s == game.num_steps:
                self.actions.popleft()
                game = game.step(a)
            else:
                game = game.step(Action.NONE)

            self.frame[:] = 0
            game.draw(self.frame)
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Replay", self.frame)
            if self.video_writer is not None:
                self.video_writer.frame[:] = np.kron(self.frame, self.frame_scale)
                self.video_writer.write_frame()

            cv2.waitKey(1)

        if self.video_writer is not None:
            self.video_writer.stop()

        cv2.destroyAllWindows()

        print("Final score", game.player_score)
