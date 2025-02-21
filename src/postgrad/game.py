"""Module providing the game loop for the PostGrad game."""

from datetime import datetime
import os
from typing import Tuple
import random

import numpy as np
import pygame
import pygame.surfarray as psa

from .core import GameConfig, GameState, PostGradState
from .postgrad import Action, PostGrad
from .screens import GameOverScreen, InfoScreen, PauseScreen, SelectScreen


class Game:
    """This class provides the game loop for the PostGrad game.
    
    Note how the state transitions are handled.
    """
    def __init__(self, size: Tuple[int, int],
                 fullscreen: bool, config: GameConfig,
                 replay_path: str = None, max_replay_length=20*60*60,
                 seed=None):
        self.size = size
        self.fullscreen = fullscreen
        self.config = config
        self.state = GameState.START
        frame_size = config.frame_size
        assert frame_size[0] % config.tilesheet.tile_size == 0
        assert frame_size[1] % config.tilesheet.tile_size == 0
        self.frame = np.zeros(frame_size + (3,), dtype=np.uint8)
        self.frame_surface = pygame.Surface(frame_size)
        self.replay_path = replay_path
        self.max_replay_length = max_replay_length
        if replay_path is not None:
            os.makedirs(replay_path, exist_ok=True)

        random.seed(seed)

        self.seed = None
        self.grad = None
        self.actions = None

        pygame.display.init()
        if fullscreen:
            self.screen = pygame.display.set_mode(size, pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(size)

        self.render_size = size[1], size[1]
        self.render_surface = pygame.Surface(self.render_size)
        self.render_pos = (size[0] - size[1]) // 2, 0
        self.clock = pygame.time.Clock()
        self.current_action = Action.NONE
        self.queued_action = Action.NONE
        self.game_over_screen: GameOverScreen = GameOverScreen.create(config.glyphsheet)
        self.pause_screen: PauseScreen = PauseScreen.create(config.glyphsheet)
        self.select_screen: SelectScreen = SelectScreen.create(config.tilesheet, config.frame_size, config.grad)
        self.info_screen: InfoScreen = InfoScreen.create(config.tilesheet, config.frame_size)
        self.paused_state: GameState = GameState.START
        self.game = None

    @staticmethod
    def key_to_action() -> Action:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            return Action.MOVE_UP

        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            return Action.MOVE_DOWN

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            return Action.MOVE_LEFT

        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            return Action.MOVE_RIGHT

        return Action.NONE

    def save_replay(self):
        if self.replay_path is None:
            return

        name = datetime.now().strftime("replay_%Y%m%d%H%M%S.npz")
        path = os.path.join(self.replay_path, name)
        actions = np.array(self.actions, dtype=np.int32)
        np.savez(path, seed=self.seed, grad=self.grad, actions=actions)

    def clear_actions(self):
        self.current_action = Action.NONE
        self.queued_action = Action.NONE

    def start(self):
        """Start the game."""
        # each game has a random seed. Setting it this way allows us
        # to reproduce the same game by using the same seed.
        self.seed = random.randint(0, 2**32 - 1)
        random.seed(self.seed)
        self.actions = []
        self.game = PostGrad.create(self.config)
        self.paused_state = None
        self.state = GameState.PLAYER_SELECT
        self.clear_actions()

    def player_select(self, action: Action):
        """Player select screen."""
        if action != self.current_action:
            self.current_action = action
        else:
            action = Action.NONE

        self.select_screen = self.select_screen.step(action)
        self.select_screen.draw(self.frame)
        if self.select_screen.ready:
            self.grad = self.select_screen.selected
            self.game = self.game.start_game(self.select_screen.selected)
            self.state = GameState.IN_LEVEL
            self.clear_actions()

    def in_level(self, action: Action):
        """The player is playing the game."""
        pos = self.game.grad.pos
        if action != Action.NONE:
            if action == self.current_action.opposite():
                self.current_action = action
            else:
                # preparing for a turn
                self.queued_action = action

        if self.game.level.snap_to_grid(pos) == pos and self.queued_action != Action.NONE:
            # we have reached a potential junction
            # apply the queued action
            self.current_action = self.queued_action
            self.queued_action = Action.NONE

        if self.replay_path is not None:
            self.actions.append((self.game.num_steps, self.current_action.value))
            if len(self.actions) == self.max_replay_length:
                print("Maximum replay length reached")
                self.state = GameState.GAME_OVER
                self.save_replay()
                return

        self.game = self.game.step(self.current_action)
        self.game.draw(self.frame)
        match self.game.state:
            case PostGradState.WIN:
                self.state = GameState.BETWEEN_LEVELS
            case PostGradState.LOSE:
                self.state = GameState.GAME_OVER
                self.save_replay()

    def between_levels(self, action: Action):
        """Between levels screen."""
        self.info_screen.draw(self.frame)
        if action != Action.NONE:
            self.game = self.game.next_level()
            self.state = GameState.IN_LEVEL
            self.clear_actions()

    def game_over(self):
        """Game over screen."""
        keys = pygame.key.get_pressed()
        self.game_over_screen.draw(self.frame)
        if keys[pygame.K_y]:
            self.state = GameState.START
        elif keys[pygame.K_n]:
            self.state = GameState.END

    def paused(self):
        match self.paused_state:
            case GameState.PLAYER_SELECT:
                self.select_screen.draw(self.frame)
            case GameState.IN_LEVEL:
                self.game.draw(self.frame)
            case GameState.BETWEEN_LEVELS:
                self.info_screen.draw(self.frame)
            case GameState.GAME_OVER:
                self.game_over_screen.draw(self.frame)

        self.pause_screen.draw(self.frame)

    def run(self):
        while self.state != GameState.END:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.state = GameState.END
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.state = GameState.END
                    elif event.key == pygame.K_SPACE:
                        if self.state == GameState.PAUSED:
                            self.state = self.paused_state
                        else:
                            self.paused_state = self.state
                            self.state = GameState.PAUSED

            self.frame[:] = 0
            action = Game.key_to_action()

            match self.state:
                case GameState.START:
                    self.start()
                case GameState.PLAYER_SELECT:
                    self.player_select(action)
                case GameState.IN_LEVEL:
                    self.in_level(action)
                case GameState.BETWEEN_LEVELS:
                    self.between_levels(action)
                case GameState.GAME_OVER:
                    self.game_over()
                case GameState.PAUSED:
                    self.paused()

            frame = np.swapaxes(self.frame, 0, 1)
            self.screen.fill("black")
            psa.blit_array(self.frame_surface, frame)
            pygame.transform.scale(self.frame_surface, self.render_size, self.render_surface)
            self.screen.blit(self.render_surface, self.render_pos)
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
