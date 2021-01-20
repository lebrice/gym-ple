import os
import gym
from gym import spaces
from ple import PLE
import numpy as np
import cv2

class PLEEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name='FlappyBird', display_screen=True,ple_game=True, root_game_name=None, reward_type='sparse', obs_type=None, **kwargs):
        # set headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        # open up a game state to communicate with emulator
        import importlib
        if ple_game:
            game_module_name = ('ple.games.%s' % game_name).lower()
        else:
            game_module_name = F"{root_game_name.lower()}.envs"
        game_module = importlib.import_module(game_module_name)
        game = getattr(game_module, game_name)(**kwargs)
        self.ple_wrapper = PLE(game, fps=30, display_screen=display_screen)
        self.ple_wrapper.init()
        game.reward_type = reward_type
        self._action_set = self.ple_wrapper.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.screen_height, self.screen_width = self.ple_wrapper.getScreenDims()
        # Assume observation space to be (64, 64, 3) due to procgen
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype = np.uint8)
        self.viewer = None
        assert obs_type is not None, obs_type
        self.obs_type = obs_type

        self.reward_range = game.rewards['win']


    def step(self, a):
        reward = self.ple_wrapper.act(self._action_set[a])
        if self.obs_type == 'state':
            state = self.ple_wrapper.game.get_state()
        elif self.obs_type == 'image':
            state = self._get_image()
        terminal = self.ple_wrapper.game_over()
        return state, reward, terminal, {}

    def _get_image(self):
        image_rotated = np.fliplr(np.rot90(self.ple_wrapper.getScreenRGB(), 3)) # Hack to fix the rotated image returned by ple
        return cv2.resize(image_rotated, (64,64), interpolation = cv2.INTER_AREA)

    @property
    def _n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def reset(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3), dtype = np.uint8)
        self.ple_wrapper.reset_game()
        if self.obs_type == 'state':
            state = self.ple_wrapper.game.get_state()
        elif self.obs_type == 'image':
            state = self._get_image()
        return state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed):
        rng = np.random.RandomState(seed)
        self.ple_wrapper.rng = rng
        self.ple_wrapper.game.rng = self.ple_wrapper.rng

        self.ple_wrapper.init()

    def get_keys_to_action(self):
        return {(): 0, (32,): 1, (119,): 2, (100,): 3, (97,): 4, (115,): 5, (100, 119): 6, (97, 119): 7, (100, 115): 8, (97, 115): 9, (32, 119): 10, (32, 100): 11, (32, 97): 12, (32, 115): 13, (32, 100, 119): 14, (32, 97, 119): 15, (32, 100, 115): 16, (32, 97, 115): 17}
