from gym import Env 
from gym.spaces import Box, Discrete
from vizdoom import DoomGame
import cv2
import numpy as np
from typing import Literal, Optional, Union


class VizDoomEnv(Env): 
    def __init__(self, render=False, game_mode="basic", action_size=3): 
        super().__init__()
        self.game = DoomGame()
        self.game.load_config(f'github/VizDoom/scenarios/{game_mode}.cfg')
        self.action_size = action_size

        if render == False: 
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
        
        self.game.init()
        
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8) 
        self.action_space = Discrete(action_size)
        
    def step(self, action):
        actions = np.identity(self.action_size)
        reward = self.game.make_action(actions[action], 4) 
        
        if self.game.get_state(): 
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = ammo
        else: 
            state = np.zeros(self.observation_space.shape)
            info = 0 
        
        info = {"info":info}
        done = self.game.is_episode_finished()
        
        return state, reward, done, info 
    
    def reset(self): 
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)
    
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100,160,1))
        return state
    
    def close(self): 
        self.game.close()


def get_env(
        game_mode: Literal["basic", "defend_the_center", "deadly_corridor"] = "basic", 
        render: bool = True,
        difficulty_level: Optional[Union[None, Literal["d1", "d2", "d3", "d4", "d5"]]] = None
    ):
    if game_mode == "basic":
        return VizDoomEnv(render=render, game_mode="basic")
    elif game_mode == "defend_the_center":
        return VizDoomEnv(render=render, game_mode="defend_the_center")
    elif game_mode == "deadly_corridor":
        if difficulty_level in ["d1", "d2", "d3", "d4", "d5"]:
            return VizDoomEnv(
                render=render, 
                game_mode="deadly_corridor_" + difficulty_level, 
                action_size=7
            )
        else:
            raise ValueError(
                "If game mode is 'deadly_corridor', "
                "difficulty level should be specified ['d1', 'd2', 'd3', 'd4', 'd5']"
            )
    else:
        raise ValueError(
            "Wrong game mode specified. "
            "game_mode should be on of the following: "
            "['basic', 'defend_the_center', 'deadly_corridor']"
        )
    
    