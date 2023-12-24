from stable_baselines3 import PPO 
from typing import Literal

def load_model(game_mode: Literal["basic", "defend_the_center", "deadly_corridor"]):
    if game_mode == "basic":
        return PPO.load(r"train\train_basic\best_model_100000.zip")
    elif game_mode == "defend_the_center":
        return PPO.load(r"train\train_defend_the_center\best_model_100000.zip")
    elif game_mode == "deadly_corridor":
        return PPO.load(r"train\train_deadly_corridor\best_model_240000.zip")
    else:
        raise ValueError(
            "This game mode is not available. "
            "Use one of the following: 'basic', 'center', 'corridor'"
        )
