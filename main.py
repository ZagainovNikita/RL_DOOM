from env import get_env
from model import load_model
import time

def main():
    game_mode = input("Enter the game mode: ")
    if game_mode == "deadly_corridor":
        difficulty_level = input("Enter difficulty level: ")
    else:
        difficulty_level = None
    
    model = load_model(game_mode=game_mode)
    env = get_env(game_mode=game_mode, difficulty_level=difficulty_level, render=True)

    done = False
    state = env.reset()
    while not done:
        action = model.predict(state)[0]
        state, _, done, _ = env.step(action=action)
        time.sleep(0.05)
    env.close()


if __name__ == "__main__":
    main()