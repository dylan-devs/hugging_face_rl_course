from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
import gymnasium as gym

def load_model():
    '''
    Load PPO model trained on the Lunar Lander V3 environment
    '''
    repo_id = "dylangirrens/PPO-LunarLander-V3" # The repo_id
    filename = "ppo-LunarLander-v3.zip" # The model filename.zip
    checkpoint = load_from_hub(repo_id, filename)
    model = PPO.load(checkpoint, print_system_info=True)
    return model

def create_environment():
    '''
    Create a Lunar Lander Version 3 environment
    '''
    env = gym.make("LunarLander-v3", render_mode="human")
    return env

def test(model, env):
    '''
    Visualize how the performs in the evironmnent 
    '''
    for _ in range(3):
        # S_0
        obs, info = env.reset()

        while True:
            # A_t
            action, _state = model.predict(obs, deterministic=True)
            # S_t, R_t
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            #print(info)
            env.render()

        env.reset()
        
    env.close()



if __name__ == "__main__":
    model = load_model()
    env = create_environment()
    test(model, env)