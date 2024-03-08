# REFs:
# https://docs.ray.io/en/latest/rllib/rllib-env.html
# https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5

# RL imports
import gymnasium as gym
import ray
from ray.rllib.algorithms import ppo

# Numbers/Array Imports
import numpy as np

# Tensorflow/Keras imports
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

class MyEnv(gym.Env):
    # Possible Classifications
    CLASS_AM = 0
    CLASS_FM = 1

    # Total number of possible classifications
    CLASS_COUNT = 2 

    # Maximum number of steps
    MAX_STEPS = 100

    # Possible Rewards
    REWARD_INCORRECT = -2
    REWARD_CORRECT = 1

    # Optional metadata dictionary
    metadata = {
        "render.modes": ["terminal"]
    }

    # Reference model to help train agent via predictions
    REF_MODEL = load_model(os.path.join(os.getcwd(),"models","model_mod_class_val_9776_ds999.h5"))

    def __init__(self, env_config):
        # Define action space where each possible action is a classificaiton type
        self.action_space = gym.spaces.Discrete(CLASS_COUNT)
        # Observation space is the model's estimated classification
        self.observation_space = gym.spaces.Discrete(1)

    def reset(self, seed, options):
        """
        Reset the environment's state for new episodes; Return initial observation
        """
        self.count = 0
        self.state = None
        self.reward = 0
        self.done = False
        self.info = {
            "correct": 0,
            "incorrect": 0,
        }
        return self.state

    def step(self, action, rf_signal):
        expert_prediction = None
        # a "edge case" that isn't expected to be reached
        if self.done:
            print("Episode completed.")
        # At the end of an episode
        elif self.count == self.MAX_STEPS:
            self.done = True
        # Most steps should hit here
        else:
            assert self.action_space.contains(action)
            self.count += 1

            # Logic for determining reward
            # Get expert's prediction
            expert_prediction = REF_MODEL.predict(rf_signal)
            # Reward depends on if agent correctly predicted the expert's prediction
            if action == expert_prediction[0]:
                self.reward = REWARD_CORRECT
                self.info["correct"] += 1
            else:
                self.reward = REWARD_INCORRECT
                self.info["incorrect"] += 1
            
            self.state = 0
            

        return [expert_prediction, self.reward, self.done, self.info]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass


ray.init()
algo = ppo.PPO(env=MyEnv, config={
    "env_config": {},  # config to pass to env class
})

while True:
    print(algo.train())