# REFs:
# https://docs.ray.io/en/latest/rllib/rllib-env.html
# https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5

# RL imports
import gymnasium
import ray
from ray.rllib.algorithms import ppo

# Garbage collector
import gc

# Numbers/Array Imports
import numpy as np

# Tensorflow/Keras imports
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
tf.compat.v1.enable_eager_execution()

class CloningEnv_v0(gymnasium.Env):
    # Maximum number of steps
    MAX_STEPS = 100

    # Possible Rewards
    REWARD_INCORRECT = -2
    REWARD_CORRECT = 1

    # Optional metadata dictionary
    metadata = {
        "render.modes": ["human"]
    }

    REF_MODEL = load_model(os.path.join(os.getcwd(),"models","model_mod_class_cp.h5"))

    def __init__(self, env_config):
        # rf_data
        self.rf_data = env_config['rf_data']
        # Get expert predictions during initialization
        # Define action space where each possible action is a classificaiton type
        self.action_space = gymnasium.spaces.Discrete(env_config['num_classes'])
        # Observation space is the model's estimated classification
        self.observation_space = gymnasium.spaces.Box(
            -np.inf, np.inf,self.rf_data[0].shape, dtype=np.float32
            )
        if env_config['max_steps'] is not None:
            self.MAX_STEPS = env_config['max_steps']

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment's state for new episodes; Return initial observation
        """
        # super().reset from FlappyBirdEnv
        super().reset(seed=seed)
        self.count = 0
        # self.state = None
        self.reward = 0
        self.done = False
        self.info = {
            "correct": 0,
            "incorrect": 0,
            "accuracy": 0.0,
            "expert_preds": [],
            "agent_preds": [],
        }        
        return self.get_observation(), self.info

    def get_observation(self):
        observation = self.rf_data[self.count,:,:]
        return observation

    def get_expert_prediction(self):
        
        observation = self.get_observation()
        observation = observation.reshape( (1, observation.shape[0], observation.shape[1]) )
        prediction = np.argmax(self.REF_MODEL.predict(observation, verbose=0))
        return prediction

    def step(self, action):
        gc.collect()
        # a "edge case" that isn't expected to be reached
        if self.done:
            print("Episode completed.")
        # At the end of an episode
        elif self.count == self.MAX_STEPS:
            self.done = True
            print(f"Count: {self.count}; Accuracy: {self.info['accuracy']:.2f}; Correct: {self.info['correct']}; Incorrect: {self.info['incorrect']}")
        # Most steps should hit here
        else:
            assert self.action_space.contains(action)
            self.count += 1

            # Logic for determining reward
            # Get expert's prediction
            expert_prediction = self.get_expert_prediction()
            self.info['expert_preds'].append(expert_prediction)
            self.info['agent_preds'].append(action)
            # Reward depends on if agent correctly predicted the expert's prediction
            if action == expert_prediction:
                self.reward = self.REWARD_CORRECT
                self.info["correct"] += 1
            else:
                self.reward = self.REWARD_INCORRECT
                self.info["incorrect"] += 1
            self.info['accuracy'] = self.info['correct']/(self.info['correct'] + self.info['incorrect'])
            
            self.state = 0
        
        observation = self.get_observation()
            

        return observation, self.reward, self.done, False, self.info

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def close(self):
        pass