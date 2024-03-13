import os
from typing import Dict, Any
import string
import imageio
import glob
import random
import numpy as np
from datetime import datetime, timezone, timedelta
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

class GetPredictionsCallback(DefaultCallbacks):
    """Saves the end of episode "score" as a custom metric"""
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index = None, **kwargs) -> None:
        info = episode.last_info_for()
        episode.custom_metrics['expert_preds'] = info['expert_preds']
        episode.custom_metrics['agent_preds'] = info['agent_preds']