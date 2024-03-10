from gymnasium.envs.registration import register

register(
    id="cloning-v0",
    entry_point="cloning_env.envs:CloningEnv_v0",
)