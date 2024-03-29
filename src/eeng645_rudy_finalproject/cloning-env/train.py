import os
import ray

from ray import tune, train
from ray.tune import register_env
from cloning_env.envs.cloning_env import CloningEnv_v0
from ray.rllib.algorithms.ppo import PPOConfig

import shutil

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from data_management import load_train_test_subset, trim_dataset_by_index, get_class_labels_normal, get_class_labels_strs

############################
##### Data Acquisition #####
############################

def get_data2(class_labels_keep=[3,8,2], num_examples=100, test_agent=False):
    SEED = 1
    np.random.seed(SEED)
    VALIDATION_SPLIT = 0.25
    # Snippet from "run_me.py" for part 3 data
    # "main" root of all data packages used in project
    data_storage_dir = os.path.join(os.getcwd(),'data','project')
    data_package = "snr10_keep100_test10_seed1"
    data_split_dir = os.path.join(data_storage_dir,data_package)
    pt3_storage_dir = os.path.join(data_split_dir,"pt3_data")
    try:
        signals_train_pt3, labels_int_train_pt3, snrs_train_pt3, signals_test_pt3, labels_int_test_pt3, snrs_test_pt3 = load_train_test_subset(pt3_storage_dir)
    except:
        print(f"Error loading in data for part 3 from {pt3_storage_dir}")

    
    indices = np.arange(len(snrs_train_pt3))
    indices_train_pt3, indices_val_pt3, labels_int_train_pt3, labels_int_val_pt3 = train_test_split(indices, labels_int_train_pt3, random_state=SEED, test_size=VALIDATION_SPLIT)
    signals_val_pt3 = signals_train_pt3[indices_val_pt3,:,:]
    signals_train_pt3 = signals_train_pt3[indices_train_pt3,:,:]

    snrs_val_pt3 = snrs_train_pt3[indices_val_pt3]
    snrs_train_pt3 = snrs_train_pt3[indices_train_pt3]

    if test_agent:
        signals_eval_pt3 = signals_test_pt3
        labels_int_eval_pt3 = labels_int_test_pt3
        snrs_eval_pt3 = snrs_test_pt3
    else:
        signals_eval_pt3 = signals_val_pt3
        labels_int_eval_pt3 = labels_int_val_pt3
        snrs_eval_pt3 = snrs_val_pt3
    
    signals_train_pt3_trimmed = np.array([])
    signals_train_pt3_trimmed = signals_train_pt3_trimmed.reshape((signals_train_pt3_trimmed.shape[0],1024,2))
    labels_train_pt3_trimmed = np.array([])

    signals_eval_pt3_trimmed = np.array([])
    signals_eval_pt3_trimmed = signals_eval_pt3_trimmed.reshape((signals_eval_pt3_trimmed.shape[0],1024,2))
    labels_eval_pt3_trimmed = np.array([])

    for class_label in class_labels_keep:
        signals, labels, _ = trim_dataset_by_index(signals_train_pt3, labels_int_train_pt3, snrs_train_pt3, [class_label])
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        indices = indices[0:num_examples]
        signals_train_pt3_trimmed = np.concatenate((signals_train_pt3_trimmed,signals[indices]),axis=0)
        labels_train_pt3_trimmed = np.concatenate((labels_train_pt3_trimmed,labels[indices]),axis=0)
        
        signals, labels, _ = trim_dataset_by_index(signals_eval_pt3, labels_int_eval_pt3, snrs_eval_pt3, [class_label])
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        indices = indices[0:num_examples]
        signals_eval_pt3_trimmed = np.concatenate((signals_eval_pt3_trimmed,signals[indices]),axis=0)
        labels_eval_pt3_trimmed = np.concatenate((labels_eval_pt3_trimmed,labels[indices]),axis=0)

    indices = np.arange(len(labels_train_pt3_trimmed))
    np.random.shuffle(indices)

    signals_train_pt3_trimmed = signals_train_pt3_trimmed[indices,:,:]
    labels_train_pt3_trimmed = labels_train_pt3_trimmed[indices]

    indices = np.arange(len(labels_eval_pt3_trimmed))
    np.random.shuffle(indices)

    signals_eval_pt3_trimmed = signals_eval_pt3_trimmed[indices,:,:]
    labels_eval_pt3_trimmed = labels_eval_pt3_trimmed[indices]

    return signals_train_pt3_trimmed, labels_train_pt3_trimmed, signals_eval_pt3_trimmed, labels_eval_pt3_trimmed

def main():
    # In this example, keep FM (3), BPSK (8)
    class_labels_keep = [3,8]
    # class_labels_keep = get_class_labels_normal()
    # Use 10 examples of each
    num_examples = 2000
    signals_train_pt3, labels_train_pt3, signals_test_pt3, labels_test_pt3 = get_data2(class_labels_keep, num_examples)
    num_classes = len(class_labels_keep)
    num_classes_max = len(get_class_labels_normal())
    total_examples = num_examples*num_classes


    # Reference model to help train agent via predictions
    REF_MODEL = load_model(os.path.join(os.getcwd(),"models","model_mod_class_cp.h5"))
    expert_predictions = np.argmax(REF_MODEL.predict(signals_train_pt3, verbose=0), axis=1)


    chkpt_root = "chkpoint/"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
    ray_results = f"{os.getcwd()}/ray_results/"
    if not os.path.exists(ray_results):
        os.mkdir(ray_results)

    ray.init(ignore_reinit_error=True)

    environment_string = "cloning-v0"
    register_env(environment_string, CloningEnv_v0)

    local_mode = False
    training_iterations = 50 # max iterations before stopping
    num_cpu = 20
    num_gpus = 0
    num_eval_workers = 1

    driver_cpu = 1 # leave this alone
    # How CPUs are spread
    num_rollout_workers = num_cpu - driver_cpu - num_eval_workers

    env_config = {
        'render_mode': 'human',
        'rf_data': signals_train_pt3,
        'expert_preds': expert_predictions,
        'num_classes': num_classes_max, # Use the 11 classes to define action space to avoid needing custom logic
        'max_steps': signals_train_pt3.shape[0]
    }


    config = (  # 1. Configure the algorithm,
        PPOConfig() # put the actual config object for the algorithm you intend to use (Such as PPO or DQN)
        .environment(environment_string, env_config=env_config)
        .experimental(_enable_new_api_stack=False)
        .rollouts(
            num_rollout_workers=num_rollout_workers,
            batch_mode='complete_episodes',
            enable_connectors=False
            )
        .resources(num_gpus=num_gpus)
        .framework("tf2", eager_tracing=True)
        .training(
            # Put hyperparams here as needed. Look in the AlgorithmConfig object and child object for available params
            # REF: https://docs.ray.io/en/master/rllib/rllib-algorithms.html#ppo
            model={"fcnet_hiddens":[512, 512, 512, 512],},
            lr=0.0001, # learning rate
            gamma=0.95, # "Discount factor of Markov Decision process"
            kl_coeff=0.0, # Initial coefficient for Kullback-Leibler divergence, penalizes new policies for beeing too different from previous policy
            train_batch_size=128,
            )
        .evaluation(evaluation_num_workers=num_eval_workers, evaluation_interval=10)
        # .callbacks(callbacks)
        .reporting(keep_per_episode_custom_metrics=True) # decides whether custom metrics in Tensorboard are per episode or mean/min/max
    )

    tuner = tune.Tuner(
        "PPO", # Put the name that matches your alg name such as 'PPO' or 'DQN'
        run_config=train.RunConfig(
            name='FinalProject_Copycat', #  Name this something reasonable
            local_dir=ray_results,
            stop={
                "episode_reward_mean": .98*total_examples, # another example of stopping criteria
                'training_iteration': training_iterations,
                },
            checkpoint_config=train.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_score_attribute='episode_reward_mean',
                checkpoint_score_order='max',
                checkpoint_frequency=10,
                num_to_keep=5
                ),
        ),
        param_space=config
        )


    results = tuner.fit()
    print("dun")

if __name__ == '__main__':
    main()