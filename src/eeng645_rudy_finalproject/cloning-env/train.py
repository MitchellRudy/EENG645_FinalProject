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
from data_management import load_train_test_subset, trim_dataset_by_index, get_class_labels_normal, get_class_labels_strs


# def get_data():
#     # Random Seed
#     SEED = 1
#     VAL_SPLIT = 0.25
#     # Need to downsample heavily for reasonable runtimes
#     # 0.15 -> by a factor of 20
#     DOWNSAMPLE_FACTOR = .15
#     # Load in the data which was only trimmed by SNR to >=10
#     data_storage_dir = os.path.join(os.getcwd(),'data','project')
#     signals_train_full, labels_int_train_full, snrs_train_full, signals_test, labels_int_test, snrs_test = load_train_test_subset(data_storage_dir)

#     # Trim down data to "normal" dataset
#     class_labels_keep = get_class_labels_normal()
#     class_labels_str = get_class_labels_strs(class_labels_keep)
#     signals_train_full, labels_int_train_full, snrs_train_full = trim_dataset_by_index(signals_train_full, labels_int_train_full, snrs_train_full, class_labels_keep)
    
#     # If using D_F = 0.15:
#     # Trim down further to total of ~66,914 samples for all training
#     # Est. ~22.3k samples total (~2k samples/class) for each project stage
#     # If using D_G = 0.075, halve the above
#     idxs = np.arange(len(snrs_train_full))
#     # First, break down indices by project parts
#     idxs_train_down, idxs_val_down, labels_int_train_down, labels_int_val_down = train_test_split(idxs, labels_int_train_full, random_state=SEED, train_size=DOWNSAMPLE_FACTOR)
#     idxs_train_pt1_full, idxs_train_pt23_full, labels_int_train_pt1_full, labels_int_train_pt23_full = train_test_split(idxs_train_down, labels_int_train_down, random_state=SEED, train_size=1/3)
#     idxs_train_pt2_full, idxs_train_pt3_full, labels_int_train_pt2_full, labels_int_train_pt3_full = train_test_split(idxs_train_pt23_full, labels_int_train_pt23_full, random_state=SEED, train_size=0.5)
#     # Return variables to check distributions are fairly equivalent for each part
#     values_pt1, counts_pt1 = np.unique(labels_int_train_pt1_full, return_counts=True)
#     values_pt2, counts_pt2 = np.unique(labels_int_train_pt2_full, return_counts=True)
#     values_pt3, counts_pt3 = np.unique(labels_int_train_pt3_full, return_counts=True)

#     # Now, break down indices by training and validation sets for each part
#     idxs_train_pt1, idxs_val_pt1, labels_int_train_pt1, labels_int_val_pt1 = train_test_split(idxs_train_pt1_full, labels_int_train_pt1_full, random_state=SEED, test_size=VAL_SPLIT)
#     idxs_train_pt2, idxs_val_pt2, labels_int_train_pt2, labels_int_val_pt2 = train_test_split(idxs_train_pt2_full, labels_int_train_pt2_full, random_state=SEED, test_size=VAL_SPLIT)
#     idxs_train_pt3, idxs_val_pt3, labels_int_train_pt3, labels_int_val_pt3 = train_test_split(idxs_train_pt3_full, labels_int_train_pt3_full, random_state=SEED, test_size=VAL_SPLIT)

#     # These arrays are the signals and snrs to be used for training/val in a given part
#     signals_train_pt3 = signals_train_full[idxs_train_pt3,:,:]
#     signals_val_pt3 = signals_train_full[idxs_val_pt3,:,:]
#     return signals_train_pt3

############################
##### Data Acquisition #####
############################

def get_data2(class_labels_keep=[3,8,2], num_examples=100):
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
    
    signals_train_pt3_trimmed = np.array([])
    signals_train_pt3_trimmed = signals_train_pt3_trimmed.reshape((signals_train_pt3_trimmed.shape[0],1024,2))
    labels_train_pt3_trimmed = np.array([])
    signals_test_pt3_trimmed = np.array([])
    signals_test_pt3_trimmed = signals_test_pt3_trimmed.reshape((signals_train_pt3_trimmed.shape[0],1024,2))
    labels_test_pt3_trimmed = np.array([])
    for class_label in class_labels_keep:
        signals, labels, _ = trim_dataset_by_index(signals_train_pt3, labels_int_train_pt3, snrs_train_pt3, [class_label])
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        indices = indices[0:num_examples]
        signals_train_pt3_trimmed = np.concatenate((signals_train_pt3_trimmed,signals[indices]),axis=0)
        labels_train_pt3_trimmed = np.concatenate((labels_train_pt3_trimmed,labels[indices]),axis=0)
        
        signals, labels, _ = trim_dataset_by_index(signals_test_pt3, labels_int_test_pt3, snrs_test_pt3, [class_label])
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        indices = indices[0:num_examples]
        signals_test_pt3_trimmed = np.concatenate((signals_test_pt3_trimmed,signals[indices]),axis=0)
        labels_test_pt3_trimmed = np.concatenate((labels_test_pt3_trimmed,labels[indices]),axis=0)

    indices = np.arange(len(labels_train_pt3_trimmed))
    np.random.shuffle(indices)

    signals_train_pt3_trimmed = signals_train_pt3_trimmed[indices,:,:]
    labels_train_pt3_trimmed = labels_train_pt3_trimmed[indices]

    indices = np.arange(len(labels_test_pt3_trimmed))
    np.random.shuffle(indices)

    signals_test_pt3_trimmed = signals_test_pt3_trimmed[indices,:,:]
    labels_test_pt3_trimmed = labels_test_pt3_trimmed[indices]

    
    # signals_FM, labels_FM, _ = trim_dataset_by_index(signals_train_pt3, labels_int_train_pt3, snrs_train_pt3, [3])
    # signals_BPSK, labels_BPSK, _ = trim_dataset_by_index(signals_train_pt3, labels_int_train_pt3, snrs_train_pt3, [8])
    # signals_OOK, labels_OOK, _ = trim_dataset_by_index(signals_train_pt3, labels_int_train_pt3, snrs_train_pt3, [22])

    # indices_FM = np.arange(len(labels_FM))
    # np.random.shuffle(indices_FM)
    # indices_FM = indices_FM[0:num_examples]    
    # signals_FM = signals_FM[indices_FM]
    # labels_FM = labels_FM[indices_FM]

    # indices_BPSK = np.arange(len(labels_BPSK))
    # np.random.shuffle(indices_BPSK)
    # indices_BPSK = indices_BPSK[0:num_examples]
    # signals_BPSK = signals_BPSK[indices_BPSK]
    # labels_BPSK = labels_BPSK[indices_BPSK]

    # indices_OOK = np.arange(len(labels_OOK))
    # np.random.shuffle(indices_OOK)
    # indices_OOK = indices_OOK[0:num_examples]
    # signals_OOK = signals_OOK[indices_OOK]
    # labels_OOK = labels_OOK[indices_OOK]

    # signals_train_pt3 = np.concatenate((signals_FM, signals_BPSK),axis=0)
    # labels_train_pt3 = np.concatenate((labels_FM, labels_BPSK),axis=0)

        
    # signals_FM_test, labels_FM_test, _ = trim_dataset_by_index(signals_test_pt3, labels_int_test_pt3, snrs_test_pt3, [3])
    # signals_BPSK_test, labels_BPSK_test, _ = trim_dataset_by_index(signals_test_pt3, labels_int_test_pt3, snrs_test_pt3, [8])
    # signals_OOK_test, labels_OOK_test, _ = trim_dataset_by_index(signals_test_pt3, labels_int_test_pt3, snrs_test_pt3, [22])

    # indices_FM_test = np.arange(len(labels_FM_test))
    # np.random.shuffle(indices_FM_test)
    # indices_FM_test = indices_FM_test[0:num_examples]    
    # signals_FM_test = signals_FM_test[indices_FM_test]
    # labels_FM_test = labels_FM_test[indices_FM_test]

    # indices_BPSK_test = np.arange(len(labels_BPSK_test))
    # np.random.shuffle(indices_BPSK_test)
    # indices_BPSK_test = indices_BPSK_test[0:num_examples]
    # signals_BPSK_test = signals_BPSK_test[indices_BPSK_test]
    # labels_BPSK_test = labels_BPSK_test[indices_BPSK_test]

    # indices_OOK_test = np.arange(len(labels_OOK_test))
    # np.random.shuffle(indices_BPSK)
    # indices_OOK_test = indices_OOK_test[0:num_examples]
    # signals_OOK_test = signals_OOK_test[indices_OOK_test]
    # labels_OOK_test = labels_OOK_test[indices_OOK_test]

    # signals_test_pt3 = np.concatenate((signals_FM_test, signals_BPSK_test),axis=0)
    # labels_test_pt3 = np.concatenate((labels_FM_test, labels_BPSK_test),axis=0)

   

    return signals_train_pt3_trimmed, labels_train_pt3_trimmed, signals_test_pt3_trimmed, labels_test_pt3_trimmed

# In this example, keep FM (3), BPSK (8)
class_labels_keep = [3,8]
class_labels_keep = get_class_labels_normal()
# Use 50 examples of each
num_examples = 10
signals_train_pt3, labels_train_pt3, signals_test_pt3, labels_test_pt3 = get_data2(class_labels_keep, num_examples)
num_classes = len(class_labels_keep)
total_examples = num_examples*num_classes



chkpt_root = "chkpoint/"
shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
ray_results = f"{os.getcwd()}/ray_results/"
if not os.path.exists(ray_results):
    os.mkdir(ray_results)
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

ray.init(ignore_reinit_error=True)

select_env = "cloning-v0"
register_env("cloning-v0", CloningEnv_v0)

local_mode = False
training_iterations = 150 # max iterations before stopping - recommended
num_cpu = 5
num_gpus = 0
num_eval_workers = 1

driver_cpu = 1 # leave this alone
# How CPUs are spread
num_rollout_workers = num_cpu - driver_cpu - num_eval_workers

env_config = {
    'render_mode': 'human',
    'rf_data': signals_train_pt3,
    'labels': None,
    'num_classes': num_classes,
    'max_steps': signals_train_pt3.shape[0]-1
}
print(env_config)

config = (  # 1. Configure the algorithm,
    PPOConfig() # put the actual config object for the algorithm you intend to use (Such as PPO or DQN)
    .environment("cloning-v0", env_config=env_config)
    .experimental(_enable_new_api_stack=False)
    .rollouts(
        num_rollout_workers=num_rollout_workers,
        batch_mode='truncate_episodes',
        enable_connectors=False
        )
    .resources(num_gpus=num_gpus)
    .framework("tf2", eager_tracing=True)
    .training(
        # Put hyperparams here as needed. Look in the AlgorithmConfig object and child object for available params
        # REF: https://docs.ray.io/en/master/rllib/rllib-algorithms.html#ppo
        lr=0.0001, # learning rate
        gamma=0.95, # "Discount factor of Markov Decision process"
        kl_coeff=0, # Initial coefficient for Kullback-Leibler divergence, penalizes new policies for beeing too different from previous policy
        # train_batch_size=128
        )
    .evaluation(evaluation_num_workers=num_eval_workers, evaluation_interval=10)
    # .callbacks(callbacks)
    .reporting(keep_per_episode_custom_metrics=False) # decides whether custom metrics in Tensorboard are per episode or mean/min/max
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
            checkpoint_frequency=100,
            num_to_keep=5
            ),
    ),
    param_space=config
    )


results = tuner.fit()
print("dun")