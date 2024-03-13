# Ref: https://github.com/DerwenAI/gym_example/blob/main/sample.py
# Following tutorial (REF: https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5)
# associated with above repo
import gymnasium as gym
import cloning_env
import os
import numpy as np
from sklearn.model_selection import train_test_split
from data_management import load_train_test_subset, trim_dataset_by_index, get_class_labels_normal, get_class_labels_strs


def run_one_episode (env):
    env.reset()
    sum_reward = 0
    for i in range(env.MAX_STEPS):
        action = env.action_space.sample()
        state, reward, done, _, info = env.step(action)
        sum_reward += reward        
        if done:
            break    
    return sum_reward

def get_data():
    # Random Seed
    SEED = 1
    VAL_SPLIT = 0.25
    # Need to downsample heavily for reasonable runtimes
    # 0.15 -> by a factor of 20
    DOWNSAMPLE_FACTOR = .999
    # Load in the data which was only trimmed by SNR to >=10
    data_storage_dir = os.path.join(os.getcwd(),'data','project')
    signals_train_full, labels_int_train_full, snrs_train_full, signals_test, labels_int_test, snrs_test = load_train_test_subset(data_storage_dir)

    # Trim down data to "normal" dataset
    class_labels_keep = get_class_labels_normal()
    class_labels_str = get_class_labels_strs(class_labels_keep)
    signals_train_full, labels_int_train_full, snrs_train_full = trim_dataset_by_index(signals_train_full, labels_int_train_full, snrs_train_full, class_labels_keep)
    
    # If using D_F = 0.15:
    # Trim down further to total of ~66,914 samples for all training
    # Est. ~22.3k samples total (~2k samples/class) for each project stage
    # If using D_G = 0.075, halve the above
    idxs = np.arange(len(snrs_train_full))
    # First, break down indices by project parts
    idxs_train_down, idxs_val_down, labels_int_train_down, labels_int_val_down = train_test_split(idxs, labels_int_train_full, random_state=SEED, train_size=DOWNSAMPLE_FACTOR)
    idxs_train_pt1_full, idxs_train_pt23_full, labels_int_train_pt1_full, labels_int_train_pt23_full = train_test_split(idxs_train_down, labels_int_train_down, random_state=SEED, train_size=1/3)
    idxs_train_pt2_full, idxs_train_pt3_full, labels_int_train_pt2_full, labels_int_train_pt3_full = train_test_split(idxs_train_pt23_full, labels_int_train_pt23_full, random_state=SEED, train_size=0.5)
    # Return variables to check distributions are fairly equivalent for each part
    values_pt1, counts_pt1 = np.unique(labels_int_train_pt1_full, return_counts=True)
    values_pt2, counts_pt2 = np.unique(labels_int_train_pt2_full, return_counts=True)
    values_pt3, counts_pt3 = np.unique(labels_int_train_pt3_full, return_counts=True)

    # Now, break down indices by training and validation sets for each part
    idxs_train_pt1, idxs_val_pt1, labels_int_train_pt1, labels_int_val_pt1 = train_test_split(idxs_train_pt1_full, labels_int_train_pt1_full, random_state=SEED, test_size=VAL_SPLIT)
    idxs_train_pt2, idxs_val_pt2, labels_int_train_pt2, labels_int_val_pt2 = train_test_split(idxs_train_pt2_full, labels_int_train_pt2_full, random_state=SEED, test_size=VAL_SPLIT)
    idxs_train_pt3, idxs_val_pt3, labels_int_train_pt3, labels_int_val_pt3 = train_test_split(idxs_train_pt3_full, labels_int_train_pt3_full, random_state=SEED, test_size=VAL_SPLIT)

    # These arrays are the signals and snrs to be used for training/val in a given part
    signals_train_pt3 = signals_train_full[idxs_train_pt3,:,:]
    signals_val_pt3 = signals_train_full[idxs_val_pt3,:,:]
    return signals_train_pt3, class_labels_keep


def main ():
    rf_data, class_labels_keep = get_data()

    # first, create the custom environment and run it for one episode
    env_config = {'render_mode': 'human', 'rf_data': rf_data, 'num_classes': len(class_labels_keep)}
    env = gym.make("cloning-v0", env_config=env_config)
    sum_reward = run_one_episode(env)

    # next, calculate a baseline of rewards based on random actions
    # (no policy)
    history = []

    for _ in range(10000):
        sum_reward = run_one_episode(env)
        history.append(sum_reward)

    avg_sum_reward = sum(history) / len(history)
    print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))

if __name__ == '__main__':
    main()