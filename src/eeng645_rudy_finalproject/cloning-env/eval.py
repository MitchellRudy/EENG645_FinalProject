import pathlib
import os
import time
import sys
import pprint
import argparse
from ray.tune import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from cloning_env.envs.cloning_env import CloningEnv_v0

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from data_management import load_train_test_subset, trim_dataset_by_index, get_class_labels_normal, get_class_labels_strs
from custom_callbacks import GetPredictionsCallback

# Fix tensorflow bug when using tf2 framework with Algorithm.from_checkpoint()
from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()
tf.compat.v1.enable_eager_execution()


def get_data2(class_labels_keep=[3,8,2], num_examples=100):
    np.random.seed(1)
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

    return signals_train_pt3_trimmed, labels_train_pt3_trimmed, signals_test_pt3_trimmed, labels_test_pt3_trimmed

def progressBar(count_value, total, suffix=''):
    """Makes a progress bar in the terminal according to count_value out of total steps"""
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()

def eval_duration_fn_args(eval_duration, tic):
    """Allows total episodes :eval_duration: and start time :tic: to be used in eval_duration_fn"""
    def eval_duration_fn(num_units_done):
        """Makes a progress bar during alg.evaluate()"""
        print(f'Episodes evaluated: {num_units_done}/{eval_duration}')
        seconds = time.perf_counter() - tic
        mm, ss = divmod(seconds, 60)
        hh, mm = divmod(mm, 60)
        print(f'Time elapsed: {int(hh)}h {int(mm)}m {int(ss)}s')
        progressBar(num_units_done, eval_duration)
        # progressBar.update(n=num_units_done)
        return eval_duration-num_units_done
    return eval_duration_fn



def evaluate(checkpoint, rf_data = None, num_classes=None, evaluation_duration = 20, evaluation_num_workers = 5):

    """Runs a model evaluation from checkpoint for a duration with num workers and saves replays to folder"""

    register_env("cloning-v0", CloningEnv_v0)

    # Get the old config from the checkpoint
    old_alg = Algorithm.from_checkpoint(checkpoint=checkpoint)
    old_config = old_alg.get_config()
    config = old_config.copy(copy_frozen=False) # make an unfrozen copy

    env_config = {
        'render_mode': 'human',
        'rf_data': rf_data,
        'labels': None,
        'num_classes': num_classes,
        'max_steps': rf_data.shape[0]-1
    }

    # Update config for evaluation only run
    config_update = {
        'env_config': env_config,
        'evaluation_config': {
            'evaluation_interval': 1,
            'evaluation_duration_unit': 'episodes',
            'evaluation_duration': evaluation_duration,
            'evaluation_num_workers': evaluation_num_workers,
        },
        'num_rollout_workers': 0,
        'callbacks': GetPredictionsCallback
        # 'explore': False, # NOTE: DO NOT turn explore off with policy algs like PPO
    }
    config.update_from_dict(config_update)

    # build new alg
    alg = config.build()

    # # restore the policy and training history
    alg.restore(checkpoint_path=checkpoint)

    # Run the evaluation
    tic = time.perf_counter()
    eval_results = alg.evaluate(duration_fn=eval_duration_fn_args(eval_duration=evaluation_duration, tic=tic))

    # Report how it went
    expert_preds = np.array(eval_results['evaluation']['custom_metrics']['expert_preds'][0])
    agent_preds = np.array(eval_results['evaluation']['custom_metrics']['agent_preds'][0])

    all_class_strs = get_class_labels_strs(get_class_labels_normal())
    class_strs_needed = [all_class_strs[x] for x in np.unique(np.concatenate((agent_preds,expert_preds))).tolist()]

    accuracy = np.sum(agent_preds == expert_preds)/len(expert_preds)
    cm_mod_class = confusion_matrix(y_pred=agent_preds,y_true=expert_preds)
    plt.figure()
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm_mod_class, display_labels=class_strs_needed)
    cm_disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    cm_file_name = "rl_simple_mod_class.png"
    cm_save_loc = os.path.join(os.getcwd(),'figures', cm_file_name)
    plt.tight_layout()
    plt.savefig(cm_save_loc, pad_inches=6)
    print(f"Part 3 confusion matrix saved to {cm_save_loc}")


    seconds = time.perf_counter() - tic
    mm, ss = divmod(seconds, 60)
    hh, mm = divmod(mm, 60)
    print(f'Total time elapsed: {int(hh)}h {int(mm)}m {int(ss)}s')

if __name__== '__main__':
    
    # In this example, keep FM (3), BPSK (8)
    class_labels_keep = [3,8]
    # class_labels_keep = get_class_labels_normal()
    # Use 10 examples of each
    num_examples = 500
    signals_train_pt3, labels_train_pt3, signals_test_pt3, labels_test_pt3 = get_data2(class_labels_keep, num_examples)
    num_classes = len(get_class_labels_normal())
    total_examples = num_examples*num_classes

    # [3, 8]
    # 2000 examples
    # 50 training iterations

    checkpoint = '/remote_home/EENG645_FinalProject/ray_results/FinalProject_Copycat/PPO_cloning-v0_97f2f_00000_0_2024-03-11_12-26-32/checkpoint_000000'
    evaluation_duration = 1
    evaluation_num_workers = 3
    evaluate(checkpoint=checkpoint, evaluation_duration=evaluation_duration, evaluation_num_workers=evaluation_num_workers, rf_data=signals_test_pt3, num_classes=num_classes)