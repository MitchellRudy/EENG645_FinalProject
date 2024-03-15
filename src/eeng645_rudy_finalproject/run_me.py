import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.data import AUTOTUNE, Dataset
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model

from data_management import save_train_test_subset, load_train_test_subset, trim_dataset_by_index, get_class_labels_normal, get_class_labels_strs, preprocess_labels

from modulation_classifier import build_mod_classifier
from snr_estimator import build_snr_estimator

def build_subdirectories(subdir_names = ['logs','models','figures']):
    cwd = os.getcwd()
    for subdir in subdir_names:
        if not os.path.exists(os.path.join(cwd,subdir)):
            os.mkdir(os.path.join(cwd,subdir))
            print(f"Created directory: {os.path.join(cwd,subdir)}")
    return


def main():
    #################
    ##### Flags #####
    #################
    GENERATE_DATASETS = False
    GEN_VAL_AND_TEST_SUBPLOTS = True

    DO_PART1 = False
    TRAIN_MODULATION_CLASSIFIER = False
    TEST_MODULATION_CLASSIFIER = False

    DO_PART2 = True
    TRAIN_SNR_ESTIMATOR = False
    TEST_SNR_ESTIMATOR = False

    ###########################
    ##### Sim. Parameters #####
    ###########################
    SEED = 1
    CLASS_LABELS_KEEP = get_class_labels_normal()
    # CLASS_LABELS_KEEP = [x for x in range(0,24)] # Use this to keep all modulation types
    CLASS_LABELS_STR = get_class_labels_strs(CLASS_LABELS_KEEP)
    MOD_CLASS_CHECKPOINT = "model_mod_class_cp.h5"
    MOD_CLASS_CHECKPOINT_LOC = os.path.join(os.getcwd(),"models",MOD_CLASS_CHECKPOINT)
    SNR_EST_CHECKPOINT = "model_snr_est_cp.h5"
    SNR_EST_CHECKPOINT_LOC = os.path.join(os.getcwd(),"models",SNR_EST_CHECKPOINT)

    ###########################
    ##### Directory Prep. #####
    ###########################
    build_subdirectories()

    #####################################
    ##### Model Training Parameters #####
    #####################################
    VALIDATION_SPLIT = 0.25
    BUFFER_SIZE = 1000
    BATCH_SIZE = 128
    EPOCHS = 100

    ############################
    ##### Data Acquisition #####
    ############################

    # "main" root of all data packages used in project
    data_storage_dir = os.path.join(os.getcwd(),'data','project')

    # Define data package to pull pt 1 / 2 / 3 from
    if GENERATE_DATASETS:
        print("Generating datasets...")
        data_package = save_train_test_subset(data_storage_dir, datakeep_percentage = 1)
    else:
        data_package = "snr10_keep100_test10_seed1"

    data_split_dir = os.path.join(data_storage_dir,data_package)
    # Storage locations for data used in each part
    pt1_storage_dir = os.path.join(data_split_dir,"pt1_data")
    pt2_storage_dir = os.path.join(data_split_dir,"pt2_data")
    pt3_storage_dir = os.path.join(data_split_dir,"pt3_data")

    try:
        signals_train_pt1, labels_int_train_pt1, snrs_train_pt1, signals_test_pt1, labels_int_test_pt1, snrs_test_pt1 = load_train_test_subset(pt1_storage_dir)
    except:
        print(f"Error loading in data for part 1 from {pt1_storage_dir}")
        DO_PART1 = False
    
    try:
        signals_train_pt2, labels_int_train_pt2, snrs_train_pt2, signals_test_pt2, labels_int_test_pt2, snrs_test_pt2 = load_train_test_subset(pt2_storage_dir)
    except:
        print(f"Error loading in data for part 2 from {pt2_storage_dir}")
        DO_PART2 = False

    try:
        signals_train_pt3, labels_int_train_pt3, snrs_train_pt3, signals_test_pt3, labels_int_test_pt3, snrs_test_pt3 = load_train_test_subset(pt3_storage_dir)
    except:
        print(f"Error loading in data for part 3 from {pt3_storage_dir}")
        DO_PART3 = False

    ##############################
    ##### Data Preprocessing #####
    ##############################

    ### 1. Trim datasets based on CLASS_LABELS_KEEP
    # Part 1 - Modulation Classifier
    signals_train_pt1, labels_int_train_pt1, snrs_train_pt1 = trim_dataset_by_index(signals_train_pt1, labels_int_train_pt1, snrs_train_pt1, CLASS_LABELS_KEEP)
    signals_test_pt1, labels_int_test_pt1, snrs_test_pt1 = trim_dataset_by_index(signals_test_pt1, labels_int_test_pt1, snrs_test_pt1, CLASS_LABELS_KEEP)
    # Part 2 - SNR Estimator
    signals_train_pt2, labels_int_train_pt2, snrs_train_pt2 = trim_dataset_by_index(signals_train_pt2, labels_int_train_pt2, snrs_train_pt2, CLASS_LABELS_KEEP)
    signals_test_pt2, labels_int_test_pt2, snrs_test_pt2 = trim_dataset_by_index(signals_test_pt2, labels_int_test_pt2, snrs_test_pt2, CLASS_LABELS_KEEP)
    # Part 3 - Reinforcement Learning
    signals_train_pt3, labels_int_train_pt3, snrs_train_pt3 = trim_dataset_by_index(signals_train_pt3, labels_int_train_pt3, snrs_train_pt3, CLASS_LABELS_KEEP)
    signals_test_pt3, labels_int_test_pt3, snrs_test_pt3 = trim_dataset_by_index(signals_test_pt3, labels_int_test_pt3, snrs_test_pt3, CLASS_LABELS_KEEP)

    ### 2. Convert integer-based labels BACK to one-hot encoded labels
    # Part 1 - Modulation Classifier
    labels_train_pt1 = preprocess_labels(labels_int_train_pt1, CLASS_LABELS_KEEP)
    labels_test_pt1 = preprocess_labels(labels_int_test_pt1, CLASS_LABELS_KEEP)
    # Part 2 - SNR Estimator
    labels_train_pt2 = preprocess_labels(labels_int_train_pt2, CLASS_LABELS_KEEP)
    labels_test_pt2 = preprocess_labels(labels_int_test_pt2, CLASS_LABELS_KEEP)
    # Part 3 - Reinforcement Learning
    labels_train_pt3 = preprocess_labels(labels_int_train_pt3, CLASS_LABELS_KEEP)
    labels_test_pt3 = preprocess_labels(labels_int_test_pt3, CLASS_LABELS_KEEP)

    ### 3. Split into training and validation sets based on VALIDATION_SPLIT. Use indices to avoid crashes from dataset memory requirements
    # Part 1 - Modulation Classifier
    indices = np.arange(len(snrs_train_pt1))
    indices_train_pt1, indices_val_pt1, labels_train_pt1, labels_val_pt1 = train_test_split(indices, labels_train_pt1, random_state=SEED, test_size=VALIDATION_SPLIT)
    
    signals_val_pt1 = signals_train_pt1[indices_val_pt1,:,:]
    snrs_val_pt1 = snrs_train_pt1[indices_val_pt1]

    signals_train_pt1 = signals_train_pt1[indices_train_pt1,:,:]
    snrs_train_pt1 = snrs_train_pt1[indices_train_pt1]
    # Part 2 - SNR Estimator
    indices = np.arange(len(snrs_train_pt2))
    indices_train_pt2, indices_val_pt2, labels_train_pt2, labels_val_pt2 = train_test_split(indices, labels_train_pt2, random_state=SEED, test_size=VALIDATION_SPLIT)
    
    signals_val_pt2 = signals_train_pt2[indices_val_pt2,:,:]
    # labels_val_pt2 = labels_train_pt2[indices_val_pt2,:]
    snrs_val_pt2 = snrs_train_pt2[indices_val_pt2]

    signals_train_pt2 = signals_train_pt2[indices_train_pt2,:,:]
    # labels_train_pt2 = labels_train_pt2[indices_train_pt2,:]
    snrs_train_pt2 = snrs_train_pt2[indices_train_pt2]

    # Part 3 - Reinforcement Learning
    # In a different file

    ### 4. Prepare and batch datasets
    # Part 1 - Modulation Classifier
    train_dataset_pt1 = Dataset.from_tensor_slices((signals_train_pt1,labels_train_pt1))
    train_batches_pt1 = train_dataset_pt1.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size = AUTOTUNE)
    total_train_samples_pt1 = len(train_batches_pt1)*BATCH_SIZE
    val_dataset_pt1 = Dataset.from_tensor_slices((signals_val_pt1,labels_val_pt1))
    val_batches_pt1 = val_dataset_pt1.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size = AUTOTUNE)

    # Part 2 - SNR Estimator
    train_dataset_pt2 = Dataset.from_tensor_slices((signals_train_pt2,snrs_train_pt2))
    train_batches_pt2 = train_dataset_pt2.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size = AUTOTUNE)
    total_train_samples_pt2 = len(train_batches_pt2)*BATCH_SIZE
    val_dataset_pt2 = Dataset.from_tensor_slices((signals_val_pt2,snrs_val_pt2))
    val_batches_pt2 = val_dataset_pt2.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size = AUTOTUNE)

    # Part 3 - Reinforcement Learning



    ##########################################
    ##### Part 1 - Modulation Classifier #####
    ##########################################

    if DO_PART1:
        if TRAIN_MODULATION_CLASSIFIER:
            # Define Callbacks
            lr_scheduler_cb = ReduceLROnPlateau(factor = 0.75, patience = 10)
            checkpoint_model_cb = ModelCheckpoint(MOD_CLASS_CHECKPOINT_LOC,save_best_only=True)
            early_stopping_cb = EarlyStopping(patience=15)
            cbs = [lr_scheduler_cb, checkpoint_model_cb, early_stopping_cb]

            mod_class_model = build_mod_classifier(num_outputs=len(CLASS_LABELS_KEEP))            
            mod_class_model.fit(
                train_batches_pt1, 
                epochs=EPOCHS, 
                validation_data=val_batches_pt1,
                callbacks=cbs
                )
        mod_class_model = load_model(MOD_CLASS_CHECKPOINT_LOC)
        ### Evaluate Model
        def plot_confusion_matrix(y_pred, y_true, class_labels=None, ax=None):
            cm_mod_class = confusion_matrix(y_pred=y_pred,y_true=y_true)
            cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm_mod_class, display_labels=class_labels)
            cm_disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax, colorbar=False)
            plt.tight_layout()
            plt.xlabel("Predicted Mod. Scheme", fontsize=12)
            plt.ylabel("True Mod. Scheme", fontsize=12)
            return
            
        # Select either TEST data or VALIDATION data
        if TEST_MODULATION_CLASSIFIER:
            print("Testing Modulation Classifier")
            eval_data_pt1 = signals_test_pt1
            y_true = np.array([np.argmax(x) for x in labels_test_pt1])
            cm_file_name = "cm_mod_class_test.png"
        else:
            print("Validating Modulation Classifier")
            eval_data_pt1 = signals_val_pt1
            y_true = np.array([np.argmax(x) for x in labels_val_pt1])
            cm_file_name = "cm_mod_class_val.png"

        # Make predictions and plot in confusion matrix
        y_pred = np.argmax(mod_class_model.predict(eval_data_pt1), axis=1)
        pt1_accuracy = np.sum(y_pred == y_true)/len(y_true)
        print(f"Modulation Classifier Accuracy: {pt1_accuracy:.2f}")
        plt.figure()
        plot_confusion_matrix(y_pred, y_true, class_labels=CLASS_LABELS_STR)
        cm_save_loc = os.path.join(os.getcwd(),'figures', cm_file_name)
        plt.savefig(cm_save_loc, pad_inches=6)
        print(f"Part 1 confusion matrix saved to {cm_save_loc}")



    ##################################
    ##### Part 2 - SNR Estimator #####
    ##################################
    if DO_PART2:
        if TRAIN_SNR_ESTIMATOR:
            # Define Callbacks
            lr_scheduler_cb = ReduceLROnPlateau(factor = 0.75, patience = 10)
            checkpoint_model_cb = ModelCheckpoint(SNR_EST_CHECKPOINT_LOC,save_best_only=True)
            early_stopping_cb = EarlyStopping(patience=50)
            cbs = [checkpoint_model_cb, early_stopping_cb]

            snr_estimator_model = build_snr_estimator()
            snr_est_fit_history = snr_estimator_model.fit(
                train_batches_pt2, 
                epochs=200, 
                validation_data=val_batches_pt2,
                callbacks=cbs
            )
        snr_estimator_model = load_model(SNR_EST_CHECKPOINT_LOC)
        ### Evaluate Model# Plot SNR estimation Errors
        def plot_snr_preds(snr_true, snr_preds, snr_values, ax=None):
            """
            plot_snr_preds(snr_true, snr_preds, snr_values):

            snr_true - true snr "labels"
            snr_preds - values predicted by model
            snr_values - values to use as tick marks
            """
            if ax is None:
                plt.scatter(snr_true, snr_preds)
                plt.xticks(snr_values)
                plt.yticks(snr_values)
                plt.xlabel("True SNR (dB)", fontsize=18)
                plt.ylabel("Predicted SNR (dB)", fontsize=18)
                plt.grid(visible=True)
                plt.show()
            else:
                ax.scatter(snr_true, snr_preds)
                ax.set_xticks(snr_values)
                ax.set_yticks(snr_values)
                ax.set_xlabel("True SNR (dB)", fontsize=18)
                ax.set_ylabel("Predicted SNR (dB)", fontsize=18)
                ax.grid(visible=True)
            return

        def plot_snr_estimation_errors(
            snr_true, 
            snr_preds, 
            snr_values, 
            mod_labels,
            CLASS_LABELS_KEEP,
            CLASS_LABELS_STR,
            ax=None
            ):
            """
            plot_snr_estimation_errors(snr_true, snr_preds, mod_labels, savedir, test_val_str):

            Generate the plots to characterizae SNR estimation errors:
            - Square Error of each mod type at a given true SNR
            - Predictions vs SNR

            Inputs
            snr_true:nparray -
            snr_preds:nparray - 
            mod_labels:nparray -
            CLASS_LABELS_KEEP: -
            CLASS_LABELS_STR: -
            ax: figure axis
            """
            num_unique_snrs = len(snr_values)

            # calculate errors
            snr_errors = snr_preds - snr_true

            # Iteratively plot the mse's for each modulation type
            mse_array = np.zeros((num_unique_snrs, len(CLASS_LABELS_KEEP)))
            for idx in range(0,len(CLASS_LABELS_KEEP)):
                mod_class = CLASS_LABELS_KEEP[idx]
                plot_str = CLASS_LABELS_STR[idx]
                masked_array = np.ma.masked_where(mod_labels == idx, mod_labels)
                mod_mask = masked_array.mask

                mse_mod_type = np.zeros(num_unique_snrs)

                for idy in range(0,len(snr_values)):
                    snr = snr_values[idy]
                    snr_masked_array = np.ma.masked_where(snr_true == snr, snr_true)
                    snr_mask = snr_masked_array.mask
                    snr_mask = np.reshape(snr_mask, (len(snr_mask),))
                    mask_use = np.logical_and(mod_mask, snr_mask)

                    snr_errors_subset = snr_errors[mask_use]
                    mse_mod_type[idy] = np.mean(snr_errors_subset**2)
                if ax is None:
                    plt.plot(snr_values, mse_mod_type, linestyle='--', marker='o', label=plot_str)
                else:
                    ax.plot(snr_values, mse_mod_type, linestyle='--', marker='o', label=plot_str)
            
            if ax is None:
                plt.axhline(y=0.05, color='r', linestyle='-', label="Baseline")

                # plt.plot(snr_values, mse_array, linestyle='--', marker='o', label=plot_str)
                plt.grid(visible=True)
                plt.legend()
                plt.xticks(snr_values)
                plt.xlabel("True SNR (dB)", fontsize=18)
                plt.ylabel("MSE ($dB^2$)", fontsize=18)
                plt.show()
            else:
                ax.axhline(y=0.05, color='r', linestyle='-', label="Baseline")
                ax.grid(visible=True)
                ax.legend()
                ax.set_xticks(snr_values)
                ax.set_xlabel("True SNR (dB)", fontsize=18)
                ax.set_ylabel("MSE ($dB^2$)", fontsize=18)

            return
        
        if not GEN_VAL_AND_TEST_SUBPLOTS:
            if TEST_SNR_ESTIMATOR:
                print("Testing SNR Estimator")
                eval_data_pt2 = signals_test_pt2
                snr_true = snrs_test_pt2
                mod_labels = np.array([np.argmax(x) for x in labels_test_pt2])
                test_val_str = "test"
            else:
                print("Validating SNR Estimator")
                eval_data_pt2 = signals_val_pt2
                snr_true = snrs_val_pt2
                mod_labels = np.array([np.argmax(x) for x in labels_val_pt2])
                test_val_str = "val"

            # Estimate SNRs
            snr_preds = snr_estimator_model.predict(eval_data_pt2)
            snr_preds = np.reshape(snr_preds, (snr_preds.shape[0],1))
            sq_error = np.abs(snr_preds - snr_true)**2       

            
            # need to get unique levels of SNR values
            snr_values, _ = np.unique(snr_true, return_counts=True)
            
            plt.figure()
            plot_snr_estimation_errors(
                snr_true, 
                snr_preds,
                snr_values,
                mod_labels,
                CLASS_LABELS_KEEP,
                CLASS_LABELS_STR,
                )
            plt.tight_layout()
            fig_mse_v_true_savepath = os.path.join(os.getcwd(),'figures', f"snr_mse_{test_val_str}.png")
            plt.savefig(fig_mse_v_true_savepath, pad_inches=6)
            print(f"Saved Part 2 MSE plot to {fig_mse_v_true_savepath}")

            # Predictions Vs True SNR
            plt.figure()
            plot_snr_preds(snr_true, snr_preds, snr_values)
            fig_pred_v_true_savepath = os.path.join(os.getcwd(),'figures', f"snr_pred_{test_val_str}.png")
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            plt.savefig(fig_pred_v_true_savepath, pad_inches=6)
            print(f"Saved Part 2 SNR Prediction plot to {fig_pred_v_true_savepath}")
        else:
            # Get predictions for both val and test sets
            snr_preds_val = snr_estimator_model.predict(signals_val_pt2)
            snr_preds_val = np.reshape(snr_preds_val, (snr_preds_val.shape[0],1))
            snr_preds_test = snr_estimator_model.predict(signals_test_pt2)
            snr_preds_test = np.reshape(snr_preds_test, (snr_preds_test.shape[0],1))

            # get true values for both val and test sets
            snr_true_val = snrs_val_pt2
            snr_true_test = snrs_test_pt2

            # get square errors
            sq_error_val = np.abs(snr_preds_val - snr_true_val)**2
            sq_error_test = np.abs(snr_preds_test - snr_true_test)**2

            # get modulation labels for val and test
            mod_labels_val = np.array([np.argmax(x) for x in labels_val_pt2])
            mod_labels_test = np.array([np.argmax(x) for x in labels_test_pt2])

            # need to get unique levels of SNR values
            snr_values, _ = np.unique(snr_true_val, return_counts=True)

            # Generate subplots for MSE fig
            fig_mse, (ax1_mse, ax2_mse) = plt.subplots(2,1, figsize=(10,10))
            plot_snr_estimation_errors(
                snr_true_val, 
                snr_preds_val,
                snr_values,
                mod_labels_val,
                CLASS_LABELS_KEEP,
                CLASS_LABELS_STR,
                ax = ax1_mse
                )
            ax1_mse.set_title("a)", size=18)
            ax1_mse.tick_params(axis='both', which='major', labelsize=14)
            plot_snr_estimation_errors(
                snr_true_test, 
                snr_preds_test,
                snr_values,
                mod_labels_test,
                CLASS_LABELS_KEEP,
                CLASS_LABELS_STR,
                ax = ax2_mse
                )
            ax2_mse.set_title("b)", size=18)
            ax2_mse.tick_params(axis='both', which='major', labelsize=14)
            fig_mse.tight_layout()
            plt.show()
            fig_mse_v_true_savepath = os.path.join(os.getcwd(),'figures', f"snr_mse_val_test.png")
            plt.savefig(fig_mse_v_true_savepath, pad_inches=6)
            print(f"Saved Part 2 MSE plot to {fig_mse_v_true_savepath}")

            # generate subplots for Prediction fig
            fig_pred, (ax1_pred, ax2_pred) = plt.subplots(2,1, figsize=(10,10))

            plot_snr_preds(snr_true_val, snr_preds_val, snr_values, ax1_pred)
            ax1_pred.set_title("a)", size=18)
            ax1_pred.tick_params(axis='both', which='major', labelsize=14)
            ax1_pred.set_aspect('equal', adjustable='box')
            
            plot_snr_preds(snr_true_test, snr_preds_test, snr_values, ax2_pred)
            ax2_pred.set_title("b)", size=18)
            ax2_pred.tick_params(axis='both', which='major', labelsize=14)
            ax2_pred.set_aspect('equal', adjustable='box')

            fig_pred.tight_layout()
            plt.show()
            fig_pred_v_true_savepath = os.path.join(os.getcwd(),'figures', f"snr_pred_val_test.png")

            plt.savefig(fig_pred_v_true_savepath, pad_inches=6)
            print(f"Saved Part 2 SNR Prediction plot to {fig_pred_v_true_savepath}")

    ###########################################
    ##### Part 3 - Reinforcement Learning #####
    ###########################################
    # See "cloning-env-debug" directory for code generating results presented in class
    # "cloning-env" directory is being used to attempt improvements
    return

if __name__ == '__main__':
    main()
    # data_storage_dir = os.path.join(os.getcwd(),'data','project')
    # signals_train_full, labels_int_train_full, snrs_train_full, signals_test, labels_int_test, snrs_test = load_train_test_subset(data_storage_dir)
    pass