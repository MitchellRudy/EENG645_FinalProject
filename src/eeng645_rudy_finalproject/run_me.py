import os
from data_management import load_train_test_subset

def build_subdirectories():
    cwd = os.getcwd()
    subdir_names = ['logs','models','figures']
    for subdir in subdir_names:
        if not os.path.exists(os.path.join(cwd,subdir)):
            os.mkdir(os.path.join(cwd,subdir))
            print(f"Created directory: {os.path.join(cwd,subdir)}")
    return



def main():
    # FLAGS / SETTINGS
    BUILD_MODULATION_CLASSIFIER = True
    BUILD_COPYCAT = True
    # Build out subdirectories for various files
    build_subdirectories()
    
    if BUILD_MODULATION_CLASSIFIER:
        pass
    else:
        pass
    return

if __name__ == '__main__':
    # main()
    data_storage_dir = os.path.join(os.getcwd(),'data','project')
    signals_train_full, labels_int_train_full, snrs_train_full, signals_test, labels_int_test, snrs_test = load_train_test_subset(data_storage_dir)
    pass