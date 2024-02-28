import os

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
    main()