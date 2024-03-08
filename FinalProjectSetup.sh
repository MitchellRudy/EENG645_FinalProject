# ensure updated system
apt update && apt upgrade -y
snap refresh
snap install --classic code
# install pip (python package manager) if not installed already
apt install -y python3-tk
python3 -m pip install --upgrade pip
# install python requirements
python3 -m pip install -r requirements.txt
pip install -U "ray[data,train,tune,serve]"
pip install optuna