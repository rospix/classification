#!/bin/bash

# install python3-venv
sudo apt-get -y install python3-venv

# create the environment
python3 -m venv python-env

# activate the environment
source ./python-env/bin/activate

# install the requirements
python3 -m pip install -r requirements.txt
