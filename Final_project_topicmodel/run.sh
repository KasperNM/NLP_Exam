#!/usr/bin/bash

apt-get install python3-venv

source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 main.py 

deactivate