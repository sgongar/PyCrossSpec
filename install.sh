#!/usr/bin/env bash
sudo apt-get install tcl-dev tk-dev python-tk python3-tk

virtualenv -p /usr/bin/python3 .venv
source .venv/bin/activate
pip install -r requirements.txt