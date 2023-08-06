#!/bin/bash

python3 -m venv literature_survey

. literature_survey/bin/activate

python3 -m pip install -r requirements.txt

deactivate