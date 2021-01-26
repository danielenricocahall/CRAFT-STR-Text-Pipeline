#!/bin/bash
git submodule init
git submodule update
ln -s CRAFT-pytorch/ craft_detection
ln -s deep-text-recognition-benchmark/ scene_text_recognition
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir CRAFT_models
pushd CRAFT_models && gdown https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ && popd || exit
mkdir STR_models
pushd STR_models || exit
gdown https://drive.google.com/uc?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0
gdown https://drive.google.com/uc?id=1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9
gdown https://drive.google.com/uc?id=1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY
popd || exit
export PYTHONPATH="${PYTHONPATH}:./craft_detection"
export PYTHONPATH="${PYTHONPATH}:./scene_text_recognition"