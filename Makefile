VENV_PATH='.venv/bin/activate'

init: ## sets up environment and installs requirements
init:
	pip3 install -r requirements.txt

env: ## Source venv and environment files for testing
env:
	python3 -m venv env
	source $(VENV_PATH)
train:
	python3 train.py