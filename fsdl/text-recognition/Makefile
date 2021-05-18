#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = text_recogntion
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# BUILD                                                                      #
#################################################################################

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using pylint
lint:
	pylint src disable=dangerous-default-value

## Deploy Model
deploy:
	`docker 

#################################################################################
# ENVIRONMENT                                                             #
#################################################################################

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
	conda env update --prune -f environment.yml
	conda activate $(PROJECT_NAME)
else
	$(PYTHON_INTERPRETER) -m virtualenv $(PROJECT_NAME)
endif

activate_environment:
ifeq (True,$(HAS_CONDA))
	conda activate $(PROJECT_NAME)
else
	`source` $(PROJECT_NAME)/bin/activate
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

exit_environment:
ifeq (True,$(HAS_CONDA))
	conda deactivate
else
	`deactivate`
endif

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install pip-tools
	pip-compile ./requirements/dev.in
	pip-sync ./requirements/dev.txt

#################################################################################
# PROJECT                                                                 #
#################################################################################

## Prepare Dataset
dataset:
	$(PYTHON_INTERPRETER) src/data/base_data_module.py

## Unit Test Model
unit_test:
	$(PYTHON_INTERPRETER) src/models/$(MODEL)/model_test.py

## Train Model
train:
	$(PYTHON_INTERPRETER) src/experiments/train_model.py