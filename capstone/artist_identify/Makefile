#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = artist_identify
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using pylint
lint:
	pylint src

#################################################################################
# ENVIRONMENT RULES                                                             #
#################################################################################

## Set up python interpreter environment
create_environment:
	$(PYTHON_INTERPRETER) -m virtualenv $(PROJECT_NAME)
	`source` $(PROJECT_NAME)/bin/activate

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

exit_environment:
	`deactivate`

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make Dataset
dataset:
	$(PYTHON_INTERPRETER) src/data/make_dataset.py $(INPUT) $(OUTPUT)

## Make Model
model:
	$(PYTHON_INTERPRETER) ./src/models/$(MODEL)/model.py

## Make Model
test: 
	$(PYTHON_INTERPRETER) ./src/models/$(MODEL)/model_test.py $(MODEL_PATH)

## Train Model
train:
	$(PYTHON_INTERPRETER) src/models/train_model.py $(BASE_PATH)

## Train Model
evaluate:
	$(PYTHON_INTERPRETER) src/models/evaluate_model.py $(MODEL_PATH)

## Predict Sample
predict:
	$(PYTHON_INTERPRETER) src/models/predict_model.py $(MODEL_PATH) $(IMAGE_PATH)

## Deploy Model
	`docker 

## Performance Test
perf:
	$(PYTHON_INTERPRETER) src/api/load_test.py
