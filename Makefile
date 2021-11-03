# set the name of the virtual environment and python version to use
VENV := venv
PYI := python3.8

.PHONY: help ce make_venv install

help:	## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep


ce:	## Show the current environment.
	@echo "Current environment:"
	@echo "\t Python - $(shell which python)"
	@echo "\t Pip - $(shell which pip)"


make_venv:	## Create a virtual environment.
ifneq ($(wildcard ./${VENV}/bin/activate),)
	@echo Warning: "${VENV} already exists. Enter Y to recreate."
	@echo continue? [Y/n]
	@read line; if [ $$line = "n" ]; then echo Aborted.; exit 1; fi
endif
	@echo "creating virtual environment..."
	@$(PYI) -m venv $(VENV)
	@./$(VENV)/bin/pip install -U pip
	@./$(VENV)/bin/pip install -r requirements.txt
	@echo "Installing ${PROJ_NAME} in dev mode..."
	@./$(VENV)/bin/pip install -e .


install:	## Install the project in editable (or dev) mode.
	@echo "Installing ${PROJ_NAME} in dev mode..."
	@./$(VENV)/bin/pip install -e .

