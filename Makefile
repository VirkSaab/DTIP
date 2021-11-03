# set the name of the virtual environment, project name, and python version
VENV := venv
PROJ_NAME := dtip
PYI := python3.8

.PHONY: help ce make_venv install clean make_docs update_docs test_base

help:	## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep


ce:	## Display current environment information
	@echo "Current environment:"
	@echo "\t Python - $(shell which python)"
	@echo "\t Pip - $(shell which pip)"


make_venv:	## Create a virtual environment.
ifneq ($(wildcard ./${VENV}/bin/activate),)
	@echo "${VENV} folder already exists. Delete it first to make new one."
else
	@echo "creating virtual environment..."
	@$(PYI) -m venv $(VENV)
	@./$(VENV)/bin/pip install -U pip
	@./$(VENV)/bin/pip install ruamel.yaml
	@./$(VENV)/bin/pip install easydict
	@./$(VENV)/bin/pip install -r requirements.txt
	@echo "Installing ${PROJ_NAME} in dev mode..."
	@./$(VENV)/bin/pip install -e .
endif

install:	## Install the project in editable (or dev) mode.
	@echo "Installing ${PROJ_NAME} in dev mode..."
	@./$(VENV)/bin/pip install -e .


clean:	## Clean unused files.
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '__pycache__' -exec rm -rf {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf docs/build
	@echo Done!


make_docs:	## create new documentation
ifneq ($(wildcard ./docs/.*),)
	@echo "docs folder already exists. Delete that first to make new one."
else
	@echo "Creating sphinx docs..."
	sphinx-quickstart docs
	sphinx-build -b html docs/source docs/build/html
	sphinx-apidoc -o docs/source ${PROJ_NAME}
	make -C docs html
endif


update_docs: docs/Makefile ## Update documentation
	sphinx-apidoc -o docs/source ${PROJ_NAME} --force
	make -C docs html
	@git log --graph --abbrev-commit --decorate --format=format:'%C(bold blue)%h%C(reset) - %C(bold green)(%ar)%C(reset) %C(white)%s%C(reset) %C(dim white)- %an%C(reset)%C(bold yellow)%d%C(reset)' --all > CHANGELOG.md


install_jupyter:
	@./$(VENV)/bin/pip install jupyter notebook ipykernel ipywidgets
	@./$(VENV)/bin/python -m ipykernel install --user --name $(VENV) --display-name "Python ($(VENV))"