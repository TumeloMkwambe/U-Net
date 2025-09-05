ENV_NAME = venv
PYTHON_VERSION = 3.12

.PHONY: venv
venv:
	@echo "Creating virtual environment..."
	python$(PYTHON_VERSION) -m venv $(ENV_NAME)

.PHONY: install
install: venv
	@echo "Installing dependencies..."
	$(ENV_NAME)/bin/pip install --upgrade pip
	$(ENV_NAME)/bin/pip install -r requirements.txt

.PHONY: clean
clean:
	@echo "Removing virtual environment..."
	rm -rf $(ENV_NAME)