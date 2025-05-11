.PHONY: setup install clean run pull-model check-env check-ollama

# Default Python interpreter
PYTHON := python3.11
VENV := .venv
VENV_BIN := $(VENV)/bin
OLLAMA_MODEL := phi

# Check if running in virtual environment
check-venv:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "Please activate virtual environment first"; \
		exit 1; \
	fi

# Check if Ollama is running
check-ollama:
	@echo "Checking if Ollama is running..."
	@if ! curl -s http://localhost:11434/api/tags > /dev/null; then \
		echo "Error: Ollama is not running. Please start Ollama first:"; \
		echo "1. Open Terminal"; \
		echo "2. Run: ollama serve"; \
		echo "3. Keep the terminal window open"; \
		exit 1; \
	fi
	@echo "Ollama is running"

# Create virtual environment
setup:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created. Activate it with: source $(VENV)/bin/activate"

# Install dependencies
install: check-venv
	@echo "Installing dependencies..."
	$(VENV_BIN)/pip install -r requirements.txt
	@echo "Dependencies installed successfully"

# Pull Ollama model
pull-model: check-ollama
	@echo "Pulling Ollama model..."
	@ollama pull $(OLLAMA_MODEL) || { \
		echo "Error: Failed to pull model. Please check:"; \
		echo "1. Ollama is running (ollama serve)"; \
		echo "2. You have sufficient disk space"; \
		echo "3. You have a stable internet connection"; \
		exit 1; \
	}
	@echo "Model pulled successfully"

# Check environment setup
check-env:
	@echo "Checking environment setup..."
	@if [ ! -f "credentials.json" ]; then \
		echo "Error: credentials.json not found. Please follow these steps:"; \
		echo ""; \
		echo "1. Go to Google Cloud Console (https://console.cloud.google.com)"; \
		echo "2. Create a new project or select an existing one"; \
		echo "3. Enable the Gmail API:"; \
		echo "   - Go to 'APIs & Services' > 'Library'"; \
		echo "   - Search for 'Gmail API'"; \
		echo "   - Click 'Enable'"; \
		echo "4. Create OAuth 2.0 credentials:"; \
		echo "   - Go to 'APIs & Services' > 'Credentials'"; \
		echo "   - Click 'Create Credentials' > 'OAuth client ID'"; \
		echo "   - Choose 'Desktop app' as application type"; \
		echo "   - Give it a name (e.g., 'Gmail Classifier')"; \
		echo "   - Click 'Create'"; \
		echo "5. Download the credentials:"; \
		echo "   - Click the download icon (⬇️) next to your new OAuth client"; \
		echo "   - Save the file as 'credentials.json' in this project directory"; \
		echo ""; \
		echo "6. Configure OAuth consent screen:"; \
		echo "   - Go to 'APIs & Services' > 'OAuth consent screen'"; \
		echo "   - Choose 'External' user type"; \
		echo "   - Fill in the required fields (App name, User support email, Developer contact)"; \
		echo "   - Add the following scopes:"; \
		echo "     * .../auth/gmail.readonly"; \
		echo "   - Add your email as a test user"; \
		echo ""; \
		echo "7. Handle 'Unverified App' warning:"; \
		echo "   - When you first run the app, you'll see an 'Unverified App' warning"; \
		echo "   - Click 'Advanced'"; \
		echo "   - Click 'Go to Gmail Classifier (unsafe)'"; \
		echo "   - Click 'Continue' to grant access"; \
		echo ""; \
		echo "After completing these steps, run 'make check-env' again."; \
		exit 1; \
	fi
	@if [ ! -f ".env" ]; then \
		echo "Error: .env file not found. Creating from template..."; \
		echo "GMAIL_CREDENTIALS_FILE=credentials.json" > .env; \
		echo "OLLAMA_MODEL=$(OLLAMA_MODEL)" >> .env; \
		echo ".env file created. Please review and modify if needed."; \
	fi
	@echo "Environment check completed"

# Run the classifier
run: check-venv check-env check-ollama
	@echo "Running Gmail classifier..."
	$(VENV_BIN)/python gmail_classifier.py

# Clean up generated files
clean:
	@echo "Cleaning up..."
	rm -f token.json classified_emails.json
	@echo "Cleanup completed"

# Show help
help:
	@echo "Available commands:"
	@echo "  make setup        - Create virtual environment"
	@echo "  make install      - Install dependencies"
	@echo "  make pull-model   - Pull Ollama model"
	@echo "  make check-env    - Check environment setup"
	@echo "  make run         - Run the classifier"
	@echo "  make clean       - Clean up generated files"
	@echo "  make help        - Show this help message" 