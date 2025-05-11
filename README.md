# Gmail Email Classifier Agent

A Python-based agent that fetches emails from Gmail and classifies them using a local LLM (Ollama) with LangChain.

## Prerequisites

1. Python 3.8 or higher
2. Ollama installed and running locally
3. Gmail account with API access enabled
4. Make (for using Makefile commands)

## Setup

### Using Makefile (Recommended)

1. Create and activate virtual environment:
```bash
make setup
source venv/bin/activate
```

2. Install dependencies:
```bash
make install
```

3. Pull the Ollama model:
```bash
make pull-model
```

4. Set up Gmail API:
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project
   - Enable Gmail API
   - Create OAuth 2.0 credentials
   - Download the credentials and save as `credentials.json` in the project root

5. Check environment setup:
```bash
make check-env
```

### Manual Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Gmail API:
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project
   - Enable Gmail API
   - Create OAuth 2.0 credentials
   - Download the credentials and save as `credentials.json` in the project root

3. Install and run Ollama:
   - Follow instructions at [Ollama's website](https://ollama.ai)
   - Pull the model you want to use (e.g., `ollama pull llama2`)

4. Create a `.env` file:
```
GMAIL_CREDENTIALS_FILE=credentials.json
OLLAMA_MODEL=llama2
```

## Usage

### Using Makefile

Run the classifier:
```bash
make run
```

Clean up generated files:
```bash
make clean
```

Show available commands:
```bash
make help
```

### Manual Usage

Run the classifier:
```bash
python gmail_classifier.py
```

The script will:
1. Authenticate with Gmail
2. Fetch recent emails
3. Classify them using the local LLM
4. Output results to `classified_emails.json`

## Categories

Emails are classified into the following categories:
- Work
- Personal
- Finance
- Promotions
- Travel
- Spam

## Security Notes

- All processing is done locally
- Gmail access is read-only
- Credentials are stored securely
- No data is sent to external services 