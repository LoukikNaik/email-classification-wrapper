from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gmail_classifier import GmailClassifier
import os
import traceback

app = Flask(__name__, static_folder='frontend/build')
CORS(app)

# Initialize classifier
classifier = None

def init_classifier(provider='ollama'):
    global classifier
    if classifier is None:
        classifier = GmailClassifier(llm_provider=provider)

@app.route('/api/auth/check', methods=['GET'])
def check_auth():
    return jsonify({'authenticated': classifier is not None})

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        provider = data.get('provider', 'ollama')
        init_classifier(provider)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    global classifier
    classifier = None
    return jsonify({'success': True})

@app.route('/api/scan_emails', methods=['POST'])
def scan_emails():
    if not classifier:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Content-Type must be application/json'}), 400

        data = request.get_json()
        max_results = data.get('max_results')
        # Handle max_results whether it's a string or integer
        if isinstance(max_results, str) and max_results.strip():
            max_results = int(max_results)
        elif not max_results:  # None, empty string, or 0
            max_results = 1000
        importance_prompt = data.get('importance_prompt', '')
        show_categories = data.get('show_categories', False)
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        # Fetch and classify emails
        emails = classifier.get_emails(
            max_results=max_results,
            start_date=start_date,
            end_date=end_date
        )
        classified_emails = classifier.classify_emails(
            emails, 
            importance_criteria=importance_prompt,
            show_categories=show_categories
        )

        # Separate important and non-important emails
        important_emails = [email for email in classified_emails if email.get('is_important', False)]
        non_important_emails = [email for email in classified_emails if not email.get('is_important', False)]

        return jsonify({
            'success': True,
            'important_emails': important_emails,
            'non_important_emails': non_important_emails
        })
    except Exception as e:
        app.logger.error(f"Scan emails error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    if not classifier:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    try:
        classifier.clear_cache()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080) 