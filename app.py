from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
from gmail_classifier import GmailClassifier
import os
from functools import wraps
import socket
import traceback

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)  # Required for session management

# Initialize classifier
classifier = None

def find_available_port(start_port=8080, max_port=8090):
    """Find an available port in the given range."""
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found between {start_port} and {max_port}")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return jsonify({'success': False, 'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login')
def login_page():
    if session.get('authenticated'):
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Content-Type must be application/json'}), 400

        data = request.get_json()
        llm_provider = data.get('llm_provider', 'ollama')
        
        global classifier
        classifier = GmailClassifier(llm_provider=llm_provider)
        session['authenticated'] = True
        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f"Login error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/oauth2callback')
def oauth2callback():
    """Handle the OAuth2 callback."""
    try:
        global classifier
        if not classifier:
            classifier = GmailClassifier(llm_provider='ollama')  # Default to Ollama
        session['authenticated'] = True
        return redirect(url_for('index'))
    except Exception as e:
        app.logger.error(f"OAuth callback error: {str(e)}\n{traceback.format_exc()}")
        return redirect(url_for('login_page'))

@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    global classifier
    classifier = None
    return redirect(url_for('login_page'))

@app.route('/scan_emails', methods=['POST'])
@login_required
def scan_emails():
    if not classifier:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Content-Type must be application/json'}), 400

        data = request.get_json()
        max_results = int(data.get('max_results', 10))
        importance_prompt = data.get('importance_prompt', '')
        show_categories = data.get('show_categories', False)

        # Fetch and classify emails
        emails = classifier.get_emails(max_results=max_results)
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

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'success': False, 'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = find_available_port()
    print(f"Starting server on port {port}")
    app.run(debug=True, port=port) 