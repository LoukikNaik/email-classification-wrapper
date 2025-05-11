# Gmail Email Classifier

A powerful email classification system that uses AI to automatically categorize and prioritize your Gmail inbox. This application helps you focus on what matters by intelligently identifying important emails and organizing them into meaningful categories.

## 🎥 Live Demo

https://github.com/LoukikNaik/email-classifier-wrapper/assets/LoukikNaik/live_demo.mov

*Watch the demo to see the application in action - from email scanning to real-time classification and filtering.*

## 🌟 Key Features

### 1. Smart Email Classification
- **Importance Detection**: Automatically identifies important emails based on your custom criteria
- **Category Organization**: Classifies emails into categories like Work, Personal, Finance, etc.
- **Custom Criteria**: Define what makes an email important to you
- **Real-time Processing**: Instant classification as emails arrive

### 2. Multiple AI Providers
- **Ollama Integration**: Use local AI models for privacy-focused processing
- **Google Gemini Support**: Leverage Google's powerful AI for enhanced accuracy
- **Easy Provider Switching**: Seamlessly switch between AI providers

### 3. Advanced Features
- **Date Range Filtering**: Filter emails by specific date ranges
- **Batch Processing**: Process multiple emails simultaneously
- **Caching System**: Thread-safe caching for improved performance
- **Multithreaded Processing**: Fast parallel email classification

### 4. User-Friendly Interface
- **Modern React Frontend**: Clean and intuitive user interface
- **Real-time Updates**: Instant feedback on classification results
- **Responsive Design**: Works on desktop and mobile devices
- **Easy Navigation**: Simple and efficient email management

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Node.js 14 or higher
- Gmail account
- (Optional) Ollama installed for local AI processing
- (Optional) Google Gemini API key for cloud AI processing

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd gmail-classifier
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up frontend**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

4. **Configure environment variables**
   Create a `.env` file in the root directory:
   ```
   GMAIL_CREDENTIALS_FILE=path/to/credentials.json
   OLLAMA_MODEL=phi  # or your preferred model
   GEMINI_API_KEY=your_gemini_api_key  # if using Gemini
   ```

5. **Set up Gmail API**
   - Go to Google Cloud Console
   - Create a new project
   - Enable Gmail API
   - Create OAuth 2.0 credentials
   - Download credentials and save as specified in `.env`

### Running the Application

1. **Start the backend server**
   ```bash
   python app.py
   ```

2. **Access the application**
   Open your browser and navigate to `http://localhost:8080`

## 🔧 Configuration

### AI Provider Selection
- **Ollama**: Set `OLLAMA_MODEL` in `.env` to your preferred model
- **Gemini**: Set `GEMINI_API_KEY` in `.env` to use Google's AI

### Email Classification Settings
- **Importance Criteria**: Define what makes an email important
- **Categories**: Customize email categories
- **Date Ranges**: Set default date ranges for filtering

## 💡 Why This Matters

### Time Management
- Reduce time spent sorting through emails
- Focus on important messages first
- Automate email organization

### Productivity
- Never miss critical emails
- Reduce email overwhelm
- Better email prioritization

### Privacy
- Option to process emails locally with Ollama
- Secure Gmail API integration
- No data storage beyond caching

## 🔒 Security

- OAuth 2.0 authentication
- Secure API key handling
- Local caching with expiration
- No permanent data storage

## 🛠️ Technical Details

### Architecture
- **Backend**: Python Flask server
- **Frontend**: React with modern UI components
- **AI Integration**: LangChain for AI processing
- **Caching**: Thread-safe file-based caching

### Performance
- Multithreaded email processing
- Efficient caching system
- Optimized AI model usage
- Rate limiting and error handling

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gmail API
- LangChain
- Ollama
- Google Gemini
- React community 