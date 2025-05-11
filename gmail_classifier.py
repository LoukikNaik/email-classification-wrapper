import os
import json
import base64
import re
import time
from typing import List, Dict, Optional, Any, Union
from email.mime.text import MIMEText
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import EnumOutputParser
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from enum import Enum
from datetime import datetime
from google import genai
from pydantic import BaseModel, Field
import hashlib

# Load environment variables
load_dotenv()

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Valid categories
VALID_CATEGORIES = ["Work", "Personal", "Finance", "Promotions", "Travel", "Spam"]

def clean_text(text: str) -> str:
    """Clean and normalize text by removing extra whitespace and normalizing line endings."""
    if not text:
        return ""
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove leading/trailing whitespace from each line
    text = '\n'.join(line.strip() for line in text.split('\n'))
    
    # Remove leading/trailing whitespace from the entire text
    text = text.strip()
    
    return text

def get_gmail_link(message_id: str) -> str:
    """Generate a Gmail web link for the given message ID."""
    return f"https://mail.google.com/mail/u/0/#inbox/{message_id}"

# LLM Provider options
class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    GEMINI = "gemini"

importance_examples = """
Examples:
- Subject: "Assessment Test for Software Engineer Role at ABC Corp"
  Body: "You have been shortlisted for an online assessment. Please complete it by Thursday."
  → IMPORTANT

- Subject: "Building Water Maintenance Notification"
  Body: "Water will be shut off on Friday between 9AM–12PM for maintenance in your building."
  → IMPORTANT

- Subject: "Transaction Alert: ₹8,500 debited from your account"
  Body: "Your A/C xxxx1234 was debited for ₹8,500 via UPI on May 3rd."
  → IMPORTANT

- Subject: "Job Application Status Update"
  Body: "Thank you for applying. Unfortunately, we've decided not to move forward with your application."
  → IMPORTANT

- Subject: "Rent Due Reminder - May 2025"
  Body: "This is a reminder that your rent of ₹15,000 is due by the 5th of May."
  → IMPORTANT

- Subject: "Get 20% Off Your Favorite Pizza!"
  Body: "Order today and enjoy 20% off with code YUMMY20."
  → USELESS 

- Subject: "Exclusive Credit Card Offer Just for You!"
  Body: "Earn cashback and points when you sign up for our new card."
  → USELESS

- Subject: "Summer Travel Deals Inside"
  Body: "Book now and save on your next adventure!"
  → USELESS

- Subject: "Weekly Digest - Top News"
  Body: "Here are this week's trending stories..."
  → USELESS

- Subject: "Update Your Preferences"
  Body: "We'd love to hear from you. Take our quick survey!"
  → USELESS
"""

class GeminiWrapper(Runnable):
    """A wrapper for the Gemini model that implements LangChain's Runnable interface."""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash"
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum time between requests in seconds
        self.max_retries = 3
        self.retry_delay = 2.0  # Delay between retries in seconds

    @property
    def input_schema(self):
        """Define the input schema for the runnable."""
        class Input(BaseModel):
            prompt: str = Field(description="The prompt to send to the model")
        return Input

    @property
    def output_schema(self):
        """Define the output schema for the runnable."""
        class Output(BaseModel):
            text: str = Field(description="The model's response")
        return Output

    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        self.last_request_time = time.time()

    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> str:
        """Invoke the model with the given input."""
        # Handle different input types
        if isinstance(input, str):
            prompt = input
        elif hasattr(input, 'to_messages'):
            # Handle ChatPromptValue
            messages = input.to_messages()
            prompt = "\n".join(msg.content for msg in messages)
        elif isinstance(input, dict):
            # Handle dictionary input
            prompt = input.get("prompt", "")
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
        
        # Implement retry logic
        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) and attempt < self.max_retries - 1:
                    print(f"Rate limit hit, retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                raise e

    def run(self, prompt: str) -> str:
        """Run the model with a simple string prompt."""
        return self.invoke(prompt)

    def __call__(self, input: Union[str, Dict[str, Any]], config: Optional[RunnableConfig] = None) -> str:
        """Make the class callable."""
        return self.invoke(input, config)

class ImportanceLabel(str, Enum):
    IMPORTANT = "IMPORTANT"
    USELESS = "USELESS"

class GmailClassifier:
    def __init__(self, llm_provider: str = "ollama"):
        self.service = None
        self.classifier = None
        self.importance_classifier = None
        self.llm_provider = LLMProvider(llm_provider.lower())
        self.setup_gmail()
        self.setup_classifiers()
        self.setup_output_dir()
        self.setup_cache()

    def setup_output_dir(self):
        """Set up output directory for email logs."""
        self.output_dir = "email_logs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def setup_gmail(self):
        """Set up Gmail API authentication."""
        creds = None
        token_path = 'token.json'
        
        # Load existing credentials if available
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        
        # Refresh or create new credentials if needed
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    os.getenv('GMAIL_CREDENTIALS_FILE'), SCOPES)
                # Try ports 8080-8090 until we find an available one
                for port in range(8080, 8091):
                    try:
                        creds = flow.run_local_server(
                            port=port,
                            success_message='Authentication successful! You can close this window.',
                            open_browser=True
                        )
                        break
                    except OSError as e:
                        if port == 8090:  # If we've tried all ports
                            raise e
                        continue  # Try next port
            
            # Save credentials for future use
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('gmail', 'v1', credentials=creds)

    def setup_classifiers(self):
        """Set up the classifiers with the selected LLM provider."""
        if self.llm_provider == LLMProvider.OLLAMA:
            self.model = Ollama(model=os.getenv('OLLAMA_MODEL', 'phi'))
        else:  # GEMINI
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            self.model = GeminiWrapper(api_key)
        
        # Category classification prompt
        self.category_prompt = ChatPromptTemplate.from_template("""
You are an email sorting assistant. 
Classify this email into one of the following categories:
["Work", "Personal", "Finance", "Promotions", "Travel", "Spam"]

Email Subject: {subject}
Email Body: {body}

Respond only with the category name.
        """)
        
        # Importance classification prompt
        self.importance_prompt = ChatPromptTemplate.from_template("""
You are an email importance analyzer. Your task is to classify emails as either IMPORTANT or USELESS.

An email is IMPORTANT **only if** it meets one or more of these criteria:
{importance_criteria}

When evaluating the email, consider both the subject and the body. Ignore formatting, writing tone, or marketing tactics — base your decision on content relevance.

{importance_examples}

Email Subject: {subject}
Email Body: {body}

CRITICAL: You must respond with EXACTLY one word: either IMPORTANT or USELESS.
Do not include any other text, explanations, or punctuation.
Do not include "Answer:" or any other prefixes.
Do not include any reasoning or analysis.
Just the single word: IMPORTANT or USELESS.
        """)

    def setup_cache(self):
        """Set up cache directory for classified emails."""
        self.cache_dir = "email_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.cache_file = os.path.join(self.cache_dir, "classified_emails.json")
        self.load_cache()

    def load_cache(self):
        """Load the cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except json.JSONDecodeError:
                self.cache = {}
        else:
            self.cache = {}

    def save_cache(self):
        """Save the cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def get_cache_key(self, email_id: str, importance_criteria: str = "", show_categories: bool = False) -> str:
        """Generate a unique cache key for an email."""
        # Include email ID, importance criteria, and show_categories in the cache key
        key_data = f"{email_id}:{importance_criteria}:{show_categories}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get_cached_classification(self, email_id: str, importance_criteria: str = "", show_categories: bool = False) -> Optional[Dict]:
        """Get cached classification for an email if it exists."""
        cache_key = self.get_cache_key(email_id, importance_criteria, show_categories)
        return self.cache.get(cache_key)

    def cache_classification(self, email_id: str, classification: Dict, importance_criteria: str = "", show_categories: bool = False):
        """Cache the classification for an email."""
        cache_key = self.get_cache_key(email_id, importance_criteria, show_categories)
        self.cache[cache_key] = classification
        self.save_cache()

    def get_emails(self, max_results: int = 10) -> List[Dict]:
        """Fetch recent emails from Gmail."""
        results = self.service.users().messages().list(
            userId='me', maxResults=max_results).execute()
        messages = results.get('messages', [])
        
        emails = []
        for message in messages:
            msg = self.service.users().messages().get(
                userId='me', id=message['id']).execute()
            
            headers = msg['payload']['headers']
            subject = clean_text(next(h['value'] for h in headers if h['name'] == 'Subject'))
            date = clean_text(next((h['value'] for h in headers if h['name'] == 'Date'), ''))
            from_addr = clean_text(next((h['value'] for h in headers if h['name'] == 'From'), ''))
            
            # Get email body
            if 'parts' in msg['payload']:
                parts = msg['payload']['parts']
                body = ''
                for part in parts:
                    if part['mimeType'] == 'text/plain':
                        body = base64.urlsafe_b64decode(
                            part['body']['data']).decode()
                    elif part['mimeType'] == 'text/html':
                        html = base64.urlsafe_b64decode(
                            part['body']['data']).decode()
                        soup = BeautifulSoup(html, 'html.parser')
                        body = soup.get_text()
            else:
                body = base64.urlsafe_b64decode(
                    msg['payload']['body']['data']).decode()
            
            # Clean the body text
            body = clean_text(body)
            
            emails.append({
                'subject': subject,
                'body': body,
                'date': date,
                'from': from_addr,
                'id': message['id'],
                'gmail_link': get_gmail_link(message['id'])
            })
        
        return emails

    def normalize_category(self, category: str) -> str:
        """Normalize the category to ensure it's one of the valid categories."""
        category = clean_text(category)
        # If the response contains quotes, extract the category
        if '"' in category:
            category = category.split('"')[1]
        # If the response is a sentence, try to find a matching category
        for valid_cat in VALID_CATEGORIES:
            if valid_cat.lower() in category.lower():
                return valid_cat
        # Default to "Personal" if no match found
        return "Personal"

    def classify_emails(self, emails: List[Dict], importance_criteria: str = "", show_categories: bool = False) -> List[Dict]:
        """Classify emails using the selected LLM provider."""
        classified_emails = []
        
        for email in emails:
            # Check cache first
            cached_classification = self.get_cached_classification(
                email['id'], 
                importance_criteria, 
                show_categories
            )
            
            if cached_classification:
                print(f"Using cached classification for email: {email['subject']}")
                classified_emails.append(cached_classification)
                continue

            # Clean text before sending to LLM
            clean_subject = clean_text(email['subject'])
            clean_body = clean_text(email['body'][:500])  # Truncate and clean long bodies
            
            # Get category if enabled
            category = "Personal"  # Default category
            if show_categories:
                try:
                    category_chain = self.category_prompt | self.model | StrOutputParser()
                    category = category_chain.invoke({
                        "subject": clean_subject,
                        "body": clean_body
                    })
                    category = self.normalize_category(category)
                except Exception as e:
                    print(f"Error classifying category: {e}")
                    category = "Personal"
            
            # Determine importance if criteria provided
            is_important = False
            if importance_criteria:
                try:
                    importance_chain = self.importance_prompt | self.model | StrOutputParser()
                    result = importance_chain.invoke({
                        "subject": clean_subject,
                        "body": clean_body,
                        "importance_criteria": importance_criteria,
                        "importance_examples": importance_examples
                    })
                    is_important = "IMPORTANT" in result.upper()
                except Exception as e:
                    print(f"Error parsing importance result: {e}")
                    is_important = False
            
            classified_email = {
                'subject': clean_subject,
                'category': category,
                'date': clean_text(email['date']),
                'from': clean_text(email['from']),
                'id': email['id'],
                'gmail_link': email['gmail_link'],
                'is_important': is_important
            }
            
            # Cache the classification
            self.cache_classification(
                email['id'],
                classified_email,
                importance_criteria,
                show_categories
            )
            
            # Save individual email to file
            self.save_email_to_file(classified_email, clean_text(email['body']))
            
            classified_emails.append(classified_email)
        
        return classified_emails

    def save_email_to_file(self, email_info: Dict, body: str):
        """Save individual email to a file."""
        # Create a safe filename from the subject
        safe_subject = "".join(c for c in email_info['subject'] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_subject = safe_subject[:30]  # Limit length
        
        # Create filename with date and category
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{date_str}_{email_info['category']}_{safe_subject}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        # Write email content to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Subject: {email_info['subject']}\n")
            f.write(f"From: {email_info['from']}\n")
            f.write(f"Date: {email_info['date']}\n")
            f.write(f"Category: {email_info['category']}\n")
            f.write(f"Important: {email_info.get('is_important', False)}\n")
            f.write(f"ID: {email_info['id']}\n")
            f.write(f"Gmail Link: {email_info['gmail_link']}\n")
            f.write("\n" + "="*80 + "\n\n")
            f.write(body)

    def save_results(self, results: List[Dict], filename: str = 'classified_emails.json'):
        """Save classification results to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

def main():
    classifier = GmailClassifier()
    
    # Fetch emails
    print("Fetching emails...")
    emails = classifier.get_emails(max_results=10)
    
    # Classify emails
    print("Classifying emails...")
    classified_emails = classifier.classify_emails(emails)
    
    # Save results
    print("Saving results...")
    classifier.save_results(classified_emails)
    
    print(f"Done! Results saved to classified_emails.json")
    print(f"Individual emails saved in the {classifier.output_dir} directory")

if __name__ == '__main__':
    main() 