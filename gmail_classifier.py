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
from datetime import datetime, timedelta
from google import genai
from pydantic import BaseModel, Field
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

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

- Subject: "30+ new jobs for "intern"
  Body: "Check out the latest opportunities for interns!"
  → USELESS

- Subject: "30+ new jobs for "intern"
  Body: "Check out the latest opportunities for interns!"
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
        self.cache_expiry = timedelta(hours=24)  # Cache expires after 24 hours

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

    def _get_cache_path(self, email_id: str) -> str:
        """Get the cache file path for an email."""
        return os.path.join(self.cache_dir, f"{email_id}.json")

    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if the cache is still valid."""
        try:
            if not os.path.exists(cache_path):
                return False
            
            # Check if cache is expired
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            return datetime.now() - cache_time < self.cache_expiry
        except Exception as e:
            logging.error(f"Error checking cache validity: {str(e)}")
            return False

    def _save_to_cache(self, email_id: str, classification_data: Dict):
        """Save classification data to cache in a thread-safe manner."""
        cache_path = self._get_cache_path(email_id)
        temp_path = f"{cache_path}.tmp"
        try:
            # First write to a temporary file
            with open(temp_path, 'w') as f:
                json.dump(classification_data, f, indent=2)
            # Then atomically rename the temp file to the final file
            os.replace(temp_path, cache_path)
            logging.info(f"Cached classification for email {email_id}")
        except Exception as e:
            logging.error(f"Error saving cache for email {email_id}: {str(e)}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    def _load_from_cache(self, email_id: str) -> Optional[Dict]:
        """Load classification data from cache if it exists and is valid."""
        cache_path = self._get_cache_path(email_id)
        if not self._is_cache_valid(cache_path):
            return None

        try:
            # Use a file lock to prevent concurrent reads/writes
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error loading cache for email {email_id}: {str(e)}")
            return None

    def get_emails(self, max_results: int = 10, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        """Fetch emails from Gmail within a date range.
        
        Args:
            max_results: Maximum number of emails to fetch
            start_date: Start date in YYYY/MM/DD format (inclusive)
            end_date: End date in YYYY/MM/DD format (inclusive)
        """
        # Build the query
        query_parts = []
        
        if start_date:
            # Convert YYYY/MM/DD to YYYY/MM/DD format for Gmail
            start_date = start_date.replace('/', '/')
            query_parts.append(f'after:{start_date}')
        
        if end_date:
            # Convert YYYY/MM/DD to YYYY/MM/DD format for Gmail
            end_date = end_date.replace('/', '/')
            query_parts.append(f'before:{end_date}')
        
        # Combine query parts
        query = ' '.join(query_parts) if query_parts else None
        
        # Fetch messages
        results = self.service.users().messages().list(
            userId='me',
            maxResults=max_results,
            q=query
        ).execute()
        
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
        if not category:
            return "Other"
            
        category = clean_text(category)
        # Remove any quotes or extra text
        category = category.strip('"\'')
        
        # Try exact match first
        if category in VALID_CATEGORIES:
            return category
            
        # Try case-insensitive match
        category_lower = category.lower()
        for valid_cat in VALID_CATEGORIES:
            if valid_cat.lower() == category_lower:
                return valid_cat
                
        # Try partial match
        for valid_cat in VALID_CATEGORIES:
            if valid_cat.lower() in category_lower:
                return valid_cat
                
        # Default to Other if no match found
        logging.warning(f"Could not normalize category: {category}, defaulting to Other")
        return "Other"

    def classify_emails(self, emails: List[Dict[str, Any]], importance_criteria: str = '', show_categories: bool = False) -> List[Dict[str, Any]]:
        """Classify emails using the configured LLM provider."""
        if not emails:
            return []

        # Process emails in parallel
        with ThreadPoolExecutor(max_workers=min(10, len(emails))) as executor:
            # Submit all email classification tasks
            future_to_email = {
                executor.submit(self._classify_single_email, email, importance_criteria, show_categories): email 
                for email in emails
            }
            
            # Collect results as they complete
            classified_emails = []
            for future in as_completed(future_to_email):
                try:
                    result = future.result()
                    if result:
                        # Ensure is_important is a boolean
                        result['is_important'] = bool(result.get('is_important', False))
                        classified_emails.append(result)
                except Exception as e:
                    logging.error(f"Error classifying email: {str(e)}")
                    # Add the original email with error status
                    email = future_to_email[future]
                    email['error'] = str(e)
                    email['is_important'] = False  # Default to not important on error
                    classified_emails.append(email)
            
            return classified_emails

    def _classify_single_email(self, email: Dict[str, Any], importance_criteria: str, show_categories: bool) -> Dict[str, Any]:
        """Classify a single email using the configured LLM provider."""
        try:
            email_id = email.get('id')
            if not email_id:
                raise ValueError("Email ID is required for classification")

            # Try to get from cache first
            cached_data = self._load_from_cache(email_id)
            if cached_data:
                return cached_data

            # Clean the email content
            clean_subject = self.clean_text(email.get('subject', ''))
            clean_body = self.clean_text(email.get('body', ''))
            
            # Get category if enabled
            category = "Other"  # Default category
            if show_categories:
                try:
                    category_chain = self.category_prompt | self.model | StrOutputParser()
                    category = category_chain.invoke({
                        "subject": clean_subject,
                        "body": clean_body
                    })
                    category = self.normalize_category(category)
                except Exception as e:
                    logging.error(f"Error classifying category: {e}")
                    category = "Other"
            
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
                    logging.error(f"Error parsing importance result: {e}")
                    is_important = False

            # Add classification results to email
            email['is_important'] = is_important
            if show_categories:
                email['category'] = category

            # Save to cache
            self._save_to_cache(email_id, email)
            return email

        except Exception as e:
            logging.error(f"Error in _classify_single_email: {str(e)}")
            email['error'] = str(e)
            email['is_important'] = False
            return email

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = ''.join(c for c in text if c.isprintable())
        return text.strip()

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

    def clear_cache(self):
        """Clear all cached classifications in a thread-safe manner."""
        if not os.path.exists(self.cache_dir):
            return

        try:
            # Get list of files first to avoid modification during iteration
            cache_files = list(os.listdir(self.cache_dir))
            for file in cache_files:
                try:
                    file_path = os.path.join(self.cache_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logging.error(f"Error removing cache file {file}: {str(e)}")
            logging.info("Cache cleared successfully")
        except Exception as e:
            logging.error(f"Error clearing cache: {str(e)}")

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