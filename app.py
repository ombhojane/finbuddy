from flask import Flask, request, render_template, flash, redirect, url_for, Response, send_from_directory, jsonify, session
from google.oauth2 import id_token  # Import id_token for Google OAuth2
import os, uuid, time
from functools import wraps
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import re
import pdfplumber
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from fpdf import FPDF
from objective import ObjectiveTest
from subjective import SubjectiveTest
import cv2
import mediapipe as mp
import numpy as np
from collections import Counter
import threading
import random
import shutil
from datetime import datetime
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from urllib.parse import quote, unquote
from tinydb import TinyDB, Query
from gtts import gTTS
import json
import pandas as pd
import wikipedia
from duckduckgo_search import DDGS
import requests
from typing import Optional, Dict, Any, List
import xml.etree.ElementTree as ET
import json
from typing import Union
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


from google import genai
from google.genai import types

import firebase_admin
from firebase_admin import credentials, auth

# Initialize Firebase Admin SDK
cred = credentials.Certificate("finbuddy-141ea-firebase-adminsdk-fbsvc-20a9da85e3.json")  # Replace with your Firebase Admin SDK JSON file
firebase_admin.initialize_app(cred)

app = Flask(__name__)
Bootstrap(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///notes.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FOLDERS'] = 'folders'
tdb = TinyDB('data/query_history.json')
# Create folders if they do not exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FOLDERS'], exist_ok=True)
os.makedirs('data', exist_ok=True)
db = SQLAlchemy(app)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

################ Initialize Groq client #########
groq_client = Groq(api_key="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")
########## End #################

# Indian Stock Market API base configuration
INDIAN_API_KEY = os.environ.get('FINANCE_KEY')  # Default to the provided key if not set
INDIAN_API_BASE_URL = "https://stock.indianapi.in"

# Google Gemini API configuration
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
client = genai.Client(api_key=GOOGLE_API_KEY)

# Define API endpoints and their parameters
API_ENDPOINTS = {
    "get_stock_details": {
        "endpoint": "/stock",
        "required_params": ["stock_name"],
        "param_mapping": {"stock_name": "name"},
        "description": "Get details for a specific stock"
    },
    "get_trending_stocks": {
        "endpoint": "/trending",
        "required_params": [],
        "param_mapping": {},
        "description": "Get trending stocks in the market"
    },
    "get_market_news": {
        "endpoint": "/news",
        "required_params": [],
        "param_mapping": {},
        "description": "Get latest stock market news"
    },
    "get_mutual_funds": {
        "endpoint": "/mutual_funds",
        "required_params": [],
        "param_mapping": {},
        "description": "Get mutual funds data"
    },
    "get_ipo_data": {
        "endpoint": "/ipo",
        "required_params": [],
        "param_mapping": {},
        "description": "Get IPO data"
    },
    "get_bse_most_active": {
        "endpoint": "/BSE_most_active",
        "required_params": [],
        "param_mapping": {},
        "description": "Get BSE most active stocks"
    },
    "get_nse_most_active": {
        "endpoint": "/NSE_most_active",
        "required_params": [],
        "param_mapping": {},
        "description": "Get NSE most active stocks"
    },
    "get_historical_data": {
        "endpoint": "/historical_data",
        "required_params": ["stock_name"],
        "optional_params": ["period"],
        "default_values": {"period": "1m", "filter": "default"},
        "param_mapping": {},
        "description": "Get historical data for a stock"
    }
}

# Unified API call function
def call_indian_api(endpoint, params=None):
    """
    Generic function to call the Indian Stock Market API
    
    Args:
        endpoint: API endpoint suffix (e.g., '/stock', '/trending')
        params: Optional parameters for the API call
        
    Returns:
        JSON response from the API
    """
    url = f"{INDIAN_API_BASE_URL}{endpoint}"
    headers = {"X-Api-Key": INDIAN_API_KEY}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Function to call API by name
def call_api_by_name(api_name, **kwargs):
    """
    Call an API by its name from the API_ENDPOINTS dictionary
    
    Args:
        api_name: Name of the API to call (key in API_ENDPOINTS)
        **kwargs: Parameters to pass to the API
        
    Returns:
        JSON response from the API
    """
    if api_name not in API_ENDPOINTS:
        return {"error": f"Unknown API: {api_name}"}
    
    api_info = API_ENDPOINTS[api_name]
    endpoint = api_info["endpoint"]
    
    # Check required parameters
    for param in api_info.get("required_params", []):
        if param not in kwargs:
            return {"error": f"Missing required parameter: {param}"}
    
    # Apply parameter mapping
    mapped_params = {}
    for param, value in kwargs.items():
        mapped_name = api_info.get("param_mapping", {}).get(param, param)
        mapped_params[mapped_name] = value
    
    # Apply default values
    for param, value in api_info.get("default_values", {}).items():
        if param not in mapped_params:
            mapped_params[param] = value
    
    return call_indian_api(endpoint, mapped_params)

# Financial assistant system prompt
SYSTEM_PROMPT = """You are FinBuddy, a helpful and knowledgeable financial assistant designed specifically for Indian users. 
Your purpose is to improve financial literacy and provide guidance on investments in the Indian market.

Key responsibilities:
1. Explain financial concepts in simple, easy-to-understand language
2. Provide information about different investment options available in India (stocks, mutual funds, bonds, PPF, FDs, etc.)
3. Help users understand investment risks and returns
4. Explain tax implications of different investments in the Indian context
5. Guide users on how to start investing based on their goals and risk tolerance
6. Answer questions about market trends and financial news in India
"""

# Improved orchestrator function
def orchestrator(query):
    """
    Determines if the query requires market data and which API to call
    Returns: (needs_api, api_function, params)
    """
    
    # Create a more precise prompt for the orchestrator
    orchestrator_prompt = """
    You are an orchestrator for a financial assistant specialized in Indian markets. Your job is to analyze user queries and determine if they need real-time market data.
    IMPORTANT: Be very precise in your analysis. Only return TRUE for "needs_api" when the query EXPLICITLY asks for current market data, stock prices, or listings.
    Examples where needs_api should be TRUE:
    - "Show me the most active stocks on NSE today" → get_nse_most_active
    - "What is the current price of Reliance?" → get_stock_details with stock_name="Reliance"
    - "Tell me about trending stocks" → get_trending_stocks
    - "What are the latest IPOs?" → get_ipo_data
    Examples where needs_api should be FALSE:
    - "What is compound interest?"
    - "How should I start investing?"
    - "What are the tax benefits of PPF?"
    - "Explain mutual funds to me"
    Available API functions:
    - get_stock_details(stock_name): Get details for a specific stock
    - get_trending_stocks(): Get trending stocks in the market
    - get_market_news(): Get latest stock market news
    - get_mutual_funds(): Get mutual funds data
    - get_ipo_data(): Get IPO data
    - get_bse_most_active(): Get BSE most active stocks
    - get_nse_most_active(): Get NSE most active stocks
    - get_historical_data(stock_name, period="1m"): Get historical data for a stock
    User query: """ + query + """
    Respond in JSON format with the following structure:
    {
        "needs_api": true/false,
        "function": "function_name_if_needed",
        "params": {
            "param1": "value1",
            "param2": "value2"
        }
    }
    """
    
    try:
        # Check if the Gemini API key is set
        # Create content for the orchestrator
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=orchestrator_prompt)
                ],
            ),
        ]
        
        # Configure generation parameters
        generate_content_config = types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_output_tokens=500,
            response_mime_type="text/plain",
        )
        
        # Generate content
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=generate_content_config,
            )
            
            # Parse the response
            decision_text = response.text
            # Extract JSON from the response
            if "```json" in decision_text:
                json_str = decision_text.split("```json")[1].split("```")[0].strip()
            elif "```" in decision_text:
                json_str = decision_text.split("```")[1].strip()
            else:
                json_str = decision_text.strip()
            
            try:
                decision = json.loads(json_str)
                return decision
            except json.JSONDecodeError as json_error:
                print(f"Error parsing JSON from orchestrator: {json_error}")
                print(f"Raw response: {decision_text}")
                return {"needs_api": False}
        
        except Exception as model_error:
            print(f"Error in orchestrator model call: {model_error}")
            return {"needs_api": False}
    
    except Exception as e:
        print(f"Error in orchestrator: {e}")
        return {"needs_api": False}

def get_gemini_response(user_query, conversation_history=""):
    """
    Get a response from the Gemini model
    
    Args:
        user_query: User's query
        conversation_history: Previous conversation history
        
    Returns:
        Response from the Gemini model
    """
    
    try:
        # Check if the Gemini API key is set
            
        # First, use the orchestrator to determine if we need to call an API
        decision = orchestrator(user_query)
        
        # If we need to call an API, do so and add the result to the context
        api_context = ""
        if decision.get("needs_api", False):
            function_name = decision.get("function", "")
            params = decision.get("params", {})
            
            if function_name in API_ENDPOINTS:
                try:
                    api_result = call_api_by_name(function_name, **params)
                    if "error" in api_result:
                        api_context = f"\nI attempted to fetch market data but encountered an error: {api_result['error']}"
                    else:
                        api_context = f"\nHere is the real-time market data from the Indian Stock Market API:\n{json.dumps(api_result, indent=2)}\n\nPlease use this data to provide an informative response to the user's query."
                except Exception as api_error:
                    api_context = f"\nI attempted to fetch market data but encountered an error: {str(api_error)}"
        
        # Prepare the user query with API context if available
        query_with_context = user_query
        if api_context:
            query_with_context = f"{user_query}\n\n[SYSTEM NOTE: {api_context}]"
        
        # Prepare the system message
        system_message = SYSTEM_PROMPT
        if conversation_history:
            system_message += f"\n\nPrevious conversation:\n{conversation_history}"
        
        # Create content for the LLM
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=system_message)
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(text="I understand my role as FinBuddy, a financial assistant for Indian users. I'll provide helpful information about investing and financial planning in simple language.")
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=query_with_context)
                ],
            ),
        ]
        
        # Configure generation parameters
        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="text/plain",
        )
        
        # Generate the response
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=contents,
                config=generate_content_config,
            )
            return response.text
        except Exception as model_error:
            print(f"Error generating content: {model_error}")
            # Check for specific error types
            error_message = str(model_error).lower()
            if "rate limit" in error_message:
                return "I'm sorry, we've hit the rate limit for our AI service. Please try again in a moment."
            elif "quota" in error_message:
                return "I'm sorry, we've exceeded our quota for the AI service. Please try again later."
            elif "permission" in error_message or "credential" in error_message:
                return "I'm sorry, there's an authentication issue with our AI service. Please contact support."
            else:
                return "I apologize, but I encountered an error while generating a response. Please try again later."
                
    except Exception as e:
        print(f"Error in Gemini response: {e}")
        # Fallback to Gemma if available
        try:
            return get_gemma_response_fallback(user_query)
        except:
            return "I apologize, but I encountered an error while processing your request. Please try again later."

# Fallback to Gemma if Gemini is not available
def get_gemma_response_fallback(query):
    """Fallback to use Groq's Gemma model when Gemini is not available"""
    try:
        system_prompt = """
        You are FinBuddy, a helpful and knowledgeable financial assistant designed specifically for Indian users. 
        Your purpose is to improve financial literacy and provide guidance on investments in the Indian market.
        """
        
        completion = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

@app.route('/chatbot')
def chatbot():
    # Check if user is logged in
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('signin'))
    return render_template('chatbot.html', active_page='chatbot')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', active_page='dashboard')

@app.route('/ask', methods=['POST'])
def ask():
    """
    Handle chat requests from both main chatbot and sidebar chatbot
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    query = request.json.get('query', '')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Get conversation history from session
    conversation_history = ""
    if 'chat_history' in session:
        history = session.get('chat_history', [])
        for i, entry in enumerate(history[-10:]):  # Get last 10 entries for context
            role = "User" if i % 2 == 0 else "Assistant"
            conversation_history += f"{role}: {entry}\n"
    
    # Call get_gemini_response function with the query and conversation history
    response = get_gemini_response(query, conversation_history)
    
    # Update session with new conversation entries
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    chat_history = session['chat_history']
    chat_history.append(query)
    chat_history.append(response)
    session['chat_history'] = chat_history
    
    return jsonify({"response": response})

#aptitude
AUDIO_DIR = os.path.join('static', 'audio')
os.makedirs(AUDIO_DIR, exist_ok=True)
TOPICS = ["Weather", "Family", "Technology", "Education", "Travel", "Food", "Sports", "Music"]
MAX_ATTEMPTS = 5

QUIZ_FILES = {
    'investor': 'investor.xlsx',
    'trader': 'trader.xlsx',
}

QUIZ_CONFIG = {
    'hard': {'questions': 60, 'time': 60},  # 60 minutes
    'medium': {'questions': 30, 'time': 30},  # 30 minutes
    'low': {'questions': 30, 'time': 60},  # 60 minutes
}


# Create prompt templates - replace with string templates
generation_prompt_template = "Generate a paragraph based on this topic: {topic}"
feedback_prompt_template = """
Evaluate the following transcription accuracy:
Original Paragraph: {original_paragraph}
Transcribed Text: {transcribed_text}
Score: {score}/5

Provide brief feedback about the accuracy of the transcription and pronunciation.
"""
#essay
topic_prompt_template = (
    "Generate an interesting writing topic that would make for a good 250-word essay. "
    "The topic should be specific enough to be focused but broad enough to allow for development. "
    "Return only the topic without any additional text or explanation."
)

evaluation_prompt_template = """
Evaluate the following 250-word essay:

Topic: {topic}
Essay: {essay}

Provide a rating on a scale of 1-5 (where 5 is excellent) based on the following criteria:
- Relevance to the topic
- Organization and structure
- Development of ideas
- Language usage and clarity
- Overall quality

First provide the numerical score as an integer between 1 and 5, then provide a brief constructive feedback explaining the rating.
Format your response exactly as:
SCORE: [number]
FEEDBACK: [your feedback]
"""

generation_Lprompt_template = "Generate a single line sentence based on this topic: {topic}"
feedback_Lprompt_template = """
Evaluate the following transcription accuracy:
Original Paragraph: {original_paragraph}
Transcribed Text: {transcribed_text}
Score: {score}/5
Provide brief feedback about the accuracy of the transcription and pronunciation.
"""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_file_path):
    with pdfplumber.open(pdf_file_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            full_text += page_text + "\n\n"
    return full_text

def remove_special_characters(text):
    return re.sub(r'[^A-Za-z0-9\s\n]', '', text)

# LLM logic for generating objective questions
question_template = """
Please generate {num_questions} 
quiz questions and options from the following input text. Each question should have four options.\n\n
Question 1: 
(A) Option 1
(B) Option 2
(C) Option 3
(D) Option 4
(Correct)
{input_text}


"""

question_prompt_template = question_template

# Function to generate questions with Groq
def generate_questions_with_groq(template, input_data):
    system_prompt = "You are a helpful AI assistant that generates quiz questions from text."
    formatted_prompt = template.format(**input_data)
    
    completion = groq_client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt}
        ],
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
    )
    
    return completion.choices[0].message.content

# Use this function instead of directly piping through model
def question_chain_invoke(data):
    return generate_questions_with_groq(question_template, data)

def parse_input(input_text):
    lines = input_text.strip().split("\n")
    questions = []
    current_question = ""
    options = []
    
    for line in lines:
        line = line.strip()
        if line.startswith("Question"):
            if current_question:
                questions.append((current_question, options))
            current_question = line
            options = []
        elif re.match(r'^[A-D]\s', line) or re.match(r'^[a-d]\s', line):
            options.append(line)  # Recognizes options that start with "A ", "B ", etc.
        elif line and not (line.startswith("Question")):
            current_question += " " + line
    
    # Add the last question
    if current_question:
        questions.append((current_question, options))
    
    return questions


@app.route('/')
def landingpage():
    return render_template('landingpage.html', active_page='home')

@app.route('/signin')
def signin():
    # If user is already logged in, redirect to chatbot
    if 'user_id' in session:
        return redirect(url_for('chatbot'))
    return render_template('signin.html', active_page='home')

@app.route('/auth', methods=['POST'])
def authenticate():
    try:
        # Get the ID token from the request
        id_token = request.json.get('idToken')
        if not id_token:
            return jsonify({'error': 'No ID token provided'}), 400
            
        # Verify the ID token
        decoded_token = auth.verify_id_token(id_token)
        
        # Get user info
        user_id = decoded_token['uid']
        email = decoded_token.get('email', '')
        name = decoded_token.get('name', '')
        
        # If name is not in token, try to get it from Firebase Auth
        if not name:
            try:
                user = auth.get_user(user_id)
                name = user.display_name or ''
            except:
                name = email.split('@')[0]  # Use part of email as name if nothing else available
        
        # Store user data in session
        session['user_id'] = user_id
        session['email'] = email
        session['name'] = name
        
        return jsonify({
            'success': True,
            'redirect': url_for('chatbot'),
            'user': {
                'id': user_id,
                'email': email,
                'name': name
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 401

@app.route('/logout')
def logout():
    # Clear the session
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('landingpage'))

@app.route('/dasboard')
def index():
    return render_template('index.html', active_page='home')

@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

@app.route('/Qna')
def Qna():
    return render_template('Qna.html', active_page='Qna')

@app.route('/assessment')
def assessment():
    return render_template('assessment.html',active_page='assessment')

@app.route('/pdfmanage')
def pdfmanage():
    folders = os.listdir(app.config['FOLDERS'])
    pdf_files = {}
    subfolders = {}
    pdf_percentages = {}

    for folder in folders:
        folder_path = os.path.join(app.config['FOLDERS'], folder)
        if os.path.isdir(folder_path):
            subfolders[folder] = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            pdf_files[folder] = {subfolder: [f for f in os.listdir(os.path.join(folder_path, subfolder)) if f.endswith('.pdf')] for subfolder in subfolders[folder]}
            pdf_files[folder][''] = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.pdf')]

            # Read scroll percentage from the data directory
            for subfolder in pdf_files[folder].keys():
                for pdf in pdf_files[folder][subfolder]:
                    pdf_info_path = os.path.join('data', f'{folder}_{subfolder}_{pdf}.txt')
                    if os.path.exists(pdf_info_path):
                        with open(pdf_info_path, 'r') as file:
                            pdf_percentages[f'{folder}/{subfolder}/{pdf}'] = file.read().strip()
                    else:
                        pdf_percentages[f'{folder}/{subfolder}/{pdf}'] = '0'

    if 'your_notes' not in folders:
        os.makedirs(os.path.join(app.config['FOLDERS'], 'your_notes'), exist_ok=True)

    return render_template('pdfmanage.html', folders=folders, pdf_files=pdf_files, subfolders=subfolders, pdf_percentages=pdf_percentages,active_page='pdfmanage')

@app.route('/reading')
def reading():
    return render_template('reading.html')

@app.route('/listening')
def listening():
    return render_template('listening.html')

@app.route('/writing')
def writing():
    return render_template('writing.html')

@app.route('/DSV')
def DSV():
    return render_template('DSV.html')

class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    color = db.Column(db.String(50), default='yellow')  # yellow, pink, purple, grey
    note_type = db.Column(db.String(50))  # YouTube Video Notes, Tutorial Notes, Scripts, etc.
    due_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Financial Literacy Course Models
class CourseModule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    order = db.Column(db.Integer, nullable=False)
    image = db.Column(db.String(200))
    sections = db.relationship('CourseSection', backref='module', lazy=True)
    
class CourseSection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    order = db.Column(db.Integer, nullable=False)
    module_id = db.Column(db.Integer, db.ForeignKey('course_module.id'), nullable=False)
    
class UserProgress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(128), nullable=False)
    section_id = db.Column(db.Integer, db.ForeignKey('course_section.id'), nullable=False)
    completed = db.Column(db.Boolean, default=False)
    completed_at = db.Column(db.DateTime)
    
    # Create a unique constraint to ensure one entry per user per section
    __table_args__ = (db.UniqueConstraint('user_id', 'section_id'),)

@app.route('/notes')
@app.route('/notes/<path:note_type>')
def notes(note_type=None):
    if note_type and note_type != 'all':
        # URL decode the note_type parameter
        decoded_type = unquote(note_type)
        notes = Note.query.filter_by(note_type=decoded_type).order_by(Note.created_at.desc()).all()
    else:
        notes = Note.query.order_by(Note.created_at.desc()).all()
    
    upcoming_notes = Note.query.filter(Note.due_date != None).order_by(Note.due_date.asc()).limit(5).all()
    return render_template('notes.html', 
                         notes=notes, 
                         upcoming_notes=upcoming_notes, active_page='notes',
                         current_type=unquote(note_type) if note_type else 'all')

@app.route('/note/new', methods=['POST'])
def new_note():
    title = request.form.get('title')
    content = request.form.get('content')
    color = request.form.get('color', 'yellow')
    note_type = request.form.get('note_type')
    due_date_str = request.form.get('due_date')
    
    due_date = None
    if due_date_str:
        try:
            due_date = datetime.strptime(due_date_str, '%Y-%m-%d')
        except:
            pass
    
    note = Note(
        title=title,
        content=content,
        color=color,
        note_type=note_type,
        due_date=due_date
    )
    db.session.add(note)
    db.session.commit()
    return redirect(url_for('notes'))

@app.route('/note/<int:note_id>/delete', methods=['POST'])
def delete_note(note_id):
    note = Note.query.get_or_404(note_id)
    db.session.delete(note)
    db.session.commit()
    return redirect(url_for('notes'))

def process_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page_num in range(len(pdf_reader.pages)):
        page_text = pdf_reader.pages[page_num].extract_text()
        text += page_text
    return text

@app.route('/test_generate', methods=["POST"])
def test_generate():
    if request.method == "POST":
        # Handle text input
        inputText = request.form.get("itext", "").strip()
        testType = request.form.get("test_type")
        noOfQues = request.form.get("noq")
        processing_method = request.form.get("processing_method")  # Get processing method

        # Handle file upload
        pdf_file = request.files.get("pdf_file")
        if pdf_file and pdf_file.filename:
            try:
                filename = secure_filename(pdf_file.filename)
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                pdf_file.save(pdf_path)
                inputText = extract_text_from_pdf(pdf_path)
                print(f"Extracted {len(inputText)} characters from PDF")
            except Exception as e:
                print(f"Error processing PDF: {str(e)}")
                flash(f"Error processing PDF: {str(e)}", "danger")
                return redirect(url_for('Qna'))
        
        # Validate input
        if not inputText or len(inputText.strip()) < 20:
            flash("Please provide more text content (at least 20 characters) to generate flashcards.", "warning")
            return redirect(url_for('Qna'))

        # Make sure noOfQues is valid
        try:
            noOfQues = int(noOfQues)
            if noOfQues < 1:
                noOfQues = 5  # Default to 5 questions if invalid
        except (ValueError, TypeError):
            noOfQues = 5  # Default to 5 questions if parsing fails
            
        # Use raw input text directly - no cleaning or filtering
        final_filtered_text = inputText

        if testType == "objective":
            if processing_method == "nlp":
                objective_generator = ObjectiveTest(final_filtered_text, noOfQues)
                mcqs = objective_generator.generate_mcqs()
                return render_template('objective.html', mcqs=mcqs)

            elif processing_method == "llm":
                # Use the new Groq-based question generator
                response = question_chain_invoke({
                    "num_questions": noOfQues, 
                    "input_text": final_filtered_text
                })
                mcq_text = response
                cleaned_mcq_text = remove_special_characters(mcq_text)

                questions = parse_input(cleaned_mcq_text)  # Make sure this is correct
                return render_template('mcqllm.html', questions=questions)

        elif testType == "subjective":
            if processing_method == "nlp":
                subjective_generator = SubjectiveTest(final_filtered_text, noOfQues)
                question_list, answer_list = subjective_generator.generate_test()
                testgenerate = zip(question_list, answer_list)
                return render_template('subjective.html', cresults=testgenerate)
            
            elif processing_method == "llm":
                # Determine source type
                source_type = "PDF" if pdf_file else "Text Input"
                content_length = len(final_filtered_text) if final_filtered_text else 0
                
                # If no content, show error
                if not final_filtered_text or content_length < 10:
                    flash('Please provide more text content to generate flashcards.', 'warning')
                    return redirect(url_for('Qna'))
                
                # Generate Q&A using Groq's Gemma model
                system_prompt = "You are an AI assistant designed to generate educational flashcards based on the provided content."
                prompt_content = f"""
                You are an AI assistant designed to generate educational flashcards based on the provided content. Your task is to create relevant and insightful questions that test the understanding of the text, followed by accurate and concise answers.

                Content: 
                {final_filtered_text}

                Now, generate {noOfQues} flashcards following these guidelines:
                1. Each flashcard should have a question that relates to a key point in the content
                2. Each answer should be concise, informative, and directly address the question
                3. Use clear, straightforward language
                4. Make sure questions cover different aspects of the content

                Format each flashcard exactly like this:
                Question X: [Your question here]
                Answer X: [Your answer here]

                Where X is the flashcard number (1, 2, 3, etc.)
                """
                
                try:
                    completion = groq_client.chat.completions.create(
                        model="gemma2-9b-it",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt_content}
                        ],
                        temperature=0.7,
                        max_tokens=2048,
                        top_p=1,
                    )
                    
                    qna_text = completion.choices[0].message.content
                    
                    # Don't strip all special characters, just clean up a bit
                    qna_text_cleaned = qna_text.replace('"', '').replace('"', '')
                except Exception as e:
                    print(f"Error calling Groq API: {str(e)}")
                    flash(f"Error generating flashcards: {str(e)}", "danger")
                    return redirect(url_for('Qna'))

                # Prepare Q&A data for rendering
                qna_items = []
                qna_lines = qna_text_cleaned.split('\n')

                # Pairing question and answer in the same object
                current_question = None
                for line in qna_lines:
                    line = line.strip()
                    if line.startswith("Question"):
                        current_question = {'type': 'qa', 'question': line, 'answer': None}
                        qna_items.append(current_question)
                    elif line.startswith("Answer") and current_question:
                        current_question['answer'] = line

                # Debug prints
                print(f"Input text length: {len(inputText)}")
                print(f"Final text length: {len(final_filtered_text)}")
                print(f"Number of QnA items: {len(qna_items)}")
                print(f"Content source: {source_type}")
                print(f"Content length: {content_length}")

                # Check for empty response and retry if needed
                if not qna_items:
                    print("No QnA items found in response, retrying with simpler prompt...")
                    # Simplified prompt for retry
                    prompt_content = f"""
                    Based on this content: 
                    {final_filtered_text}

                    Generate {noOfQues} simple question and answer pairs in this exact format:
                    Question 1: [Question]
                    Answer 1: [Answer]
                    Question 2: [Question]
                    Answer 2: [Answer]
                    """
                    
                    # Retry with simpler prompt
                    try:
                        completion = groq_client.chat.completions.create(
                            model="gemma2-9b-it",
                            messages=[
                                {"role": "system", "content": "Generate simple question-answer pairs."},
                                {"role": "user", "content": prompt_content}
                            ],
                            temperature=0.7,
                            max_tokens=2048,
                            top_p=1,
                        )
                        
                        qna_text = completion.choices[0].message.content
                        qna_text_cleaned = qna_text.replace('"', '').replace('"', '')
                    except Exception as e:
                        print(f"Error in retry attempt: {str(e)}")
                        # Create a default item instead of showing an error
                        summary = final_filtered_text[:200] + "..." if len(final_filtered_text) > 200 else final_filtered_text
                        qna_items = [
                            {
                                'type': 'qa',
                                'question': 'Question 1: What is the main topic of this text?',
                                'answer': f'Answer 1: The text appears to discuss {summary}'
                            }
                        ]
                        return render_template('qna_template.html', 
                                             qna_items=qna_items, 
                                             source_type=source_type,
                                             content_length=content_length)
                    
                    # Process the response again
                    qna_lines = qna_text_cleaned.split('\n')
                    for line in qna_lines:
                        line = line.strip()
                        if line.startswith("Question"):
                            current_question = {'type': 'qa', 'question': line, 'answer': None}
                            qna_items.append(current_question)
                        elif line.startswith("Answer") and current_question:
                            current_question['answer'] = line
                    
                    print(f"Retry resulted in {len(qna_items)} QnA items")
                    
                    # If still no items, create a default set
                    if not qna_items:
                        print("Still no QnA items, creating default set")
                        # Create at least one item to prevent empty display
                        summary = final_filtered_text[:200] + "..." if len(final_filtered_text) > 200 else final_filtered_text
                        qna_items = [
                            {
                                'type': 'qa',
                                'question': 'Question 1: What is the main topic of this text?',
                                'answer': f'Answer 1: The text appears to discuss {summary}'
                            }
                        ]

                # Render HTML template with Q&A data
                return render_template('qna_template.html', 
                                      qna_items=qna_items, 
                                      source_type=source_type,
                                      content_length=content_length)
        else:
            flash('Error Occurred!')
            return redirect(url_for('Qna'))

# Function to generate a paragraph
def generate_paragraph(topic):
    system_prompt = "Generate a paragraph based on the given topic."
    
    completion = groq_client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a paragraph based on this topic: {topic}"}
        ],
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
    )
    
    return completion.choices[0].message.content

def generate_topic():#essay
    """Generate a writing topic using Groq"""
    system_prompt = "Generate an interesting writing topic that would make for a good 250-word essay."
    
    completion = groq_client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Generate an interesting writing topic that would make for a good 250-word essay. The topic should be specific enough to be focused but broad enough to allow for development. Return only the topic without any additional text or explanation."}
        ],
        temperature=0.7,
        max_tokens=100,
        top_p=1,
    )
    
    return completion.choices[0].message.content.strip()

@app.route('/get_topic', methods=['GET'])
def get_topic():
    try:
        topic = generate_topic()
        return jsonify({'topic': topic})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Function to compare and evaluate transcription
def evaluate_accuracyr(generated_paragraph, transcribed_text):
    # Basic comparison of words
    generated_words = generated_paragraph.split()
    transcribed_words = transcribed_text.split()
    
    total_words = len(generated_words)
    correct_words = sum(1 for g, t in zip(generated_words, transcribed_words) if g.lower() == t.lower())
    accuracy = (correct_words / total_words) * 100
    
    # Convert accuracy to 1-5 scale
    score = round((accuracy / 100) * 5)
    
    # Generate feedback using Groq
    system_prompt = "You are evaluating the accuracy of a transcription compared to an original paragraph."
    
    completion = groq_client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
            Evaluate the following transcription accuracy:
            Original Paragraph: {generated_paragraph}
            Transcribed Text: {transcribed_text}
            Score: {score}/5
            
            Provide brief feedback about the accuracy of the transcription and pronunciation.
            """}
        ],
        temperature=0.7,
        max_tokens=512,
        top_p=1,
    )
    
    feedback = completion.choices[0].message.content
    
    return score, feedback

def evaluate_essay(topic, essay):
    """Evaluate the essay and return score and feedback"""
    system_prompt = "You are evaluating essays for clarity, organization, and content."
    
    completion = groq_client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
            Evaluate the following 250-word essay:
            
            Topic: {topic}
            Essay: {essay}
            
            Provide a rating on a scale of 1-5 (where 5 is excellent) based on the following criteria:
            - Relevance to the topic
            - Organization and structure
            - Development of ideas
            - Language usage and clarity
            - Overall quality
            
            First provide the numerical score as an integer between 1 and 5, then provide a brief constructive feedback explaining the rating.
            Format your response exactly as:
            SCORE: [number]
            FEEDBACK: [your feedback]
            """}
        ],
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
    )
    
    response = completion.choices[0].message.content
    
    try:
        lines = response.split('\n')
        score_line = next((line for line in lines if line.startswith('SCORE:')), '')
        feedback_line = next((line for line in lines if line.startswith('FEEDBACK:')), '')
        
        score_text = score_line.replace('SCORE:', '').strip()
        try:
            score = round(float(score_text))
            score = max(1, min(5, score))
        except ValueError:
            print(f"Could not parse score: {score_text}")
            score = 3
        
        feedback = feedback_line.replace('FEEDBACK:', '').strip()
        if not feedback:
            feedback = "No detailed feedback provided."
        
        return score, feedback
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Full response: {response}")
        return 3, "Error processing evaluation. Please try submitting again."


@app.route('/generate', methods=['POST'])
def generate():
    topic = request.form.get('topic')
    if not topic:
        return jsonify({'error': 'No topic provided'}), 400
    
    paragraph = generate_paragraph(topic)
    return jsonify({'paragraph': paragraph})

@app.route('/evaluatereading', methods=['POST'])
def evaluate():
    data = request.json
    generated_paragraph = data.get('paragraph')
    transcribed_text = data.get('transcription')
    
    if not generated_paragraph or not transcribed_text:
        return jsonify({'error': 'Missing required data'}), 400
    
    score, feedback = evaluate_accuracyr(generated_paragraph, transcribed_text)
    
    return jsonify({
        'score': score,
        'feedback': feedback
    })

@app.route('/evaluatewriting', methods=['POST'])
def evaluateR():
    data = request.json
    if not data or 'essay' not in data or 'topic' not in data:
        return jsonify({'error': 'Missing essay or topic'}), 400
    
    essay = data['essay']
    topic = data['topic']
    
    # Update word count validation
    word_count = len(essay.split())
    if word_count < 100:  # Give some flexibility for minimum
        return jsonify({'error': f'Essay is too short. Current word count: {word_count}. Please write at least 200 words.'}), 400
    if word_count > 250:
        return jsonify({'error': f'Essay exceeds 250 words. Current word count: {word_count}.'}), 400
    
    try:
        score, feedback = evaluate_essay(topic, essay)
        return jsonify({
            'score': score,
            'feedback': feedback,
            'wordCount': word_count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/start-assessment', methods=['POST'])
def start_assessment():
    session['current_attempt'] = 0
    session['scores'] = []
    session['feedbacks'] = []
    return next_task()

@app.route('/next-task', methods=['POST'])
def next_task():
    app.logger.debug(f"Current Attempt: {session.get('current_attempt', 0)}")
    app.logger.debug(f"Scores: {session.get('scores', [])}")
    app.logger.debug(f"Feedbacks: {session.get('feedbacks', [])}")

    if session['current_attempt'] >= MAX_ATTEMPTS:
        final_score = sum(session['scores']) / len(session['scores']) if session['scores'] else 0
        return jsonify({
            'complete': True,
            'final_score': final_score,
            'feedbacks': session['feedbacks']
        })
    
    topic = random.choice(TOPICS)
    paragraph = generate_Lparagraph(topic)
    audio_url = generate_audio(paragraph)

    session['current_attempt'] += 1
    session['current_paragraph'] = paragraph

    return jsonify({
        'attempt': session['current_attempt'],
        'paragraph': paragraph,
        'audio_url': audio_url
    })

def generate_Lparagraph(topic):
    system_prompt = "Generate a single line sentence based on this topic."
    
    completion = groq_client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a single line sentence based on this topic: {topic}"}
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=1,
    )
    
    return completion.choices[0].message.content

def evaluate_Laccuracy(generated_paragraph, transcribed_text):
    generated_words = generated_paragraph.split()
    transcribed_words = transcribed_text.split()
    
    total_words = len(generated_words)
    correct_words = sum(1 for g, t in zip(generated_words, transcribed_words) if g.lower() == t.lower())
    accuracy = (correct_words / total_words) * 100
    score = round((accuracy / 100) * 5)
    
    system_prompt = "You are evaluating the accuracy of a transcription compared to an original paragraph."
    
    completion = groq_client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
            Evaluate the following transcription accuracy:
            Original Paragraph: {generated_paragraph}
            Transcribed Text: {transcribed_text}
            Score: {score}/5
            
            Provide brief feedback about the accuracy of the transcription and pronunciation.
            """}
        ],
        temperature=0.7,
        max_tokens=512,
        top_p=1,
    )
    
    feedback = completion.choices[0].message.content
    
    return score, feedback

@app.route('/evaluatelistening', methods=['POST'])
def evaluatelistening():
    task_data = session.get('task_data', [])
    score, feedback = evaluate_Laccuracy(session['current_paragraph'], request.json['transcription'])

    task_data.append({
        'paragraph': session['current_paragraph'],
        'transcription': request.json['transcription'],
        'score': score,
        'feedback': feedback
    })

    session['task_data'] = task_data
    session['scores'].append(score)
    session['feedbacks'].append(feedback)

    if len(session['scores']) == MAX_ATTEMPTS:
        final_score = sum(session['scores']) / len(session['scores'])
        return jsonify({
            'complete': True,
            'final_score': final_score,
            'feedbacks': session['feedbacks'],
            'task_data': task_data
        })

    return jsonify({'score': score, 'feedback': feedback})

def generate_audio(text):
    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)
    tts = gTTS(text=text, lang='en')
    tts.save(filepath)
    return f"/static/audio/{filename}"

def cleanup_audio():
    for file in os.listdir(AUDIO_DIR):
        filepath = os.path.join(AUDIO_DIR, file)
        if os.path.getmtime(filepath) < time.time() - 3600:
            os.remove(filepath)

def load_knowledge_base(pdf_path):
    # Extract text from PDF
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page_num in range(len(pdf_reader.pages)):
        page_text = pdf_reader.pages[page_num].extract_text()
        text += page_text
    
    def process_query_with_groq(query):
        system_prompt = "You are a helpful assistant answering questions based on provided documents."
        
        completion = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Based on the following context, please answer the question: {query}\n\nContext: {text}"}
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
        )
        
        return {"result": completion.choices[0].message.content}
    
    return type('DummyQA', (), {'invoke': process_query_with_groq})()



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf_file' not in request.files:
        return redirect(request.url)
    file = request.files['pdf_file']
    folder = request.form.get('folder', '')
    subfolder = request.form.get('subfolder', '')

    if file and file.filename.endswith('.pdf'):
        filename = file.filename
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        if folder:
            if subfolder:
                folder_path = os.path.join(app.config['FOLDERS'], folder, subfolder)
            else:
                folder_path = os.path.join(app.config['FOLDERS'], folder)
            os.makedirs(folder_path, exist_ok=True)
            shutil.move(upload_path, os.path.join(folder_path, filename))
        else:
            # If no folder is selected, save in a default location
            default_folder = 'Default'
            folder_path = os.path.join(app.config['FOLDERS'], default_folder)
            os.makedirs(folder_path, exist_ok=True)
            shutil.move(upload_path, os.path.join(folder_path, filename))
    return redirect(url_for('pdfmanage'))

@app.route('/move_pdf/<filename>/<source_folder>/<source_subfolder>', methods=['POST'])
def move_pdf(filename, source_folder, source_subfolder=None):
    target_folder = request.form.get('target_folder', source_folder)
    target_subfolder = request.form.get('target_subfolder', '')

    source_path = os.path.join(app.config['FOLDERS'], source_folder, source_subfolder if source_subfolder else '', filename)
    target_path = os.path.join(app.config['FOLDERS'], target_folder, target_subfolder if target_subfolder else '', filename)

    if not os.path.exists(source_path):
        return f"Source file not found: {source_path}", 404

    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.move(source_path, target_path)
    except Exception as e:
        return f"Error moving file: {e}", 500

    return redirect(url_for('pdfmanage'))

@app.route('/view_pdf/<folder>/<subfolder>/<filename>')
@app.route('/view_pdf/<folder>/<filename>')
def view_pdf(folder, subfolder=None, filename=None):
    # Fetch all folders and subfolders
    folders = os.listdir(app.config['FOLDERS'])
    pdf_files = {}
    subfolders = {}

    if folder in folders:
        folder_path = os.path.join(app.config['FOLDERS'], folder)
        if os.path.isdir(folder_path):
            # List all subfolders
            subfolders[folder] = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            pdf_files[folder] = {}
            
            for subfolder_name in subfolders[folder]:
                subfolder_path = os.path.join(folder_path, subfolder_name)
                pdf_files[folder][subfolder_name] = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f)) and f.endswith('.pdf')]
            
            # Include PDFs in the main folder if no subfolder
            pdf_files[folder][''] = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.pdf')]

    return render_template('view_pdf.html', folder=folder, subfolder=subfolder, filename=filename, pdf_files=pdf_files, all_subfolders=subfolders)

@app.route('/serve_pdf/<folder>/<subfolder>/<filename>')
@app.route('/serve_pdf/<folder>/<filename>')
def serve_pdf(folder, subfolder=None, filename=None):
    if subfolder:
        file_path = os.path.join(app.config['FOLDERS'], folder, subfolder, filename)
    else:
        file_path = os.path.join(app.config['FOLDERS'], folder, filename)
    
    print(f"Serving PDF from: {file_path}")  # Debugging statement
    
    if os.path.exists(file_path):
        return send_from_directory(os.path.dirname(file_path), filename)
    else:
        return "PDF file not found.", 404

@app.route('/create_folder', methods=['POST'])
def create_folder():
    new_folder = request.form.get('new_folder')
    if new_folder:
        folder_path = os.path.join(app.config['FOLDERS'], new_folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    return redirect(url_for('pdfmanage'))

@app.route('/create_subfolder', methods=['POST'])
def create_subfolder():
    folder = request.form.get('folder')
    new_subfolder = request.form.get('new_subfolder')
    if folder and new_subfolder:
        subfolder_path = os.path.join(app.config['FOLDERS'], folder, new_subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
    return redirect(url_for('pdfmanage'))


@app.route('/update_pdf_scroll', methods=['POST'])
def update_pdf_scroll():
    filename = request.form.get('filename')
    folder = request.form.get('folder', '')
    subfolder = request.form.get('subfolder', '')
    scroll_percentage = request.form.get('scroll_percentage', '0')

    # New path for storing the scroll percentage
    pdf_info_path = os.path.join('data', f'{folder}_{subfolder}_{filename}.txt')

    with open(pdf_info_path, 'w') as file:
        file.write(scroll_percentage)

    return '', 204

@app.route('/get_query_history/<folder>/<subfolder>/<filename>')
@app.route('/get_query_history/<folder>/<filename>')
def get_query_history(folder, subfolder=None, filename=None):
    pdf_id = f"{folder}_{subfolder}_{filename}" if subfolder else f"{folder}_{filename}"
    Query_db = Query()
    queries = tdb.search(Query_db.pdf_id == pdf_id)
    # Add doc_id to each query
    for query in queries:
        query['doc_id'] = query.doc_id
    return jsonify(queries)

@app.route('/delete_query', methods=['POST'])
def delete_query():
    data = request.json
    query_id = data.get('query_id')
    if query_id:
        tdb.remove(doc_ids=[query_id])
        return jsonify({'success': True})
    return jsonify({'success': False}), 400

@app.route('/query_rag', methods=['POST'])
def query_rag():
    data = request.json
    query = data.get('query')
    filename = data.get('filename')
    folder = data.get('folder')
    subfolder = data.get('subfolder', '')  # Ensure subfolder defaults to an empty string

    # Debugging print statements
    print(f"Query: {query}")
    print(f"Filename: {filename}")
    print(f"Folder: {folder}")
    print(f"Subfolder: '{subfolder}'")

    # Construct the PDF path
    if subfolder:
        pdf_path = os.path.join(app.config['FOLDERS'], folder, subfolder, filename)
    else:
        pdf_path = os.path.join(app.config['FOLDERS'], folder, filename)

    print(f"Constructed PDF path: {pdf_path}")

    if not os.path.exists(pdf_path):
        return jsonify({'response': 'PDF file not found.'}), 404

    # Load the knowledge base for the PDF
    qa_chain = load_knowledge_base(pdf_path)

    # Perform RAG query
    response = qa_chain.invoke(input=query)

    pdf_id = f"{folder}_{subfolder}_{filename}" if subfolder else f"{folder}_{filename}"
    tdb.insert({
        'pdf_id': pdf_id,
        'query': query,
        'response': response['result'],
        'timestamp': datetime.now().isoformat()
    })

    return jsonify({'response': response['result']})


@app.route('/create_note', methods=['POST'])
def create_note():
    title = request.form.get('title')
    content = request.form.get('note_content')
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt=title, ln=1, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=content)
    
    # Create notes folder if it doesn't exist
    notes_folder = os.path.join(app.config['FOLDERS'], 'your_notes')
    os.makedirs(notes_folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{title}_{timestamp}.pdf"
    pdf_path = os.path.join(notes_folder, filename)
    pdf.output(pdf_path)
    
    return redirect(url_for('pdfmanage'))

@app.route('/delete_pdf/<folder>/<filename>')
@app.route('/delete_pdf/<folder>/<subfolder>/<filename>')
def delete_pdf(folder, subfolder=None, filename=None):
    if subfolder:
        file_path = os.path.join(app.config['FOLDERS'], folder, subfolder, filename)
    else:
        file_path = os.path.join(app.config['FOLDERS'], folder, filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        
        # Remove scroll percentage data if exists
        pdf_info_path = os.path.join('data', f'{folder}_{subfolder}_{filename}.txt')
        if os.path.exists(pdf_info_path):
            os.remove(pdf_info_path)
    
    return redirect(url_for('pdfmanage'))

# Sorting algorithms
def bubble_sort(data):
    steps = []
    n = len(data)
    arr = data.copy()
    
    for i in range(n-1):
        for j in range(n-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                steps.append({
                    'array': arr.copy(),
                    'comparing': [j, j+1]
                })
    return steps

def selection_sort(data):
    steps = []
    n = len(data)
    arr = data.copy()
    
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        steps.append({
            'array': arr.copy(),
            'comparing': [i, min_idx]
        })
    return steps

def insertion_sort(data):
    steps = []
    n = len(data)
    arr = data.copy()
    
    for i in range(1, n):
        key = arr[i]
        j = i-1
        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
            steps.append({
                'array': arr.copy(),
                'comparing': [j+1, j+2]
            })
        arr[j+1] = key
    return steps

def merge_sort(data):
    steps = []
    arr = data.copy()
    
    def merge(arr, l, m, r):
        left = arr[l:m+1]
        right = arr[m+1:r+1]
        i = j = 0
        k = l
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
            steps.append({
                'array': arr.copy(),
                'comparing': [k-1, k]
            })
            
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
            steps.append({
                'array': arr.copy(),
                'comparing': [k-1]
            })
            
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1
            steps.append({
                'array': arr.copy(),
                'comparing': [k-1]
            })
    
    def merge_sort_recursive(arr, l, r):
        if l < r:
            m = (l + r) // 2
            merge_sort_recursive(arr, l, m)
            merge_sort_recursive(arr, m + 1, r)
            merge(arr, l, m, r)
    
    merge_sort_recursive(arr, 0, len(arr)-1)
    return steps

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self, max_size=10):
        self.head = None
        self.size = 0
        self.max_size = max_size
    
    def insert_at_beginning(self, data):
        if self.size >= self.max_size:
            return False, "Linked List is full"
        
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
        return True, "Insert successful"
    
    def insert_at_end(self, data):
        if self.size >= self.max_size:
            return False, "Linked List is full"
            
        new_node = Node(data)
        
        if not self.head:
            self.head = new_node
            self.size += 1
            return True, "Insert successful"
            
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
        self.size += 1
        return True, "Insert successful"
    
    def delete_at_beginning(self):
        if not self.head:
            return False, "Linked List is empty"
            
        data = self.head.data
        self.head = self.head.next
        self.size -= 1
        return True, data
    
    def delete_at_end(self):
        if not self.head:
            return False, "Linked List is empty"
            
        if not self.head.next:
            data = self.head.data
            self.head = None
            self.size -= 1
            return True, data
            
        current = self.head
        while current.next.next:
            current = current.next
        data = current.next.data
        current.next = None
        self.size -= 1
        return True, data
    
    def get_all_nodes(self):
        nodes = []
        current = self.head
        while current:
            nodes.append(current.data)
            current = current.next
        return nodes

# Initialize linked list
linked_list = LinkedList()

# Add new routes
@app.route('/linkedlist/insert_beginning', methods=['POST'])
def linkedlist_insert_beginning():
    data = request.get_json()
    value = int(data['value'])
    success, message = linked_list.insert_at_beginning(value)
    return jsonify({
        'success': success,
        'message': message,
        'nodes': linked_list.get_all_nodes()
    })

@app.route('/linkedlist/insert_end', methods=['POST'])
def linkedlist_insert_end():
    data = request.get_json()
    value = int(data['value'])
    success, message = linked_list.insert_at_end(value)
    return jsonify({
        'success': success,
        'message': message,
        'nodes': linked_list.get_all_nodes()
    })

@app.route('/linkedlist/delete_beginning', methods=['POST'])
def linkedlist_delete_beginning():
    success, result = linked_list.delete_at_beginning()
    return jsonify({
        'success': success,
        'message': result if not success else "Delete successful",
        'value': result if success else None,
        'nodes': linked_list.get_all_nodes()
    })

@app.route('/linkedlist/delete_end', methods=['POST'])
def linkedlist_delete_end():
    success, result = linked_list.delete_at_end()
    return jsonify({
        'success': success,
        'message': result if not success else "Delete successful",
        'value': result if success else None,
        'nodes': linked_list.get_all_nodes()
    })

# Stack and Queue data structures
class Stack:
    def __init__(self, max_size=10):
        self.items = []
        self.max_size = max_size
    
    def push(self, item):
        if len(self.items) >= self.max_size:
            return False, "Stack is full"
        self.items.append(item)
        return True, "Push successful"
    
    def pop(self):
        if not self.items:
            return False, "Stack is empty"
        return True, self.items.pop()
    
    def peek(self):
        if not self.items:
            return None
        return self.items[-1]
    
    def get_items(self):
        return self.items

class Queue:
    def __init__(self, max_size=10):
        self.items = []
        self.max_size = max_size
    
    def enqueue(self, item):
        if len(self.items) >= self.max_size:
            return False, "Queue is full"
        self.items.append(item)
        return True, "Enqueue successful"
    
    def dequeue(self):
        if not self.items:
            return False, "Queue is empty"
        return True, self.items.pop(0)
    
    def peek(self):
        if not self.items:
            return None
        return self.items[0]
    
    def get_items(self):
        return self.items

# Initialize data structures
stack = Stack()
queue = Queue()

@app.route('/generateDS', methods=['POST'])
def generateDS():
    data = request.get_json()
    size = int(data['size'])
    min_val = int(data['min'])
    max_val = int(data['max'])
    
    array = [random.randint(min_val, max_val) for _ in range(size)]
    return jsonify({'array': array})

@app.route('/sort', methods=['POST'])
def sort():
    data = request.get_json()
    array = data['array']
    algorithm = data['algorithm']
    
    algorithms = {
        'bubble': bubble_sort,
        'selection': selection_sort,
        'insertion': insertion_sort,
        'merge': merge_sort
    }
    
    steps = algorithms[algorithm](array)
    return jsonify({'steps': steps})

@app.route('/stack/push', methods=['POST'])
def stack_push():
    data = request.get_json()
    value = int(data['value'])
    success, message = stack.push(value)
    return jsonify({
        'success': success,
        'message': message,
        'items': stack.get_items()
    })

@app.route('/stack/pop', methods=['POST'])
def stack_pop():
    success, result = stack.pop()
    return jsonify({
        'success': success,
        'message': result if not success else "Pop successful",
        'value': result if success else None,
        'items': stack.get_items()
    })

@app.route('/queue/enqueue', methods=['POST'])
def queue_enqueue():
    data = request.get_json()
    value = int(data['value'])
    success, message = queue.enqueue(value)
    return jsonify({
        'success': success,
        'message': message,
        'items': queue.get_items()
    })

@app.route('/queue/dequeue', methods=['POST'])
def queue_dequeue():
    success, result = queue.dequeue()
    return jsonify({
        'success': success,
        'message': result if not success else "Dequeue successful",
        'value': result if success else None,
        'items': queue.get_items()
    })

class QuizManager:
    _instance = None
    _questions = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = QuizManager()
        return cls._instance

    def store_questions(self, quiz_id, questions):
        self._questions[quiz_id] = questions

    def get_questions(self, quiz_id):
        return self._questions.get(quiz_id)

    def clear_questions(self, quiz_id):
        if quiz_id in self._questions:
            del self._questions[quiz_id]

quiz_manager = QuizManager.get_instance()

def load_questions(field, difficulty):
    file_path = QUIZ_FILES.get(field)
    if file_path and os.path.exists(file_path):
        df = pd.read_excel(file_path)
        num_questions = QUIZ_CONFIG[difficulty]['questions']
        return df.sample(n=min(num_questions, len(df))).reset_index(drop=True)
    return None

@app.route('/select_questions', methods=['GET', 'POST'])
def select_quiz():
    if request.method == 'GET':
        session.clear()
        # Clear any old quiz data
        if 'quiz_id' in session:
            quiz_manager.clear_questions(session['quiz_id'])
        return render_template('select_questions.html',active_page='select_questions')
    
    field = request.form.get('field')
    difficulty = request.form.get('difficulty')
    
    if not field or not difficulty or field not in QUIZ_FILES or difficulty not in QUIZ_CONFIG:
        return redirect(url_for('select_quiz'))
    
    questions_df = load_questions(field, difficulty)
    if questions_df is None:
        return redirect(url_for('select_quiz'))
    
    # Generate a unique quiz ID
    quiz_id = str(int(time.time()))
    
    # Store questions in QuizManager instead of session
    quiz_manager.store_questions(quiz_id, questions_df.to_dict(orient='records'))
    
    # Store only essential data in session
    session['quiz_id'] = quiz_id
    session['current_page'] = 0
    session['field'] = field
    session['difficulty'] = difficulty
    session['start_time'] = int(time.time())
    session['total_questions'] = len(questions_df)
    session['question_status'] = ['unattempted'] * len(questions_df)
    session['marked_for_review'] = [False] * len(questions_df)
    session['answers'] = {}
    
    return redirect(url_for('quiz'))

@app.route('/start-quiz', methods=['POST'])
def start_quiz():
    field = request.form.get('field')
    difficulty = request.form.get('difficulty')
    
    if field not in QUIZ_FILES or difficulty not in QUIZ_CONFIG:
        return redirect(url_for('select_quiz'))
    
    questions_df = load_questions(field, difficulty)
    if questions_df is None:
        return redirect(url_for('select_quiz'))
    
    # Store quiz configuration in session
    session['questions'] = questions_df.to_dict(orient='records')
    session['current_page'] = 0
    session['field'] = field
    session['difficulty'] = difficulty
    session['start_time'] = int(time.time())
    session['total_questions'] = len(questions_df)
    session['question_status'] = ['unattempted'] * len(questions_df)
    session['marked_for_review'] = [False] * len(questions_df)
    session['answers'] = {}
    
    return redirect(url_for('quiz'))

@app.route('/update-question-status', methods=['POST'])
def update_question_status():
    data = request.json
    question_index = data.get('questionIndex')
    status = data.get('status')
    is_review = data.get('isReview', False)
    
    if 'question_status' in session and question_index is not None:
        question_status = session['question_status']
        marked_for_review = session['marked_for_review']
        
        question_status[question_index] = status
        marked_for_review[question_index] = is_review
        
        session['question_status'] = question_status
        session['marked_for_review'] = marked_for_review
        
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    if 'quiz_id' not in session:
        return redirect(url_for('select_quiz'))
    
    questions = quiz_manager.get_questions(session['quiz_id'])
    if not questions:
        return redirect(url_for('select_quiz'))
    
    total_questions = len(questions)
    current_page = session.get('current_page', 0)
    start_time = session.get('start_time')
    difficulty = session.get('difficulty')
    
    total_time = QUIZ_CONFIG[difficulty]['time'] * 60
    elapsed_time = int(time.time()) - start_time if start_time else 0
    remaining_time = max(0, total_time - elapsed_time)
    
    if remaining_time <= 0:
        return calculate_score()
        
    if request.method == 'POST':
        answer = request.form.get(f"question{current_page}")
        if answer:
            session['answers'] = session.get('answers', {})
            session['answers'][str(current_page)] = answer

        action = request.form.get('action')
        
        if action == 'submit':
            return calculate_score()
        elif action == 'navigate':
            new_page = int(request.form.get('current_question', 0))
            if 0 <= new_page < total_questions:
                current_page = new_page
        else:
            if action == 'next' and current_page < total_questions - 1:
                current_page += 1
            elif action == 'prev' and current_page > 0:
                current_page -= 1
        
        session['current_page'] = current_page
    
    saved_answers = session.get('answers', {})
    current_answer = saved_answers.get(str(current_page))
    
    return render_template('aptitudeL.html',
                         question=questions[current_page],
                         current_page=current_page,
                         total_pages=total_questions,
                         total_questions=total_questions,
                         remaining_time=remaining_time,
                         question_status=session.get('question_status', []),
                         marked_for_review=session.get('marked_for_review', []),
                         current_answer=current_answer,
                         show_results=False)

def calculate_score():
    if 'quiz_id' not in session:
        return redirect(url_for('select_quiz'))
        
    questions = quiz_manager.get_questions(session['quiz_id'])
    if not questions:
        return redirect(url_for('select_quiz'))
        
    answers = session.get('answers', {})
    score = 0
    detailed_results = []
    
    # Calculate score and prepare detailed results
    for i, question in enumerate(questions):
        selected_option = answers.get(str(i))
        is_correct = selected_option == question['correct_option']
        if is_correct:
            score += 1
            
        # Get the text of selected and correct options
        selected_text = None
        if selected_option:
            selected_text = question[selected_option]  # e.g., question['option1']
            
        correct_text = question[question['correct_option']]  # Get text of correct option
        
        # Prepare detailed result for this question
        result = {
            'question': question['question'],
            'selected_option': selected_option,
            'selected_text': selected_text,
            'correct_option': question['correct_option'],
            'correct_text': correct_text,
            'is_correct': is_correct,
            'explanation': question['Explanation']
        }
        detailed_results.append(result)
    
    return render_template('quiz_results.html',
                         score=score,
                         total_questions=len(questions),
                         detailed_results=detailed_results,
                         show_results=True)


############ Testing WALA ################
def safe_wikipedia_search(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=5)
    except Exception as e:
        return f"Error fetching information from Wikipedia: {str(e)}"

def safe_duckduckgo_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        return "\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Error searching with DuckDuckGo: {str(e)}"

def safe_math_eval(expression: str) -> str:
    try:
        # Validate the expression for safety
        allowed_chars = set("0123456789+-*/() .")
        if not all(c in allowed_chars for c in expression):
            return "Invalid characters in expression."
        
        # Use eval to calculate the result (risky but controlled)
        return str(eval(expression))
    except Exception as e:
        return f"Error in calculation: {str(e)}"

def search_arxiv(query: str) -> str:
    try:
        # Basic URL for arXiv API
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "max_results": 5
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return f"Error: Received status code {response.status_code} from arXiv"
        
        # Parse XML response
        root = ET.fromstring(response.text)
        
        results = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            url = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
            
            # Truncate summary if too long
            if len(summary) > 300:
                summary = summary[:300] + "..."
            
            results.append(f"Title: {title}\nSummary: {summary}\nURL: {url}\n")
        
        return "\n".join(results) if results else "No results found on arXiv for your query."
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"

def search_pubmed(query: str) -> str:
    try:
        # Use PubMed API to search for papers
        # First get IDs of matching papers
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        esearch_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": 5
        }
        
        esearch_response = requests.get(esearch_url, params=esearch_params)
        esearch_data = json.loads(esearch_response.text)
        
        if "esearchresult" not in esearch_data or "idlist" not in esearch_data["esearchresult"]:
            return "No results found in PubMed for your query."
        
        id_list = esearch_data["esearchresult"]["idlist"]
        
        if not id_list:
            return "No results found in PubMed for your query."
        
        # Get summaries for the IDs
        esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        esummary_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json"
        }
        
        esummary_response = requests.get(esummary_url, params=esummary_params)
        esummary_data = json.loads(esummary_response.text)
        
        results = []
        
        for pmid in id_list:
            if pmid in esummary_data["result"]:
                paper = esummary_data["result"][pmid]
                title = paper.get("title", "No Title Available")
                abstract = "Abstract not available"
                
                # Try to get the abstract for each paper
                efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                efetch_params = {
                    "db": "pubmed",
                    "id": pmid,
                    "retmode": "xml"
                }
                
                efetch_response = requests.get(efetch_url, params=efetch_params)
                
                # Parse XML to extract abstract
                root = ET.fromstring(efetch_response.text)
                abstract_elements = root.findall(".//AbstractText")
                if abstract_elements:
                    abstract = " ".join([elem.text for elem in abstract_elements if elem.text])
                
                # Truncate abstract if too long
                if len(abstract) > 300:
                    abstract = abstract[:300] + "..."
                
                results.append(f"Title: {title}\nAbstract: {abstract}\nPMID: {pmid}\n")
        
        return "\n".join(results) if results else "No detailed results found in PubMed for your query."
    except Exception as e:
        return f"Error searching PubMed: {str(e)}"

# List of available research tools
research_tools = [
    {"name": "Wikipedia", "description": "Get detailed explanations and summaries from Wikipedia"},
    {"name": "DuckDuckGo", "description": "Search the web for current information"},
    {"name": "BasicMath", "description": "Performs basic mathematical calculations"},
    {"name": "ArXiv", "description": "Searches arXiv for scientific papers"},
    {"name": "PubMed", "description": "Searches PubMed for medical research"}
]

def process_research_with_groq(query):
    tool_descriptions = "\n".join([f"{tool['name']}: {tool['description']}" for tool in research_tools])
    system_prompt = f"""You are a research assistant with access to the following tools:
{tool_descriptions}

Based on the user's query, determine which tool would be most appropriate to use. 
Then, provide a comprehensive answer using the information retrieved from that tool.
If you need information from multiple tools, use them in sequence and combine the results.

Format your response in clear, well-organized sections with appropriate headings if necessary.
"""

    completion = groq_client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
    )
    
    return completion.choices[0].message.content

@app.route('/research')
def research():
    return render_template('research.html', tools=research_tools)

@app.route('/search', methods=['POST'])
def search():
    try:
        user_input = request.json.get('query')
        if not user_input:
            return jsonify({'error': 'No query provided'}), 400
        
        result = process_research_with_groq(user_input)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

############### END ######################


GOOGLE_CLIENT_ID = "278065291652-p0ihdmqajacn7lgvodbei0lj5nqrfrpv.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "YOUR_GOOGLE_CLIENT_SECRET"  #
from google.auth.transport.requests import Request

@app.route('/auth/google', methods=['POST'])
def google_auth():
    token = request.json.get('id_token')
    try:
        request_instance = Request()

        idinfo = id_token.verify_oauth2_token(token, request_instance, GOOGLE_CLIENT_ID)

        user_id = idinfo['sub']
        email = idinfo['email']
        name = idinfo['name']

        return jsonify(success=True, message="Google Sign-In successful")
    except ValueError:
        return jsonify(success=False, message="Invalid token")

@app.route('/auth/google/callback', methods=['GET'])
def google_callback():
    """
    Handle the redirect from Google Sign-In and redirect to the landing page.
    """
    
    return redirect(url_for('landingpage'))

@app.route('/auth/firebase', methods=['POST'])
def firebase_auth():
    data = request.json
    try:
        decoded_token = auth.verify_id_token(data.get('uid'))
        user_id = decoded_token['uid']
        email = data.get('email')
        name = data.get('name')

        session['user_id'] = user_id
        session['email'] = email
        session['name'] = name

        return jsonify(success=True, message="Firebase Sign-In successful")
    except Exception as e:
        return jsonify(success=False, message=str(e))

@app.route('/quizpage')
def quizpage():
    return render_template('quizpage.html')

# Financial Literacy Course Routes
@app.route('/courses')
def courses():
    # Hardcoded module data instead of database queries
    modules = [
        {
            'id': 1,
            'title': 'Investment Basics',
            'description': 'Learn the fundamentals of investing and building wealth',
            'image': 'investment_basics.jpg'
        },
        {
            'id': 2,
            'title': 'Stock Market 101',
            'description': 'Understanding how the stock market works',
            'image': 'stock_market.jpg'
        },
        {
            'id': 3,
            'title': 'Mutual Funds Explained',
            'description': 'Learn how mutual funds work and their benefits',
            'image': 'mutual_funds.jpg'
        },
        {
            'id': 4,
            'title': 'Retirement Planning',
            'description': 'Building a secure financial future',
            'image': 'retirement.jpg'
        },
        {
            'id': 5,
            'title': 'Tax-Efficient Investing',
            'description': 'Strategies to minimize your tax burden',
            'image': 'tax_efficient.jpg'
        }
    ]
    
    # Hardcoded progress data
    module_progress = {
        1: 0,  # 0% complete
        2: 0,  # 0% complete
        3: 0,  # 0% complete
        4: 0,  # 0% complete
        5: 0   # 0% complete
    }
    
    return render_template('courses.html', modules=modules, module_progress=module_progress, active_page='courses')

@app.route('/course/<int:module_id>')
def course_module(module_id):
    # Hardcoded module data
    modules = {
        1: {
            'id': 1,
            'title': 'Investment Basics',
            'description': 'Learn the fundamentals of investing and building wealth'
        },
        2: {
            'id': 2,
            'title': 'Stock Market 101',
            'description': 'Understanding how the stock market works'
        },
        3: {
            'id': 3,
            'title': 'Mutual Funds Explained',
            'description': 'Learn how mutual funds work and their benefits'
        },
        4: {
            'id': 4,
            'title': 'Retirement Planning',
            'description': 'Building a secure financial future'
        },
        5: {
            'id': 5,
            'title': 'Tax-Efficient Investing',
            'description': 'Strategies to minimize your tax burden'
        }
    }
    
    # Hardcoded sections data
    sections_by_module = {
        1: [
            {
                'id': 1,
                'title': 'What is Investing?',
                'order': 1,
                'module_id': 1
            },
            {
                'id': 2,
                'title': 'Types of Investments',
                'order': 2,
                'module_id': 1
            },
            {
                'id': 3,
                'title': 'Risk vs. Return',
                'order': 3,
                'module_id': 1
            }
        ],
        2: [
            {
                'id': 4,
                'title': 'What is the Stock Market?',
                'order': 1,
                'module_id': 2
            },
            {
                'id': 5,
                'title': 'Stock Exchanges',
                'order': 2,
                'module_id': 2
            },
            {
                'id': 6,
                'title': 'Bull vs. Bear Markets',
                'order': 3,
                'module_id': 2
            }
        ],
        3: [
            {
                'id': 7,
                'title': 'What are Mutual Funds?',
                'order': 1,
                'module_id': 3
            },
            {
                'id': 8,
                'title': 'Types of Mutual Funds',
                'order': 2,
                'module_id': 3
            },
            {
                'id': 9,
                'title': 'Advantages and Disadvantages',
                'order': 3,
                'module_id': 3
            }
        ],
        4: [
            {
                'id': 10,
                'title': 'Why Plan for Retirement?',
                'order': 1,
                'module_id': 4
            },
            {
                'id': 11,
                'title': 'Retirement Accounts',
                'order': 2,
                'module_id': 4
            },
            {
                'id': 12,
                'title': 'The 4% Rule',
                'order': 3,
                'module_id': 4
            }
        ],
        5: [
            {
                'id': 13,
                'title': 'The Impact of Taxes on Investments',
                'order': 1,
                'module_id': 5
            },
            {
                'id': 14,
                'title': 'Tax-Advantaged Accounts',
                'order': 2,
                'module_id': 5
            },
            {
                'id': 15,
                'title': 'Tax-Loss Harvesting',
                'order': 3,
                'module_id': 5
            }
        ]
    }
    
    # Hardcoded progress data (simulated completed sections)
    progress_data = {}
    
    # Get the module
    module = modules.get(module_id)
    if not module:
        return "Module not found", 404
    
    # Get sections for this module
    sections = sections_by_module.get(module_id, [])
    
    return render_template('course_module.html', module=module, sections=sections, progress_data=progress_data, active_page='courses')

@app.route('/course/section/<int:section_id>')
def course_section(section_id):
    # Hardcoded section content
    sections = {
        1: {
            'id': 1,
            'title': 'What is Investing?',
            'content': '<h2>What is Investing?</h2><p>Investing is the act of allocating resources, usually money, with the expectation of generating income or profit over time. Unlike saving, which is setting aside money for future use with minimal risk, investing involves taking on risk in pursuit of greater returns.</p>',
            'module_id': 1,
            'order': 1
        },
        2: {
            'id': 2,
            'title': 'Types of Investments',
            'content': '<h2>Types of Investments</h2><p>Common investment types include stocks (ownership in a company), bonds (loans to companies or governments), mutual funds (professionally managed portfolios), real estate, and more specialized options like ETFs, REITs, and cryptocurrency.</p>',
            'module_id': 1,
            'order': 2
        },
        3: {
            'id': 3,
            'title': 'Risk vs. Return',
            'content': '<h2>Risk vs. Return</h2><p>A fundamental principle of investing is the relationship between risk and return. Generally, investments with higher potential returns come with higher risks. Understanding your risk tolerance is crucial for building an appropriate investment portfolio.</p>',
            'module_id': 1,
            'order': 3
        },
        4: {
            'id': 4,
            'title': 'What is the Stock Market?',
            'content': '<h2>What is the Stock Market?</h2><p>The stock market is a collection of exchanges where stocks (pieces of ownership in businesses) are traded. It provides companies with access to capital and investors with potential investment returns through capital gains and dividends.</p>',
            'module_id': 2,
            'order': 1
        },
        5: {
            'id': 5,
            'title': 'Stock Exchanges',
            'content': '<h2>Stock Exchanges</h2><p>Major stock exchanges include the New York Stock Exchange (NYSE) and NASDAQ in the US, as well as international exchanges like the London Stock Exchange, Tokyo Stock Exchange, and Shanghai Stock Exchange.</p>',
            'module_id': 2,
            'order': 2
        },
        6: {
            'id': 6,
            'title': 'Bull vs. Bear Markets',
            'content': '<h2>Bull vs. Bear Markets</h2><p>A bull market is characterized by rising stock prices and optimism, while a bear market features falling prices and pessimism. Understanding market cycles can help investors make more informed decisions.</p>',
            'module_id': 2,
            'order': 3
        },
        7: {
            'id': 7,
            'title': 'What are Mutual Funds?',
            'content': '<h2>What are Mutual Funds?</h2><p>Mutual funds are investment vehicles that pool money from many investors to purchase a diversified portfolio of stocks, bonds, or other securities. They are managed by professional fund managers who make investment decisions on behalf of the fund\'s investors.</p>',
            'module_id': 3,
            'order': 1
        },
        8: {
            'id': 8,
            'title': 'Types of Mutual Funds',
            'content': '<h2>Types of Mutual Funds</h2><p>Common types include equity funds (stocks), fixed-income funds (bonds), money market funds (short-term debt), balanced funds (mix of stocks and bonds), and index funds (track specific market indices).</p>',
            'module_id': 3,
            'order': 2
        },
        9: {
            'id': 9,
            'title': 'Advantages and Disadvantages',
            'content': '<h2>Advantages and Disadvantages</h2><p>Advantages include diversification, professional management, and accessibility. Disadvantages include fees and expenses, potential tax inefficiency, and lack of control over specific investments.</p>',
            'module_id': 3,
            'order': 3
        },
        10: {
            'id': 10,
            'title': 'Why Plan for Retirement?',
            'content': '<h2>Why Plan for Retirement?</h2><p>Retirement planning involves determining income goals for retirement and the actions and decisions necessary to achieve those goals. Planning early allows you to take advantage of compound interest and adjust your strategy as needed.</p>',
            'module_id': 4,
            'order': 1
        },
        11: {
            'id': 11,
            'title': 'Retirement Accounts',
            'content': '<h2>Retirement Accounts</h2><p>Common retirement accounts include 401(k)s, IRAs (Traditional and Roth), pension plans, and annuities. Each has different tax advantages, contribution limits, and withdrawal rules.</p>',
            'module_id': 4,
            'order': 2
        },
        12: {
            'id': 12,
            'title': 'The 4% Rule',
            'content': '<h2>The 4% Rule</h2><p>The 4% rule is a guideline suggesting that retirees can withdraw 4% of their retirement portfolio in the first year, then adjust that amount for inflation each subsequent year, with a high probability of not running out of money for at least 30 years.</p>',
            'module_id': 4,
            'order': 3
        },
        13: {
            'id': 13,
            'title': 'The Impact of Taxes on Investments',
            'content': '<h2>The Impact of Taxes on Investments</h2><p>Taxes can significantly reduce investment returns over time. Understanding how different investments and accounts are taxed can help you maximize after-tax returns through strategic placement of assets.</p>',
            'module_id': 5,
            'order': 1
        },
        14: {
            'id': 14,
            'title': 'Tax-Advantaged Accounts',
            'content': '<h2>Tax-Advantaged Accounts</h2><p>These include retirement accounts like 401(k)s and IRAs, Health Savings Accounts (HSAs), and 529 college savings plans. Each offers specific tax advantages for different financial goals.</p>',
            'module_id': 5,
            'order': 2
        },
        15: {
            'id': 15,
            'title': 'Tax-Loss Harvesting',
            'content': '<h2>Tax-Loss Harvesting</h2><p>This strategy involves selling investments that have experienced losses to offset capital gains tax liability. It can help reduce taxable income while maintaining your overall investment allocation.</p>',
            'module_id': 5,
            'order': 3
        }
    }
    
    # Hardcoded module data
    modules = {
        1: {
            'id': 1,
            'title': 'Investment Basics',
            'description': 'Learn the fundamentals of investing and building wealth'
        },
        2: {
            'id': 2,
            'title': 'Stock Market 101',
            'description': 'Understanding how the stock market works'
        },
        3: {
            'id': 3,
            'title': 'Mutual Funds Explained',
            'description': 'Learn how mutual funds work and their benefits'
        },
        4: {
            'id': 4,
            'title': 'Retirement Planning',
            'description': 'Building a secure financial future'
        },
        5: {
            'id': 5,
            'title': 'Tax-Efficient Investing',
            'description': 'Strategies to minimize your tax burden'
        }
    }
    
    # Hardcoded progress data
    completed = False
    
    # Get the section
    section = sections.get(section_id)
    if not section:
        return "Section not found", 404
    
    # Get the module
    module = modules.get(section['module_id'])
    if not module:
        return "Module not found", 404
    
    # Get prev and next sections for navigation
    section_ids = [s['id'] for s in sorted([s for s in sections.values() if s['module_id'] == section['module_id']], key=lambda x: x['order'])]
    current_index = section_ids.index(section_id)
    
    prev_section = sections.get(section_ids[current_index - 1]) if current_index > 0 else None
    next_section = sections.get(section_ids[current_index + 1]) if current_index < len(section_ids) - 1 else None
    
    return render_template('course_section.html', 
                          section=section, 
                          module=module, 
                          completed=completed, 
                          prev_section=prev_section, 
                          next_section=next_section, 
                          active_page='courses')

@app.route('/course/complete-section/<int:section_id>', methods=['POST'])
def complete_section(section_id):
    # Simply return the redirect URL without updating any database
    # The completion status is now tracked in localStorage on the client side
    
    # Find next section for redirection
    sections = {
        1: {'module_id': 1, 'order': 1},
        2: {'module_id': 1, 'order': 2},
        3: {'module_id': 1, 'order': 3},
        4: {'module_id': 2, 'order': 1},
        5: {'module_id': 2, 'order': 2},
        6: {'module_id': 2, 'order': 3},
        7: {'module_id': 3, 'order': 1},
        8: {'module_id': 3, 'order': 2},
        9: {'module_id': 3, 'order': 3},
        10: {'module_id': 4, 'order': 1},
        11: {'module_id': 4, 'order': 2},
        12: {'module_id': 4, 'order': 3},
        13: {'module_id': 5, 'order': 1},
        14: {'module_id': 5, 'order': 2},
        15: {'module_id': 5, 'order': 3}
    }
    
    section = sections.get(section_id)
    if not section:
        return jsonify({'success': False, 'message': 'Section not found'})
    
    # Get all sections for this module
    module_sections = sorted([s for s_id, s in sections.items() if s['module_id'] == section['module_id']], key=lambda x: x['order'])
    
    # Find current section index
    current_index = next((i for i, s in enumerate(module_sections) if s.get('order') == section['order']), -1)
    
    if current_index < len(module_sections) - 1:
        # There is a next section
        next_section_id = next((s_id for s_id, s in sections.items() 
                              if s['module_id'] == section['module_id'] and s['order'] == module_sections[current_index + 1]['order']), None)
        if next_section_id:
            return jsonify({'success': True, 'redirect': url_for('course_section', section_id=next_section_id)})
    
    # No next section, redirect to module page
    return jsonify({'success': True, 'redirect': url_for('course_module', module_id=section['module_id'])})

@app.route('/profile')
def profile():
    # Check if user is logged in
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('signin'))
        
    user_id = session.get('user_id')
    user_email = session.get('email')
    user_name = session.get('name')
    
    # User must be logged in due to check above
    user_data = {
        'id': user_id,
        'email': user_email,
        'name': user_name
    }
    
    # We don't need to pass module progress data as it's handled by JavaScript with localStorage
    return render_template('profile.html', user=user_data, active_page='profile')

# Admin route to seed course data
@app.route('/admin/seed-course-data')
def seed_course_data():
    # Since we're using static data, this endpoint is no longer needed
    # But keep it for compatibility
    return "Using static hardcoded data instead of database. No need to seed data."

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('signin'))
        return f(*args, **kwargs)
    return decorated_function

if __name__ == '__main__':
    app.run(debug=True)