from flask import Flask, request, render_template, flash, redirect, url_for, Response, send_from_directory, jsonify, session
from google.oauth2 import id_token  # Import id_token for Google OAuth2
import os, uuid, time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import re
import pdfplumber
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from fpdf import FPDF
from objective import ObjectiveTest
from subjective import SubjectiveTest
from text_filter import clean_text, preserve_content_length, filter_unwanted_text
import cv2
import mediapipe as mp
import numpy as np
from collections import Counter
import threading
import spacy
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
#testing
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.llms import Ollama
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
import wikipedia
from duckduckgo_search import DDGS
import requests
from typing import Optional, Dict, Any, List
import xml.etree.ElementTree as ET
import json
from typing import Union
import re
from dotenv import load_dotenv

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
nlp = spacy.load("en_core_web_sm")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

################ Testing waala #########
llama = Ollama(model="llama3.1:8b")
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
SYSTEM_PROMPT = """You are Flaix, a helpful and knowledgeable financial assistant designed specifically for Indian users. 
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
                    types.Part.from_text(text="I understand my role as Flaix, a financial assistant for Indian users. I'll provide helpful information about investing and financial planning in simple language.")
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
        # Fallback to Llama if available
        try:
            return get_llama_response(user_query)
        except:
            return "I apologize, but I encountered an error while processing your request. Please try again later."

# Fallback to Llama if Gemini is not available
def get_llama_response(query):
    """Fallback to use Llama model when Gemini is not available"""
    try:
        template = """
        You are Flaix, a helpful and knowledgeable financial assistant designed specifically for Indian users. 
        Your purpose is to improve financial literacy and provide guidance on investments in the Indian market.

        Please respond to the following query:
        {query}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llama
        response = chain.invoke({"query": query})
        return response
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html', active_page='chatbot')

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


# Create prompt templates
generation_prompt = ChatPromptTemplate.from_template("Generate a paragraph based on this topic: {topic}")
feedback_prompt = ChatPromptTemplate.from_template("""
Evaluate the following transcription accuracy:
Original Paragraph: {original_paragraph}
Transcribed Text: {transcribed_text}
Score: {score}/5

Provide brief feedback about the accuracy of the transcription and pronunciation.
""")
#essay
topic_prompt = ChatPromptTemplate.from_template(
    "Generate an interesting writing topic that would make for a good 250-word essay. "
    "The topic should be specific enough to be focused but broad enough to allow for development. "
    "Return only the topic without any additional text or explanation."
)

evaluation_prompt = ChatPromptTemplate.from_template("""
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
""")

generation_Lprompt = ChatPromptTemplate.from_template("Generate a single line sentence based on this topic: {topic}")
feedback_Lprompt = ChatPromptTemplate.from_template("""
Evaluate the following transcription accuracy:
Original Paragraph: {original_paragraph}
Transcribed Text: {transcribed_text}
Score: {score}/5
Provide brief feedback about the accuracy of the transcription and pronunciation.
""")

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

question_prompt = ChatPromptTemplate.from_template(question_template)
model = OllamaLLM(model="llama3.1:8b")
question_chain = question_prompt | model
text_splitter = CharacterTextSplitter(separator="/n", chunk_size=1000, chunk_overlap=200)
embeddings = HuggingFaceEmbeddings()

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
    return render_template('signin.html', active_page='home')

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
        inputText = request.form.get("itext", "")
        testType = request.form.get("test_type")
        noOfQues = request.form.get("noq")
        processing_method = request.form.get("processing_method")  # Get processing method

        # Handle file upload
        pdf_file = request.files.get("pdf_file")
        if pdf_file:
            filename = secure_filename(pdf_file.filename)
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pdf_file.save(pdf_path)
            inputText = extract_text_from_pdf(pdf_path)

        # Filter the input text
        cleaned_text = clean_text(inputText)
        filtered_text = filter_unwanted_text(cleaned_text)
        final_filtered_text = preserve_content_length(inputText, filtered_text, retention_factor=0.9)

        if testType == "objective":
            if processing_method == "nlp":
                objective_generator = ObjectiveTest(final_filtered_text, noOfQues)
                mcqs = objective_generator.generate_mcqs()
                return render_template('objective.html', mcqs=mcqs)

            elif processing_method == "llm":
                # Here, include num_questions in the prompt input
                prompt_input = question_template.format(num_questions=noOfQues, input_text=final_filtered_text)
                response = question_chain.invoke({"input_text": final_filtered_text, "num_questions": noOfQues})
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
                # Generate Q&A using Llama 3.1
                template = """
                You are an AI assistant designed to generate educational questions and answers based on the provided content. Your task is to create relevant and insightful questions that test the understanding of the text, followed by accurate and concise answers.

                Content: 
                {question}

                Now, generate the following:
                1. A set of well-structured questions that directly relate to the key points in the content.
                2. For each question, provide a clear and accurate answer.

                Format:
                Question 1: [Your question here]
                Answer 1: 
                [Your answer here]

                Question 2: [Your question here]
                Answer 2: 
                [Your answer here]

                Please ensure the questions cover different aspects of the content, and the answers are informative and to the point.
                """

                prompt = ChatPromptTemplate.from_template(template)
                model = OllamaLLM(model="llama3.1:8b")
                chain = prompt | model

                response = chain.invoke({"question": final_filtered_text})
                qna_text = response 
                qna_text_cleaned = remove_special_characters(qna_text)

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

                # Render HTML template with Q&A data
                return render_template('qna_template.html', qna_items=qna_items)
        else:
            flash('Error Occurred!')
            return redirect(url_for('Qna'))

# Function to generate a paragraph
def generate_paragraph(topic):
    paragraph_chain = generation_prompt | model
    return paragraph_chain.invoke({"topic": topic})

def generate_topic():#essay
    """Generate a writing topic using Ollama"""
    topic_chain = topic_prompt | model
    return topic_chain.invoke({}).strip()

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
    
    # Generate feedback using the template
    feedback_chain = feedback_prompt | model
    feedback = feedback_chain.invoke({
        "original_paragraph": generated_paragraph,
        "transcribed_text": transcribed_text,
        "score": score
    })
    
    return score, feedback

def evaluate_essay(topic, essay):
    """Evaluate the essay and return score and feedback"""
    evaluation_chain = evaluation_prompt | model
    response = evaluation_chain.invoke({
        "topic": topic,
        "essay": essay
    })
    
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
    paragraph_chain = generation_Lprompt | model
    return paragraph_chain.invoke({"topic": topic})

def evaluate_Laccuracy(generated_paragraph, transcribed_text):
    generated_words = generated_paragraph.split()
    transcribed_words = transcribed_text.split()
    
    total_words = len(generated_words)
    correct_words = sum(1 for g, t in zip(generated_words, transcribed_words) if g.lower() == t.lower())
    accuracy = (correct_words / total_words) * 100
    score = round((accuracy / 100) * 5)
    
    feedback_chain = feedback_Lprompt | model
    feedback = feedback_chain.invoke({
        "original_paragraph": generated_paragraph,
        "transcribed_text": transcribed_text,
        "score": score
    })
    
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
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_chunks = text_splitter.split_documents(documents)
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    return RetrievalQA.from_chain_type(model, retriever=knowledge_base.as_retriever())



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
    """Safely search Wikipedia with error handling"""
    try:
        query = query.strip('"\'').replace('+', ' ')
        search_results = wikipedia.search(query, results=1)
        if not search_results:
            return f"No Wikipedia articles found for '{query}'"
        
        page_title = search_results[0]
        page = wikipedia.page(page_title, auto_suggest=False)
        return page.summary[0:500]
        
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            return page.summary[0:500]
        except:
            return f"Multiple Wikipedia articles found for '{query}'. Please be more specific."
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia article found for '{query}'"
    except Exception as e:
        return f"An error occurred while searching Wikipedia: {str(e)}"

def safe_duckduckgo_search(query: str) -> str:
    """Safely search DuckDuckGo with error handling"""
    try:
        query = query.strip('"\'').replace('+', ' ')
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        
        if not results:
            return "No results found on DuckDuckGo."
            
        formatted_results = []
        for r in results:
            link = r.get('link', 'No link available')
            body = r['body'][:200] + '...' if 'body' in r else ''
            formatted_results.append(f"- {r['title']}\n  {body}\n  Source: {link}")
        
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"An error occurred while searching DuckDuckGo: {str(e)}"

def safe_math_eval(expression: str) -> str:
    """Safely evaluate mathematical expressions"""
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Invalid mathematical expression. Only basic operations are allowed."
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating mathematical expression: {str(e)}"

def search_arxiv(query: str) -> str:
    """Search ARXiv with improved response parsing"""
    if not query:
        return "No query provided."
    try:
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=3"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.text)
        results = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
            link = entry.find('{http://www.w3.org/2005/Atom}id').text
            results.append(f"Title: {title}\nLink: {link}\nSummary: {summary[:200]}...\n")
        
        return "\n".join(results) if results else "No results found on arXiv."
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"

def search_pubmed(query: str) -> str:
    """Search PubMed with improved response handling"""
    if not query:
        return "No query provided."
    try:
        esearch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax=3&format=json"
        response = requests.get(esearch_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        ids = data.get('esearchresult', {}).get('idlist', [])
        
        if not ids:
            return "No results found on PubMed."
            
        ids_string = ",".join(ids)
        esummary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={ids_string}&format=json"
        response = requests.get(esummary_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        for id in ids:
            paper = data['result'][id]
            title = paper.get('title', 'No title available')
            abstract = paper.get('abstract', 'No abstract available')
            results.append(f"Title: {title}\nPubMed ID: {id}\nAbstract: {abstract[:200]}...\n")
            
        return "\n".join(results)
    except Exception as e:
        return f"Error searching PubMed: {str(e)}"

# Define the tools
tools = [
    Tool(
        name="Wikipedia",
        func=safe_wikipedia_search,
        description="Get detailed explanations and summaries from Wikipedia. who, what, when, where, why, how, story.",
    ),
    Tool(
        name="DuckDuckGo",
        func=safe_duckduckgo_search,
        description="Search the web for current information. Input: search query. who, what, when, where, why, how.",
    ),
    Tool(
        name="BasicMath",
        func=safe_math_eval,
        description="Performs basic mathematical calculations. Input: mathematical expression using +, -, *, /, ().",
    ),
    Tool(
        name="ArXiv",
        func=search_arxiv,
        description="Searches arXiv for scientific papers. research paper topic or keywords.",
    ),
    Tool(
        name="PubMed",
        func=search_pubmed,
        description="Searches PubMed for medical research. research paper on medical topic or keywords.",
    ),
]

tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

CUSTOM_PROMPT = f"""You are a helpful research assistant that finds information using the available tools. Follow these guidelines strictly:

Available tools:

{tool_descriptions}

Instructions:
. Details from Internal Knowledge: Related information you already know
. Add line breaks between sections
. Use the exact tool name from the list above
. Keep queries simple and clear
. Use tools according to the type of information needed
. In case of research paper only use ArXiv and Pubmed
. Summarize information from multiple sources when relevant
. Stop as soon as you have a clear answer with reference links

Use this exact format:

Question: the input question you must answer
Thought: analyze the question and decide which tool to use
Action: use EXACTLY one of these tools: Wikipedia, DuckDuckGo, BasicMath, ArXiv, or PubMed
Action Input: just the plain search query or math expression
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat up to 3 times if needed)
Thought: I now know the final answer
Final Answer: provide a complete answer based on the information gathered with link (Details from Web Search: 
- First result title and brief description
- Second result title and brief description

References:
[1] link1
[2] link2
[etc])

Begin!

Question: {{input}}
Thought:"""

# Initialize the agent
agent = initialize_agent(
    tools,
    llama,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "prefix": CUSTOM_PROMPT,
        "max_iterations": 3,
        "early_stopping_method": "generate",

    },
    handle_parsing_errors=True
)

@app.route('/research')
def research():
    return render_template('research.html', tools=tools)

@app.route('/search', methods=['POST'])
def search():
    try:
        user_input = request.json.get('query')
        if not user_input:
            return jsonify({'error': 'No query provided'}), 400
        
        result = agent.run(user_input)
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
    return render_template('quizpage.html', active_page='quizpage')

if __name__ == '__main__':
    app.run(debug=True)