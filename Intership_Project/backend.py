import streamlit as st
import random
import sqlite3
import re
import os
import cv2
import numpy as np
import face_recognition
from PIL import Image
from twilio.rest import Client
from datetime import datetime
import base64
import smtplib
import ssl
from email.message import EmailMessage
import pytesseract
import requests
import json
import time
from gtts import gTTS  # Google Text-to-Speech
import io
import pygame  # For playing audio

# Initialize pygame mixer
pygame.mixer.init()

# Configure page
st.set_page_config(
    page_title="ABC Bank Chat Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Text-to-Speech Functions ---
import threading  # Add this at the top with other imports

# --- Modified Text-to-Speech Functions ---
def text_to_speech(text, lang='en'):
    """Convert text to speech and play it (modified to run in thread)"""
    try:
        # Create gTTS object
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # Save to bytes buffer
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        # Play audio
        pygame.mixer.music.load(audio_bytes)
        pygame.mixer.music.play()
        
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")

def speak_response(response):
    """Speak the assistant's response if TTS is enabled (runs in background thread)"""
    if st.session_state.get('tts_enabled', True):
        # Create and start a thread for TTS
        tts_thread = threading.Thread(target=text_to_speech, args=(response,))
        tts_thread.daemon = True  # This ensures the thread won't prevent program exit
        tts_thread.start()


# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Ollama Mistral Integration ---
class MistralChatbot:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "mistral"
        self.system_prompt = """You are ABC Bank's helpful assistant. You help customers with account opening process.

IMPORTANT RULES:
1. You MUST follow the exact account opening steps in this order:
   - Step 1: Ask for full name
   - Step 2: Ask for age (minimum 18 years)
   - Step 3: Face capture/upload
   - Step 4: PAN card upload and OCR
   - Step 5: Email verification with OTP
   - Step 6: Account type selection (Savings/Current)
   - Step 7: Mobile number and SMS OTP verification
   - Step 8: Account creation and number generation

2. Be conversational and friendly, but stay focused on the banking process.
3. If user asks about existing account, help them login with phone number.
4. Always ask for one piece of information at a time.
5. Be encouraging and helpful throughout the process.
6. Use banking terminology appropriately.
7. Keep responses concise and direct - no more than 2 sentences.

Current conversation context: You are helping with account opening process."""

    def get_response(self, user_input, step, context=None):
        # Create context-aware prompts based on step
        step_prompts = {
            0: "Greet the user and ask for their full name to start account opening.",
            1: "Ask for the user's age. Remind them minimum age is 18 years.",
            2: "Ask the user to upload a photo or use webcam for face verification.",
            3: "Ask the user to upload their PAN card image for verification.",
            4: "Ask for email address to send OTP for verification.",
            5: "Ask the user to choose account type: Savings or Current account.",
            6: "Ask for mobile number in format +91XXXXXXXXXX to send SMS OTP.",
            7: "Ask the user to enter the OTP they received on their mobile.",
            8: "Congratulate on successful account creation and provide account details."
        }

        context_prompt = step_prompts.get(step, "Help the user with their banking query.")

        prompt = f"""
        {self.system_prompt}

        Current step: {step}
        Context: {context_prompt}
        User input: {user_input}

        Respond naturally and conversationally. Keep responses very concise (1-2 sentences max).
        """

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=5
            )

            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                return self.get_fallback_response(step)

        except Exception as e:
            return self.get_fallback_response(step)

    def get_fallback_response(self, step):
        fallback_responses = {
            0: "Hello! Welcome to ABC Bank. Please tell me your full name to start.",
            1: "Great! What's your age? (Must be 18+)",
            2: "Perfect! Now upload your photo or use webcam for face verification.",
            3: "Excellent! Please upload a clear image of your PAN card.",
            4: "Great! What's your email address for verification?",
            5: "Perfect! Choose account type: Savings or Current?",
            6: "Excellent! Enter your mobile number (+91XXXXXXXXXX) for SMS verification.",
            7: "Enter the OTP sent to your mobile number.",
            8: "🎉 Account created successfully!"
        }
        return fallback_responses.get(step, "How can I help you today?")

# Initialize chatbot
@st.cache_resource
def init_chatbot():
    return MistralChatbot()

# --- Utility Functions ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except:
        return None

def init_db():
    conn = sqlite3.connect("bank_accounts.db")
    c = conn.cursor()

    # Create table if it doesn't exist
    c.execute("""CREATE TABLE IF NOT EXISTS accounts (
                name TEXT,
                age INTEGER,
                phone TEXT,
                pan_card TEXT,
                account_type TEXT,
                account_number TEXT,
                email TEXT,
                face_encoding BLOB
                )""")

    conn.commit()
    conn.close()

def generate_account_number():
    return f"ACC{random.randint(1000000000, 9999999999)}"

def save_to_db(name, age, phone, pan_card, account_type, account_number, email, face_encoding):
    conn = sqlite3.connect("bank_accounts.db")
    c = conn.cursor()
    c.execute("""INSERT INTO accounts
                 (name, age, phone, pan_card, account_type, account_number, email, face_encoding)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
              (name, age, phone, pan_card, account_type, account_number, email, face_encoding))
    conn.commit()
    conn.close()

def fetch_user_by_phone(phone):
    conn = sqlite3.connect("bank_accounts.db")
    c = conn.cursor()
    c.execute("SELECT * FROM accounts WHERE phone = ?", (phone,))
    result = c.fetchone()
    conn.close()
    return result

def is_valid_pan(pan):
    return bool(re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", pan))

def save_face_image(name, image):
    folder = "known_faces"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
    image.save(filename)
    return filename

def get_face_encoding(image):
    img_array = np.array(image)
    rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_img)
    if len(face_encodings) > 0:
        return face_encodings[0]
    return None

def get_all_known_face_encodings():
    """Get all face encodings from the database to check for duplicates"""
    conn = sqlite3.connect("bank_accounts.db")
    c = conn.cursor()
    c.execute("SELECT face_encoding FROM accounts WHERE face_encoding IS NOT NULL")
    results = c.fetchall()
    conn.close()

    encodings = []
    for result in results:
        if result[0]:  # Check if not None
            try:
                encodings.append(np.frombuffer(result[0], dtype=np.float64))
            except:
                continue
    return encodings

def check_for_duplicate_faces(new_encoding, threshold=0.6):
    """Check if the new face matches any existing faces in the database"""
    known_encodings = get_all_known_face_encodings()
    if not known_encodings or new_encoding is None:
        return False

    matches = face_recognition.compare_faces(known_encodings, new_encoding, tolerance=threshold)
    return any(matches)

def compare_faces(known_encoding, unknown_image):
    try:
        unknown_encoding = get_face_encoding(unknown_image)
        if known_encoding is None or unknown_encoding is None:
            return False

        results = face_recognition.compare_faces([known_encoding], unknown_encoding)
        return results[0] if results else False
    except Exception as e:
        st.error(f"Error comparing faces: {e}")
        return False

def send_email_otp(receiver_email, otp):
    sender_email = "ramyaputta9618@gmail.com"
    sender_password = "fvrf evol bqdo yave"

    message = EmailMessage()
    message.set_content(f"Your OTP for ABC Bank Signup is: {otp}")
    message["Subject"] = "ABC Bank Email Verification OTP"
    message["From"] = sender_email
    message["To"] = receiver_email

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, sender_password)
        server.send_message(message)

def extract_pan_and_name_from_image(image):
    """
    Extracts PAN card number and attempts to extract name from the image.
    Returns a tuple: (extracted_pan, extracted_name)
    """
    text = pytesseract.image_to_string(image, lang='eng')

    pan_number = None
    name = None

    # Regex for PAN number
    pan_match = re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", text)
    if pan_match:
        pan_number = pan_match.group(0)

    # Attempt to extract name (This is heuristic and might need refinement)
    lines = text.split('\n')

    # Heuristic approach for name extraction:
    # Look for lines that are not too short, not numeric, and often appear near "Name" or "Father's Name"
    # PAN cards usually have "Name" or "Father's Name" followed by the actual name.
    
    pan_index = -1
    if pan_number:
        for i, line in enumerate(lines):
            if pan_number in line:
                pan_index = i
                break
    
    # Try to find name line based on common PAN card layouts
    for i, line in enumerate(lines):
        line_upper = line.upper().strip()
        # Look for "NAME" or "FATHER'S NAME" or similar identifiers
        if "NAME" in line_upper and len(line_upper) < 20: # Short line likely an identifier
            # The actual name is often on the next line or same line after a colon
            if i + 1 < len(lines):
                potential_name = lines[i+1].strip()
                # Basic filtering: ensure it's not a number, not too short, not a common keyword
                if len(potential_name) > 3 and not re.match(r'^\d+$', potential_name) and \
                   not any(kw in potential_name.upper() for kw in ["INCOME", "TAX", "PERMANENT", "ACCOUNT", "NUMBER"]):
                    name = potential_name
                    break
        elif pan_index != -1 and i > pan_index - 3 and i < pan_index: # Check lines just above PAN
             # This is a very loose heuristic, might need tuning based on actual PAN card images
            potential_name = line.strip()
            if len(potential_name.split()) >= 2 and len(potential_name) > 5 and not re.match(r'^\d+$', potential_name) and \
               not any(kw in potential_name.upper() for kw in ["INCOME", "TAX", "PERMANENT", "ACCOUNT", "NUMBER"]):
                name = potential_name
                # Take the first plausible name, as there might be other text.
                # Often, the name is the line right before the PAN number or 2 lines above.
                break

    # Clean up extracted name (remove extra spaces, symbols, and convert to title case)
    if name:
        name = re.sub(r'[^a-zA-Z\s]', '', name).strip() # Remove non-alphabetic characters except space
        name = re.sub(r'\s+', ' ', name).strip() # Replace multiple spaces with single
        name = name.title() # Convert to Title Case for consistent comparison

    return pan_number, name

def fuzzy_name_match(name1, name2):
    """
    Performs fuzzy matching between two names with some tolerance for minor differences
    Returns True if names match with reasonable similarity
    """
    # Remove extra spaces and convert to lowercase
    name1 = re.sub(r'\s+', ' ', name1.strip()).lower()
    name2 = re.sub(r'\s+', ' ', name2.strip()).lower()
    
    # Exact match
    if name1 == name2:
        return True
    
    # Split into parts and check if one contains the other
    parts1 = name1.split()
    parts2 = name2.split()
    
    # Check if all parts of one name exist in the other
    if all(part in name2 for part in parts1) or all(part in name1 for part in parts2):
        return True
    
    # Check initials match (for cases like "A B" vs "A B C")
    if len(parts1) > 1 and len(parts2) > 1:
        if all(p1[0] == p2[0] for p1, p2 in zip(parts1, parts2)):
            return True
    
    return False

# --- Enhanced CSS for Better Chat Interface ---
def load_css():
    st.markdown(r"""
    <style>
    /* Main App Background with Image and Overlay */
    .stApp {
        background-image: url("https://cdn2.vectorstock.com/i/1000x1000/89/46/robot-virtual-assistance-or-chatbot-background-vector-39118946.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center center;
        font-family: 'Segoe UI', 'Arial', sans-serif;
        min-height: 100vh;
        position: relative;
        z-index: 1;
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(0,0,0,0.9), rgba(26,0,51,0.8));
        z-index: -1;
    }

    /* Remove default streamlit padding and spacing */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 1200px;
        gap: 0.2rem !important;
    }

    /* Compact header styling */
    .main-header {
        background: linear-gradient(135deg, #000000 0%, #2d0052 30%, #4a0080 70%, #000000 100%);
        backdrop-filter: blur(20px);
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin-bottom: 0.3rem;
        border: 2px solid #6600cc;
        box-shadow: 0 8px 32px rgba(102, 0, 204, 0.4);
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    .main-header h1 {
        color: #ffffff;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        text-shadow: 2px 2px 8px rgba(102, 0, 204, 0.8);
        letter-spacing: 1px;
    }

    .main-header p {
        color: #e6ccff;
        margin: 0.3rem 0 0 0;
        font-size: 1rem;
        font-weight: 400;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7);
    }

    .chat-container {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        padding: 0.8rem;
        height: 68vh;
        overflow-y: auto;
        margin-bottom: 0.3rem;
        border: 1px solid rgba(102, 0, 204, 0.3);
        backdrop-filter: blur(10px);
    }

    .chat-container::-webkit-scrollbar {
        width: 6px;
    }

    .chat-container::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
    }

    .chat-container::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6600cc, #9933ff);
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(102, 0, 204, 0.3);
    }

    .chat-container::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #9933ff, #cc66ff);
    }

    .bot-message {
        background: linear-gradient(135deg, #1a0033 0%, #330066 50%, #1a0033 100%);
        color: #ffffff;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.2rem 0;
        max-width: 80%;
        animation: slideInLeft 0.5s ease;
        box-shadow: 0 4px 20px rgba(51, 0, 102, 0.5);
        border: 1px solid #6600cc;
        font-size: 0.95rem;
        line-height: 1.5;
        backdrop-filter: blur(15px);
        position: relative;
    }

    .bot-message::before {
        content: '🤖';
        position: absolute;
        top: -5px;
        left: -5px;
        font-size: 0.7rem;
        background: linear-gradient(135deg, #6600cc, #9933ff);
        border-radius: 50%;
        padding: 2px;
        width: 16px;
        height: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .user-message {
        background: linear-gradient(135deg, #000000 0%, #2d0052 50%, #000000 100%);
        color: #ffffff;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.2rem 0 0.2rem auto;
        max-width: 80%;
        animation: slideInRight 0.5s ease;
        box-shadow: 0 4px 20px rgba(45, 0, 82, 0.5);
        border: 1px solid #4a0080;
        font-size: 0.95rem;
        line-height: 1.5;
        backdrop-filter: blur(15px);
    }

    .input-container {
        background: linear-gradient(135deg, #000000 0%, #1a0033 50%, #000000 100%);
        border-radius: 25px;
        padding: 0.4rem;
        margin-top: 0.3rem;
        box-shadow: 0 6px 25px rgba(102, 0, 204, 0.4);
        border: 2px solid #6600cc;
        backdrop-filter: blur(20px);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .stTextInput > div > div > input {
        border: none !important;
        border-radius: 20px !important;
        padding: 0.8rem 1.2rem !important;
        font-size: 0.95rem !important;
        background: linear-gradient(135deg, #1a0033 0%, #000000 100%) !important;
        color: #ffffff !important;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: blur(10px) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #b3b3ff !important;
        opacity: 0.9 !important;
    }

    .stTextInput > div > div > input:focus {
        border: 2px solid #9933ff !important;
        outline: none !important;
        box-shadow: 0 0 15px rgba(153, 51, 255, 0.6) !important;
        background: linear-gradient(135deg, #2d0052 0%, #1a0033 100%) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6600cc 0%, #9933ff 50%, #6600cc 100%) !important;
        border: 2px solid #4a0080 !important;
        border-radius: 50% !important;
        color: white !important;
        width: 50px !important;
        height: 50px !important;
        font-size: 1.3rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(102, 0, 204, 0.5) !important;
        backdrop-filter: blur(10px) !important;
    }

    .stButton > button:hover {
        transform: scale(1.15) rotate(5deg) !important;
        box-shadow: 0 8px 30px rgba(153, 51, 255, 0.7) !important;
        background: linear-gradient(135deg, #9933ff 0%, #cc66ff 50%, #9933ff 100%) !important;
        border: 2px solid #ffffff !important;
    }

    .stButton > button:active {
        transform: scale(1.05) !important;
    }

    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px) scale(0.95); }
        to { opacity: 1; transform: translateX(0) scale(1); }
    }

    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px) scale(0.95); }
        to { opacity: 1; transform: translateX(0) scale(1); }
    }

    .typing-indicator {
        background: linear-gradient(135deg, #1a0033 0%, #2d0052 100%);
        color: #b3b3ff;
        padding: 0.6rem 1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.2rem 0;
        max-width: 150px;
        font-style: italic;
        animation: typingPulse 1.5s ease-in-out infinite;
        border: 1px solid #6600cc;
        backdrop-filter: blur(15px);
        font-size: 0.9rem;
    }

    @keyframes typingPulse {
        0%, 100% { opacity: 0.6; transform: scale(0.98); }
        50% { opacity: 1; transform: scale(1); }
    }

    .bot-message:hover, .user-message:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 30px rgba(102, 0, 204, 0.7);
    }

    .element-container, .stMarkdown {
        margin-bottom: 0.2rem !important;
        margin-top: 0.2rem !important;
    }

    .account-success {
        background: linear-gradient(135deg, #000000 0%, #1a0033 50%, #000000 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(51, 0, 102, 0.5);
        border: 2px solid #6600cc;
        backdrop-filter: blur(20px);
    }

    .file-uploader {
        background: linear-gradient(135deg, #1a0033 0%, #330066 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 2px solid #6600cc;
        color: white;
        backdrop-filter: blur(15px);
    }

    * {
        transition: all 0.3s ease;
    }

    #MainMenu, footer, header, .stDeployButton {
        visibility: hidden;
    }

    .bot-message *, .user-message * {
        color: #ffffff !important;
    }
</style>

    """, unsafe_allow_html=True)

# --- Main Application ---
# Replace your main() function with this updated version

def main():
    init_db()
    load_css()
    chatbot = init_chatbot()
    
    # Twilio Configuration
    account_sid = 'AC91f71e48ddd9dfb6dc2c9abfb2a47804'
    auth_token = '064fcab1c1e67d2c62f9ace9714733d7'
    verify_sid = 'VAc6db2bea2f4d8cc3c4406f516ee86304'
    client = Client(account_sid, auth_token)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello! Welcome to Deccan Axis Bank. How can I help you today?"}
        ]
    
    if "step" not in st.session_state:
        st.session_state.step = -1  # -1 for initial state
        st.session_state.user_data = {}
        st.session_state.mode = None
        st.session_state.processing = False
        st.session_state.face_encoding = None
    
    # Initialize input clearing mechanism
    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0

    # Add TTS toggle to sidebar
    st.sidebar.title("Settings")
    st.session_state.tts_enabled = st.sidebar.checkbox("Enable Text-to-Speech", value=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Deccan Axis Chat Assistant</h1>
        <p>Your friendly banking assistant - Fast & Secure</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container with auto-scroll
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="user-message">{message["content"]} 👤</div>', unsafe_allow_html=True)
        
        # Show typing indicator when processing
        if st.session_state.get('processing', False):
            st.markdown('<div class="typing-indicator">🤖 Typing...</div>', unsafe_allow_html=True)
    
    # Input section with better layout
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([8, 1])
    
    with col1:
        # Use input_counter as key to force refresh and clear input
        user_input = st.text_input(
            "", 
            placeholder="Type your message here...", 
            key=f"user_input_{st.session_state.input_counter}", 
            label_visibility="collapsed"
        )
        
    with col2:
        send_button = st.button("➤", key="send_btn", help="Send message")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process user input with improved response time
    if send_button and user_input.strip():
        # Set processing state
        st.session_state.processing = True
        
        # Add user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        
        # Process and get response quickly
        response = process_conversation(user_input.strip(), chatbot, client, verify_sid)
        
        # Add bot response
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Speak the response if TTS is enabled
            speak_response(response)
        
        # Clear processing state and increment counter to clear input
        st.session_state.processing = False
        st.session_state.input_counter += 1  # This will clear the input field
        
        # Rerun to update the UI
        st.rerun()
    
    # Handle Enter key press (alternative method)
    if user_input and user_input.strip():
        # Check if this is a new input (different from last processed)
        if not hasattr(st.session_state, 'last_input') or st.session_state.last_input != user_input.strip():
            # Process the input automatically when Enter is pressed
            st.session_state.last_input = user_input.strip()
    
    # Handle special UI elements based on step
    render_step_specific_ui(client, verify_sid, chatbot)
    
    # Auto-scroll to bottom
    st.markdown("""
    <script>
    var chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    </script>
    """, unsafe_allow_html=True)

def process_conversation(user_input, chatbot, client, verify_sid):
    # Check for initial commands
    if st.session_state.step == -1:
        if any(word in user_input.lower() for word in ["new account", "open account", "signup", "register", "create account"]):
            st.session_state.mode = "signup"
            st.session_state.step = 0
            return "Perfect! Let's get started. What's your full name?"
        else:
            return """I can help you with:

🆕 Open a new account - Say "new account"  

What would you like to do?"""
    

    # Handle signup flow
    if st.session_state.mode == "signup":
        return handle_signup_step(user_input, chatbot, client, verify_sid)
    
    return "How can I help you today?"

def handle_signup_step(user_input, chatbot, client, verify_sid):
    step = st.session_state.step
    
    if step == 0:  # Name
        st.session_state.user_data['name'] = user_input.strip()
        st.session_state.step = 1
        return f"Nice to meet you, {user_input.strip()}! What's your age?"
    
    elif step == 1:  # Age
        try:
            age = int(user_input.strip())
            if age < 18:
                return "⚠ You must be 18+ to open an account. Please verify your age."
            st.session_state.user_data['age'] = age
            st.session_state.step = 2
            return "Great! Now let's verify your identity with a photo."
        except ValueError:
            return "Please enter your age in numbers only."
    
    elif step == 2:  # Face verification handled in UI
        st.session_state.step = 3
        return "Face verified! Now please upload your PAN card."
    
    elif step == 3:  # PAN verification handled in UI
        st.session_state.step = 4
        return "PAN verified! What's your email address?"
    
    elif step == 4:  # Email
        if "@" in user_input and "." in user_input:
            st.session_state.user_data['email'] = user_input.strip()
            otp = str(random.randint(100000, 999999))
            st.session_state.email_otp = otp
            try:
                send_email_otp(user_input.strip(), otp)
                st.session_state.step = 41  # Email OTP verification
                return f"📧 OTP sent to {user_input}! Enter the 6-digit code:"
            except Exception as e:
                return "Failed to send email. Please try again."
        else:
            return "Please enter a valid email address."
    
    elif step == 41:  # Email OTP verification
        if user_input.strip() == str(st.session_state.get('email_otp', '')):
            st.session_state.step = 5
            return "✅ Email verified! Choose account type: Savings or Current?"
        else:
            return "❌ Incorrect OTP. Please try again."
    
    elif step == 5:  # Account type
        if user_input.lower() in ['savings', 'current']:
            st.session_state.user_data['account_type'] = user_input.capitalize()
            st.session_state.step = 6
            return f"{user_input.capitalize()} account selected! Enter mobile number (+91XXXXXXXXXX):"
        else:
            return "Please choose 'Savings' or 'Current'."
    
    elif step == 6:  # Mobile number
        if re.match(r"^\+91\d{10}$", user_input.strip()):
            # Check if account already exists
            existing_user = fetch_user_by_phone(user_input.strip())
            if existing_user:
                return f"⚠ Account exists with this number! Your account: {existing_user[5]}"
            
            st.session_state.user_data['phone'] = user_input.strip()
            try:
                verification = client.verify.services(verify_sid).verifications.create(
                    to=user_input.strip(), 
                    channel="sms"
                )
                st.session_state.verification_sid = verification.sid
                st.session_state.step = 7
                return "📱 SMS OTP sent! Enter the code:"
            except Exception as e:
                return f"Failed to send SMS: {str(e)}"
        else:
            return "Format: +91XXXXXXXXXX"
    
    elif step == 7:  # SMS OTP verification
        try:
            verification_check = client.verify.services(verify_sid).verification_checks.create(
                to=st.session_state.user_data['phone'], 
                code=user_input.strip()
            )
            if verification_check.status == "approved":
                # Create account
                account_number = generate_account_number()
                st.session_state.user_data['account_number'] = account_number
                
                # Convert face encoding to bytes for storage
                face_encoding_bytes = None
                if st.session_state.face_encoding is not None:
                    face_encoding_bytes = np.array(st.session_state.face_encoding).tobytes()
                
                save_to_db(
                    st.session_state.user_data['name'],
                    st.session_state.user_data['age'],
                    st.session_state.user_data['phone'],
                    st.session_state.user_data['pan_card'],
                    st.session_state.user_data['account_type'],
                    account_number,
                    st.session_state.user_data['email'],
                    face_encoding_bytes
                )
                
                st.session_state.step = 8
                return f"""🎉 Account Created Successfully!

Your Details:
- Name: {st.session_state.user_data['name']}
- Account Number: {account_number}
- Type: {st.session_state.user_data['account_type']}

Welcome to Deccan Axis Bank! 🏦"""
            else:
                return "❌ Invalid OTP. Please try again."
        except Exception as e:
            return f"Error verifying OTP: {str(e)}"
    
    return "How can I help you?"

def render_step_specific_ui(client, verify_sid, chatbot):
    """Render step-specific UI elements with improved styling"""
    
    # Face capture for step 2
    if st.session_state.get('step') == 2:
        st.markdown("### 📸 Face Verification")
        
        upload_method = st.radio("Choose method:", ["📤 Upload Photo", "📷 Use Webcam"], key="face_upload_method")
        
        if upload_method == "📤 Upload Photo":
            uploaded_file = st.file_uploader("Upload clear photo", type=["jpg", "png", "jpeg"], key="face_uploader")
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Your Photo", width=250)
                
                # Get face encoding
                face_encoding = get_face_encoding(image)
                if face_encoding is not None:
                    # Check for duplicate faces
                    if check_for_duplicate_faces(face_encoding):
                        st.error("⚠ WARNING: This face matches an existing account. Duplicate accounts are not allowed.")
                        st.warning("If this is a mistake, please contact customer support.")
                        return
                    
                    st.session_state.face_encoding = face_encoding
                    filename = save_face_image(st.session_state.user_data['name'], image)
                    st.success("✅ Face detected and saved!")
                    if st.button("✅ Continue", key="face_continue_upload"):
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Face verified! Now please upload your PAN card."
                        })
                        st.session_state.step = 3
                        st.rerun()
                else:
                    st.warning("⚠ No face detected. Please upload a clear photo with your face visible.")
        
        elif upload_method == "📷 Use Webcam":
            captured_image = st.camera_input("Capture photo", key="face_webcam")
            if captured_image:
                image = Image.open(captured_image)
                
                # Get face encoding
                face_encoding = get_face_encoding(image)
                if face_encoding is not None:
                    # Check for duplicate faces
                    if check_for_duplicate_faces(face_encoding):
                        st.error("⚠ WARNING: This face matches an existing account. Duplicate accounts are not allowed.")
                        st.warning("If this is a mistake, please contact customer support.")
                        return
                    
                    st.session_state.face_encoding = face_encoding
                    filename = save_face_image(st.session_state.user_data['name'], image)
                    st.success("✅ Face captured and saved!")
                    if st.button("✅ Continue", key="webcam_continue_capture"):
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Face verified! Now please upload your PAN card."
                        })
                        st.session_state.step = 3
                        st.rerun()
                else:
                    st.warning("⚠ No face detected. Try better lighting and position your face clearly.")
    
    # PAN card upload for step 3 with face matching and name verification
    elif st.session_state.get('step') == 3:
        st.markdown("### 📄 PAN Card Verification")
        
        uploaded_pan = st.file_uploader("Upload PAN card image (with photo)", type=["jpg", "jpeg", "png", "webp"], key="pan_uploader")
        
        if uploaded_pan:
            image = Image.open(uploaded_pan)
            st.image(image, caption="PAN Card Image", width=400)
            
            # First verify PAN card has a face that matches the stored face
            with st.spinner("Verifying face on PAN card..."):
                pan_face_match = False
                if st.session_state.face_encoding is not None:
                    pan_face_match = compare_faces(st.session_state.face_encoding, image)
                
                if not pan_face_match:
                    st.error("❌ Face on PAN card doesn't match your uploaded photo. Please ensure it's a clear image of your own PAN card.")
                    return # Stop here if face doesn't match
            
            st.success("✅ Face on PAN card matched!")
            
            # Now, extract PAN details and name using OCR
            with st.spinner("Extracting PAN details and Name..."):
                extracted_pan_number, extracted_name_from_pan = extract_pan_and_name_from_image(image)
                
                # Clean and standardize the initially entered name for robust comparison
                entered_name = st.session_state.user_data.get('name', '').upper().strip()
                entered_name = re.sub(r'[^a-zA-Z\s]', '', entered_name)
                entered_name = re.sub(r'\s+', ' ', entered_name).strip()
                
                # Clean and standardize the extracted name from PAN for robust comparison
                cleaned_extracted_name = None
                if extracted_name_from_pan:
                    cleaned_extracted_name = extracted_name_from_pan.upper().strip()
                    cleaned_extracted_name = re.sub(r'[^a-zA-Z\s]', '', cleaned_extracted_name)
                    cleaned_extracted_name = re.sub(r'\s+', ' ', cleaned_extracted_name).strip()
                
                name_match = False
                if cleaned_extracted_name and entered_name:
                    # Use fuzzy matching
                    name_match = fuzzy_name_match(cleaned_extracted_name, entered_name)
            
            if not extracted_pan_number or not is_valid_pan(extracted_pan_number):
                st.warning("⚠ Could not detect a valid PAN number from the image. Please upload a clearer image of your PAN card.")
            elif not cleaned_extracted_name:
                st.warning("⚠ Could not accurately extract your name from the PAN card. Please ensure your name is clearly visible.")
                st.info(f"Entered Name (for reference): *{st.session_state.user_data['name']}*") # Show user what they entered
            elif not name_match:
                st.error(f"❌ Name mismatch! The name on the PAN card ({extracted_name_from_pan or 'Not Detected'}) does not match the name you entered ({st.session_state.user_data['name']}).")
                st.warning("Please ensure the PAN card belongs to you and the name entered initially is correct.")
            else:
                st.success(f"✅ PAN Card detected: *{extracted_pan_number}*")
                st.success(f"✅ Name on PAN matches entered name: *{extracted_name_from_pan}*")
                st.session_state.user_data['pan_card'] = extracted_pan_number
                if st.button("✅ Continue", key="pan_continue"):
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "PAN verified! What's your email address?"
                    })
                    st.session_state.step = 4
                    st.rerun()

if __name__ == "__main__":
    main()