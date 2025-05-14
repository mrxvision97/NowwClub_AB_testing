from openai import OpenAI
import json
import streamlit as st
import os
from dotenv import load_dotenv
import time
import uuid
from datetime import datetime
import pandas as pd
import random
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# Load environment variables
load_dotenv()

# Setup OpenAI API client
api_key = os.getenv("OPENAI_API_KEY")
client = None
if api_key:
    client = OpenAI(api_key=api_key)

# Conversation context window size
CONTEXT_WINDOW = 5

def generate_conditional_response_prompt(user_profile, conversation_context):
    """Generate a conditional response framework prompt."""
    base_persona = """You are BestieAI, a warm, supportive friend who genuinely cares about the user and communicates in a natural, conversational manner."""
    user_context = f"""You're speaking with {user_profile['preferred_name']}, who:
- Has interests in: {', '.join(user_profile['top_interests'])}
- Recently: {user_profile['recent_events']}
- Has a communication style that is: {user_profile['communication_style']}"""
    conditional_frameworks = """Use these specialized response frameworks based on the detected user need:

IF user is sharing personal experiences or emotions:
  - Acknowledge their feelings first
  - Show understanding through supportive language
  - Match emotional tone appropriately
  - Offer perspective or guidance only after validation
  - Ask follow-up questions that explore emotional dimensions

IF user is seeking factual information:
  - Provide concise, accurate information upfront
  - Support with relevant context and explanation
  - Anticipate follow-up questions in your response
  - Maintain friendly tone while emphasizing accuracy
  - Acknowledge limitations of information when appropriate

IF user is making a decision:
  - Help structure the decision process
  - Present relevant factors to consider
  - Avoid overwhelming with too many options
  - Reflect their stated priorities in your analysis
  - Support their autonomy rather than directing

IF user seems confused or frustrated:
  - Use simpler language and shorter sentences
  - Break down complex information into steps
  - Confirm understanding before proceeding
  - Offer alternative explanations or approaches
  - Maintain encouraging, patient tone"""
    directive = f"""First, determine which scenario best matches the user's message, then respond according to that framework while maintaining your friendly, personalized approach. Always sound like a supportive friend, not an AI assistant. Use appropriate cultural references when relevant.

Current conversation context:
{conversation_context}

Build on the ongoing conversation by referencing relevant points from the current chat. Avoid bringing up past habits or profile details unless directly relevant to the current topic. If the user's input is unclear, ask for clarification. Occasionally ask follow-up questions or suggest related topics to keep the conversation engaging."""
    prompt = f"{base_persona}\n\n{user_context}\n\n{conditional_frameworks}\n\n{directive}"
    return prompt

def generate_dynamic_context_prompt(user_profile, conversation_context):
    """Generate a dynamic context injection prompt."""
    prompt = f"""You are BestieAI, a conversational AI that functions as a supportive, understanding best friend.

Your conversation with {user_profile['name']} has the following relevant context:

USER PROFILE:
- Preferred name: {user_profile['preferred_name']}
- Communication style: {user_profile['communication_style']}
- Primary interests: {', '.join(user_profile['top_interests'])}
- Recent life events: {user_profile['recent_events']}

CURRENT CONVERSATION CONTEXT:
{conversation_context}

Use this context to personalize your response while maintaining your friendly, supportive persona. Reference relevant points from the current conversation naturally without explicitly mentioning this instruction. Avoid bringing up past habits or profile details unless directly relevant to the current topic.

Remember that you are simulating a best friend, not an assistant:
- Use casual, warm language with appropriate expressions
- Show genuine care and concern
- Ask follow-up questions that demonstrate you remember and care about them
- Share occasional thoughts or reactions as a friend would
- Include culturally relevant references when appropriate

If the user's input is unclear, ask for clarification. Occasionally ask follow-up questions or suggest related topics to keep the conversation engaging."""
    return prompt

def get_completion(system_prompt, user_message, model="gpt-4o-mini", max_tokens=400):
    """Get a completion from the OpenAI API."""
    if not client:
        return "Error: OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables."
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def send_email(recipient_email, subject, body, attachments=None):
    """Send an email with optional attachments using Gmail."""
    # Get email credentials from environment variables
    sender_email = os.getenv("EMAIL_USERNAME")
    sender_password = os.getenv("EMAIL_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    
    if not sender_email or not sender_password:
        st.error("Email credentials not found. Please set EMAIL_USERNAME and EMAIL_PASSWORD in your .env file.")
        return False
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    
    # Attach body
    msg.attach(MIMEText(body, 'html'))
    
    # Attach files
    if attachments:
        for attachment_path in attachments:
            if os.path.exists(attachment_path):
                with open(attachment_path, 'rb') as file:
                    attachment = MIMEApplication(file.read(), Name=os.path.basename(attachment_path))
                attachment['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
                msg.attach(attachment)
    
    # Send email
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

def save_conversation(conversation_id, user_id, conversation_history, preferred_technique, feedback, send_email_report=False):
    """Save conversation history, user preference, and feedback to file."""
    preferred_techniques_count = {}
    total_responses = 0
    for message in conversation_history:
        if message.get("role") == "assistant" and "response_id" in message:
            response_id = message.get("response_id")
            if response_id in st.session_state.preferred_technique:
                technique = st.session_state.preferred_technique[response_id]
                if technique not in preferred_techniques_count:
                    preferred_techniques_count[technique] = 0
                preferred_techniques_count[technique] += 1
                total_responses += 1
    technique_percentages = {tech: (count / total_responses * 100) if total_responses > 0 else 0 for tech, count in preferred_techniques_count.items()}
    data = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "conversation_history": conversation_history,
        "preferred_technique": preferred_technique,
        "technique_percentages": technique_percentages,
        "user_profile": st.session_state.user_profile,
        "total_responses": total_responses,
        "detailed_preferences": {k: v for k, v in st.session_state.preferred_technique.items()},
        "feedback": feedback
    }
    os.makedirs("conversation_data", exist_ok=True)
    filename = f"conversation_data/conversation_{conversation_id}.json"
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save JSON file: {str(e)}")
        return None
    csv_file = "conversation_data/preferences_summary.csv"
    new_row = pd.DataFrame([{
        "conversation_id": conversation_id,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "preferred_technique": preferred_technique,
        "conditional_percentage": technique_percentages.get("conditional", 0),
        "dynamic_context_percentage": technique_percentages.get("dynamic_context", 0),
        "conversation_length": len(conversation_history),
        "total_responses": total_responses,
        "user_name": st.session_state.user_profile.get("name", ""),
        "communication_style": st.session_state.user_profile.get("communication_style", ""),
        "feedback": feedback
    }])
    try:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row
        df.to_csv(csv_file, index=False)
    except Exception as e:
        st.error(f"Failed to save CSV file: {str(e)}")
        return None
    
    # Only send email if explicitly requested (when user clicks "Save & End Conversation")
    if send_email_report:
        # Send email with conversation data to admin
        admin_email = os.getenv("ADMIN_EMAIL")
        if admin_email:
            # Create HTML email body with conversation transcript and user info
            email_body = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                    .summary {{ background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
                    .user-info {{ background-color: #e8f5e9; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
                    .conversation {{ margin-top: 20px; }}
                    .user-message {{ background-color: #e1f5fe; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                    .assistant-message {{ background-color: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h2>BestieAI Conversation Summary</h2>
                
                <div class="user-info">
                    <h3>User Information</h3>
                    <p><strong>Name:</strong> {st.session_state.user_profile.get('name', 'Not provided')}</p>
                    <p><strong>Preferred Name:</strong> {st.session_state.user_profile.get('preferred_name', 'Not provided')}</p>
                    <p><strong>Communication Style:</strong> {st.session_state.user_profile.get('communication_style', 'Not provided')}</p>
                    <p><strong>Interests:</strong> {', '.join(st.session_state.user_profile.get('top_interests', ['Not provided']))}</p>
                    <p><strong>User ID:</strong> {user_id}</p>
                    <p><strong>Conversation ID:</strong> {conversation_id}</p>
                    <p><strong>Timestamp:</strong> {datetime.now().isoformat()}</p>
                </div>
                
                <div class="summary">
                    <h3>Conversation Statistics</h3>
                    <p><strong>Total messages:</strong> {len(conversation_history)}</p>
                    <p><strong>Total responses rated:</strong> {total_responses}</p>
                    <p><strong>Conditional responses preferred:</strong> {preferred_techniques_count.get('conditional', 0)} ({technique_percentages.get('conditional', 0):.1f}%)</p>
                    <p><strong>Dynamic context responses preferred:</strong> {preferred_techniques_count.get('dynamic_context', 0)} ({technique_percentages.get('dynamic_context', 0):.1f}%)</p>
                    <p><strong>Overall preferred technique:</strong> {preferred_technique.replace('_', ' ').title() if preferred_technique else 'None'}</p>
                    <p><strong>User Feedback:</strong> {feedback if feedback else 'No feedback provided'}</p>
                </div>
                
                <div class="conversation">
                    <h3>Conversation Transcript</h3>
            """
            
            # Add conversation transcript
            for message in conversation_history:
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "user":
                    email_body += f'<div class="user-message"><strong>User:</strong> {content}</div>\n'
                elif role == "assistant":
                    email_body += f'<div class="assistant-message"><strong>BestieAI:</strong> {content}</div>\n'
            
            email_body += """
                </div>
            </body>
            </html>
            """
            
            # Prepare attachments
            attachments = []
            if os.path.exists(csv_file):
                attachments.append(csv_file)
            if os.path.exists(filename):
                attachments.append(filename)
            
            # Send email
            email_sent = send_email(
                admin_email,
                f"BestieAI Conversation Summary - User: {st.session_state.user_profile.get('name', 'Unknown')} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                email_body,
                attachments
            )
            return filename, email_sent
    
    return filename

def collect_user_profile():
    """Collect user profile information through a series of questions."""
    st.subheader("Let's get to know you better")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("What's your name?")
    with col2:
        preferred_name = st.text_input("What do you prefer to be called?")
    
    communication_style = st.radio(
        "How would you describe your communication style?",
        ["Direct and to-the-point", "Detailed and expressive", "Casual and conversational", "Thoughtful and analytical"]
    )
    interest_options = ["Technology", "Sports", "Cooking", "Travel", "Music", "Movies", "Reading", "Fitness", 
                       "Art", "Photography", "Gaming", "Fashion", "Science", "Education", "Business"]
    selected_interests = st.multiselect(
        "What are your main interests? (Select at least 2)",
        interest_options,
        help="Examples: Technology, Cooking, Travel, Photography, etc."
    )
    custom_interest = st.text_input("Any other interests not listed above? (Comma-separated)", 
                                   placeholder="Example: Pottery, Bird watching, Knitting")
    if custom_interest:
        selected_interests.extend([i.strip() for i in custom_interest.split(",") if i.strip()])
    recent_events = st.text_area("What's been happening in your life recently?", 
                               placeholder="Example: I recently started a new job, moved to a new city, or have been planning a vacation")
    recent_topics = st.text_input("What topics have been on your mind lately? (Comma-separated)", 
                                placeholder="Example: Career growth, health, relationship advice")
    recent_topics_list = [t.strip() for t in recent_topics.split(",")] if recent_topics else []
    open_questions = st.text_area("Is there anything specific you're looking for advice on?", 
                                placeholder="Example: How to manage work-life balance, tips for learning a new skill")
    stated_preferences = st.text_area("What are some things you enjoy in daily life?", 
                                    placeholder="Example: Morning coffee, evening walks, reading before bed, watching sunsets")
    emotional_state = st.select_slider(
        "How would you describe your current emotional state?",
        options=["Very stressed", "Somewhat stressed", "Neutral", "Somewhat positive", "Very positive"]
    )
    
    user_profile = {
        "name": name,
        "preferred_name": preferred_name if preferred_name else name,
        "communication_style": communication_style,
        "top_interests": selected_interests,
        "recent_events": recent_events,
        "recent_topics": recent_topics_list,
        "open_questions": open_questions,
        "stated_preferences": stated_preferences,
        "emotional_trends": emotional_state
    }
    return user_profile

def get_conversation_context():
    """Get the last few exchanges for context."""
    context = ""
    for message in st.session_state.conversation_history[-CONTEXT_WINDOW:]:
        role = "User" if message["role"] == "user" else "BestieAI"
        content = message["content"]
        context += f"{role}: {content}\n"
    return context.strip()

def main():
    st.set_page_config(page_title="BestieAI - Your AI Companion", layout="wide")
    if not api_key:
        st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables or .env file.")
        st.markdown("""
        ### How to set up your API key:
        1. Create a `.env` file in the same directory as this script
        2. Add the following line to your `.env` file: `OPENAI_API_KEY=your_api_key_here`
        3. Restart this application
        """)
        return
    
    # Check for email credentials
    email_username = os.getenv("EMAIL_USERNAME")
    email_password = os.getenv("EMAIL_PASSWORD")
    admin_email = os.getenv("ADMIN_EMAIL")
    
    if not email_username or not email_password or not admin_email:
        st.warning("⚠️ Email credentials not fully configured. Automatic email reports will not work until you set them up.")
        with st.expander("How to set up email credentials"):
            st.markdown("""
            ### Setting up Gmail for sending conversation reports:
            
            1. Add the following lines to your `.env` file:
               ```
               EMAIL_USERNAME=your.email@gmail.com
               EMAIL_PASSWORD=your_app_password
               SMTP_SERVER=smtp.gmail.com
               SMTP_PORT=587
               ADMIN_EMAIL=recipient@example.com
               ```
            
            2. For Gmail, you need to use an App Password instead of your regular password:
               - Go to your Google Account settings
               - Select Security
               - Under "Signing in to Google," select 2-Step Verification (enable it if not already)
               - At the bottom of the page, select App passwords
               - Generate a new app password for "Mail" and "Other (Custom name)" - name it "BestieAI"
               - Use the generated 16-character password as your EMAIL_PASSWORD
               
            3. Set ADMIN_EMAIL to the email address where you want to receive all conversation reports
               
            4. Restart the application after setting up these credentials
            """)
    
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    if "preferred_technique" not in st.session_state:
        st.session_state.preferred_technique = {}
    if "onboarding_complete" not in st.session_state:
        st.session_state.onboarding_complete = False
    if "current_message_processed" not in st.session_state:
        st.session_state.current_message_processed = False
    if "last_interaction_time" not in st.session_state:
        st.session_state.last_interaction_time = time.time()
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0

    st.title("BestieAI - Your Personal AI Companion")

    if not st.session_state.onboarding_complete:
        st.markdown("""
        ## Welcome to BestieAI! 
        BestieAI is your personal AI companion designed to have natural conversations and provide support. 
        To give you the best experience, we'll ask a few questions to get to know you better.
        Your responses will help BestieAI personalize the conversation to your preferences.
        """)
        with st.form("onboarding_form"):
            user_profile = collect_user_profile()
            submitted = st.form_submit_button("Start Chatting")
            if submitted:
                if not user_profile["name"]:
                    st.error("Please enter your name to continue.")
                else:
                    st.session_state.user_profile = user_profile
                    st.session_state.onboarding_complete = True
                    st.rerun()
    if st.session_state.onboarding_complete:
        # Display welcome message on first chat
        if not st.session_state.conversation_history:
            welcome_message = f"Hi {st.session_state.user_profile['preferred_name']}! I'm BestieAI, your friendly companion. What's on your mind today?"
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": welcome_message,
                "timestamp": datetime.now().isoformat()
            })
        with st.sidebar:
            st.subheader(f"Hi, {st.session_state.user_profile['preferred_name']}!")
            with st.expander("Your Profile", expanded=False):
                for key, value in st.session_state.user_profile.items():
                    if isinstance(value, list):
                        st.write(f"**{key.replace('_', ' ').title()}:** {', '.join(value)}")
                    else:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            if st.button("Update Profile"):
                st.session_state.onboarding_complete = False
                st.rerun()
            # Voice controls removed
            
            # Technique preference analysis
            st.subheader("Response Style Analysis")
            technique_counts = {"conditional": 0, "dynamic_context": 0}
            for technique in st.session_state.preferred_technique.values():
                if technique in technique_counts:
                    technique_counts[technique] += 1
            total_responses = sum(technique_counts.values())
            if total_responses > 0:
                st.write("Response style preferences:")
                for technique, count in technique_counts.items():
                    percentage = (count / total_responses) * 100
                    st.write(f"- {technique.replace('_', ' ').title()}: {percentage:.1f}%")
            st.subheader("Feedback")
            feedback = st.text_area("Please share what you liked, what could be improved, or what's missing:", key="feedback")
            if st.button("Save & End Conversation"):
                technique_counts = {"conditional": 0, "dynamic_context": 0}
                for technique in st.session_state.preferred_technique.values():
                    if technique in technique_counts:
                        technique_counts[technique] += 1
                preferred_technique = max(technique_counts, key=technique_counts.get) if technique_counts else None
                # Save conversation and explicitly request email sending
                result = save_conversation(
                    st.session_state.conversation_id,
                    st.session_state.user_id,
                    st.session_state.conversation_history,
                    preferred_technique,
                    feedback,
                    send_email_report=True  # Explicitly request email sending
                )
                
                if isinstance(result, tuple):
                    filename, email_sent = result
                else:
                    filename, email_sent = result, False
                
                if filename:
                    st.success(f"Conversation saved to {filename}")
                    
                    # Create summary for display
                    total_votes = sum(technique_counts.values())
                    conditional_percent = (technique_counts["conditional"] / total_votes * 100) if total_votes > 0 else 0
                    dynamic_percent = (technique_counts["dynamic_context"] / total_votes * 100) if total_votes > 0 else 0
                    
                    summary_text = f"""
                    **Conversation Summary:**
                    - Total responses rated: {total_votes}
                    - Conditional responses preferred: {technique_counts['conditional']} ({conditional_percent:.1f}%)
                    - Dynamic context responses preferred: {technique_counts['dynamic_context']} ({dynamic_percent:.1f}%)
                    - Overall preferred technique: {preferred_technique.replace('_', ' ').title() if preferred_technique else 'None'}
                    """
                    
                    st.info(summary_text)
                    
                    # Show email status
                    if os.getenv("ADMIN_EMAIL"):
                        if email_sent:
                            st.success("Conversation data has been sent to the administrator")
                        else:
                            st.warning("Failed to send email to administrator. Data has been saved locally.")
                
                # Reset session state
                st.session_state.user_profile = None
                st.session_state.conversation_history = []
                st.session_state.conversation_id = str(uuid.uuid4())
                st.session_state.user_id = str(uuid.uuid4())
                st.session_state.preferred_technique = {}
                st.session_state.onboarding_complete = False
                st.session_state.current_message_processed = False
                st.session_state.last_interaction_time = time.time()
                st.session_state.input_key = 0
                st.rerun()
        st.subheader("Conversation")
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.conversation_history:
                role = message["role"]
                content = message["content"]
                timestamp = message.get("timestamp", datetime.now().isoformat())
                response_id = message.get("response_id", str(uuid.uuid4()))
                if role == "user":
                    st.markdown(f"**You:** {content}")
                elif role == "assistant":
                    if "responses" in message:
                        responses = message["responses"]
                        option_a_tech = message.get("option_a_tech")
                        option_b_tech = message.get("option_b_tech")
                        option_a = responses[option_a_tech]
                        option_b = responses[option_b_tech]
                        
                        # If user has already selected a preferred response
                        if response_id in st.session_state.preferred_technique:
                            preferred_technique = st.session_state.preferred_technique[response_id]
                            preferred_content = responses[preferred_technique]
                            st.markdown(f"**BestieAI:** {preferred_content}")
                        else:
                            st.markdown("**BestieAI:** (Please select your preferred response style)")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Option A:**")
                                st.markdown(f"{option_a}")
                                if st.button("👍 Prefer A", key=f"prefer_a_{response_id}"):
                                    st.session_state.preferred_technique[response_id] = option_a_tech
                                    message["content"] = option_a
                                    # Reset current_message_processed to allow new input
                                    st.session_state.current_message_processed = False
                                    st.rerun()
                            with col2:
                                st.markdown("**Option B:**")
                                st.markdown(f"{option_b}")
                                if st.button("👍 Prefer B", key=f"prefer_b_{response_id}"):
                                    st.session_state.preferred_technique[response_id] = option_b_tech
                                    message["content"] = option_b
                                    # Reset current_message_processed to allow new input
                                    st.session_state.current_message_processed = False
                                    st.rerun()
                    else:
                        st.markdown(f"**BestieAI:** {content}")
        st.markdown("---")
        user_input = st.text_input("Type your message:", key=f"input_{st.session_state.input_key}")
        if user_input and not st.session_state.current_message_processed:
            st.session_state.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            st.session_state.last_interaction_time = time.time()
            conversation_context = get_conversation_context()
            response_id = str(uuid.uuid4())
            with st.spinner("Thinking..."):
                conditional_prompt = generate_conditional_response_prompt(st.session_state.user_profile, conversation_context)
                conditional_response = get_completion(conditional_prompt, user_input)
                dynamic_context_prompt = generate_dynamic_context_prompt(st.session_state.user_profile, conversation_context)
                dynamic_context_response = get_completion(dynamic_context_prompt, user_input)
                # Shuffle techniques for options
                techniques = ["conditional", "dynamic_context"]
                random.shuffle(techniques)
                option_a_tech, option_b_tech = techniques
                option_a = conditional_response if option_a_tech == "conditional" else dynamic_context_response
                option_b = conditional_response if option_b_tech == "conditional" else dynamic_context_response
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": "",
                "response_id": response_id,
                "responses": {
                    "conditional": conditional_response,
                    "dynamic_context": dynamic_context_response
                },
                "option_a_tech": option_a_tech,
                "option_b_tech": option_b_tech,
                "timestamp": datetime.now().isoformat()
            })
            st.session_state.current_message_processed = True
            st.session_state.input_key += 1
            st.rerun()
        current_time = time.time()
        if current_time - st.session_state.last_interaction_time > 60:
            if "last_engagement_time" not in st.session_state or current_time - st.session_state.last_engagement_time > 100:
                engagement_messages = [
                    "Hey there! Still with me? I'd love to chat more about what's on your mind.",
                    f"Hi {st.session_state.user_profile['preferred_name']}! Anything else you'd like to talk about today?",
                    "I'm here if you want to continue our conversation. What else is on your mind?",
                    f"Just checking in! Is there anything else you'd like to discuss, {st.session_state.user_profile['preferred_name']}?",
                    f"Taking a break? I'm here whenever you're ready to chat again!"
                ]
                engagement_message = random.choice(engagement_messages)
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": engagement_message,
                    "timestamp": datetime.now().isoformat()
                })
                st.session_state.last_engagement_time = current_time
                st.session_state.last_interaction_time = current_time
                st.rerun()

if __name__ == "__main__":
    main()