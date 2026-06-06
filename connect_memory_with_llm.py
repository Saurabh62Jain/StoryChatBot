import os
import json
import streamlit as st
import streamlit.components.v1 as components
import requests
from typing import Any, List, Optional
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
try:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
from pathlib import Path
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
HF_TOKEN = os.getenv("HF_TOKEN")

# Updated to a valid HuggingFace model repo
REPO_ID = "Qwen/Qwen2.5-7B-Instruct" 
DB_FAISS_PATH = "vectorstore/db_faiss"

def is_welcome_gesture(text: str) -> bool:
    cleaned = text.lower().strip("?!. ")
    welcome_words = {"hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening", "namaste", "wassup", "sup"}
    return cleaned in welcome_words

def get_welcome_response() -> str:
    return """Welcome! I am **Saurabh Jain's StoryBot**, your AI assistant for exploring the creative works, screenplays, and stories written by Saurabh Jain.

Here are the ways I can serve you today:
1. 📚 **Summarize a Story**: Ask me to summarize any story (e.g., *"Summarize 4th Idiot"* or *"What is Astitva about?"*).
2. 👥 **Character Analysis**: Ask about specific characters, their relationships, and their roles in the stories.
3. 💡 **Explore Themes & Philosophy**: Inquire about underlying themes, management concepts, or philosophical dilemmas explored in the writings (e.g., *"Explain Deadlock Handling in 4th Idiot"*).
4. 🔍 **Find Story References**: Ask where a specific story or concept is located in the source documents.
5. ❓ **Answer Specific Plot Questions**: Ask about how a story ends, specific plot twists, or character choices.

How can I serve you today? Please enter a story title or ask a question!"""

class StoryRetriever(BaseRetriever):
    vectorstore: object
    search_kwargs: dict

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Extract unique story names from the vector store dynamically
        story_names = set()
        for doc in self.vectorstore.docstore._dict.values():
            story = doc.metadata.get("story")
            if story:
                story_names.add(story)
        
        # Check if any story name is explicitly mentioned in the query
        q_lower = query.lower()
        matched_story = None
        for story in sorted(list(story_names), key=len, reverse=True):
            if story.lower() in q_lower:
                matched_story = story
                break
                
        # If a story is explicitly mentioned, return its complete text directly
        if matched_story:
            story_content = None
            source = None
            for doc in self.vectorstore.docstore._dict.values():
                if doc.metadata.get("story") == matched_story:
                    story_content = doc.metadata.get("story_content")
                    source = doc.metadata.get("source")
                    break
            if story_content:
                return [Document(page_content=story_content, metadata={"story": matched_story, "source": source})]

        # Otherwise, fall back to semantic vector similarity search
        docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        expanded_docs = []
        for doc in docs:
            story_content = doc.metadata.get("story_content")
            if story_content:
                expanded_docs.append(Document(page_content=story_content, metadata=doc.metadata))
            else:
                expanded_docs.append(doc)
        return expanded_docs

class SimpleHFChatModel(BaseChatModel):
    repo_id: str
    hf_token: str
    temperature: float = 1.0
    max_new_tokens: int = 512

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        formatted_messages = []
        for msg in messages:
            role = "user"
            if msg.type == "human" or msg.type == "user":
                role = "user"
            elif msg.type == "ai" or msg.type == "assistant":
                role = "assistant"
            elif msg.type == "system":
                role = "system"
            formatted_messages.append({"role": role, "content": msg.content})

        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "model": self.repo_id,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens
        }
        url = "https://router.huggingface.co/v1/chat/completions"
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise ValueError(f"Hugging Face Router API Error ({response.status_code}): {response.text}")
        res_data = response.json()
        content = res_data["choices"][0]["message"]["content"]
        ai_msg = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    @property
    def _llm_type(self) -> str:
        return "simple_hf_chat"

@st.cache_resource
def get_vectorstore():
    """Loads the FAISS database locally and caches it for Streamlit."""
    if not os.path.exists(DB_FAISS_PATH):
        return None
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt():
    """Defines the instructions for the LLM specializing in Saurabh Jain's stories."""
    custom_prompt_template = """You are an AI assistant specialized in the stories and creative works written by Saurabh Jain.
    Use the pieces of story text provided in the context to answer the user's question.
    If the answer cannot be found in the provided context, just say that you don't know, do not try to make up, extrapolate, or invent details.
    Keep the tone engaging, helpful, and focused on detailing the characters, events, and plots from the stories.
    
    Context: {context}
    Question: {input}
    
    Start the answer directly.
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "input"])
    return prompt

def load_llm(hf_token):
    """Connects to the LLM via Hugging Face API."""
    if not hf_token:
        raise ValueError("HF_TOKEN not found. Please set it in your .env file or in the sidebar.")
        
    chat_model = SimpleHFChatModel(
        repo_id=REPO_ID,
        hf_token=hf_token,
        temperature=1.0,
        max_new_tokens=512
    )
    return chat_model

def render_assistant_response(content: str):
    st.markdown(content)
    
    # Clean up formatting to keep the speech voice clear (e.g. remove markdown indicators)
    clean_content = (
        content
        .replace("**", "")
        .replace("*", "")
        .replace("`", "")
        .replace("###", "")
        .replace("##", "")
        .replace("#", "")
        .strip()
    )
    
    # Securely strip references/sources block from the email text to prevent security breach
    import re
    clean_response = re.sub(r"---[\s\S]*$", "", clean_content).strip()
    
    # Create a clean, secure email body template
    email_body = (
        "Hello,\n\n"
        "I am reporting an incorrect response or error in the StoryBot:\n\n"
        "--- CHATBOT RESPONSE ---\n"
        f"{clean_response}\n"
        "------------------------\n\n"
        "Please describe the error or correct response here:\n\n"
    )
    
    # URL-encode the clean email body for the Gmail compose link
    import urllib.parse
    encoded_body = urllib.parse.quote(email_body)
    encoded_subject = urllib.parse.quote("StoryBot Incorrect Response Report")
    gmail_url = f"https://mail.google.com/mail/?view=cm&fs=1&to=saurabh62jain@gmail.com&su={encoded_subject}&body={encoded_body}"
    
    js_text = json.dumps(clean_content)
    
    html_code = f"""
    <!DOCTYPE html>
    <html>    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&display=swap');
    body {{
        margin: 0;
        padding: 0;
        background-color: transparent;
        overflow: hidden;
    }}
    .container {{
        display: flex;
        gap: 10px;
        align-items: center;
    }}
    .speaker-btn {{
        background-color: rgba(30, 41, 59, 0.6);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 6px 14px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 0.82rem;
        font-weight: 600;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        backdrop-filter: blur(4px);
    }}
    .speaker-btn:hover {{
        background-color: rgba(56, 189, 248, 0.15);
        border-color: #38bdf8;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.35);
        transform: translateY(-1px);
    }}
    .report-btn {{
        background-color: rgba(30, 41, 59, 0.6);
        color: #f43f5e;
        border: 1px solid rgba(244, 63, 94, 0.3);
        padding: 6px 14px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 0.82rem;
        font-weight: 600;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        display: inline-flex;
        align-items: center;
        gap: 6px;
        text-decoration: none;
        font-family: 'Plus Jakarta Sans', sans-serif;
        backdrop-filter: blur(4px);
    }}
    .report-btn:hover {{
        background-color: rgba(244, 63, 94, 0.15);
        border-color: #f43f5e;
        box-shadow: 0 0 12px rgba(244, 63, 94, 0.35);
        transform: translateY(-1px);
    }}
    .note {{
        font-size: 0.8rem;
        color: #cbd5e1;
        margin-top: 8px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        line-height: 1.4;
        letter-spacing: 0.01em;
    }}
    .note-link {{
        color: #38bdf8;
        text-decoration: underline;
        font-weight: 600;
    }}
    .note-link:hover {{
                color: #93c5fd;
            }}
            </style>
    </head>
    <body>
    <div class="container">
        <button id="speak-btn" class="speaker-btn">
            🔊 Listen Response
        </button>
        <a id="report-btn" class="report-btn" href="{gmail_url}" target="_blank">
            ✉ Report Response
        </a>
    </div>
    <div class="note">
        <strong>Note:</strong> In case of any error and wrong response please click 'Report Response' or mail the response of chatbot to <a href="{gmail_url}" class="note-link" target="_blank">saurabh62jain@gmail.com</a>.
    </div>
    <script>
    const text = {js_text};
    const btn = document.getElementById('speak-btn');
    let utterance = null;
    
    btn.addEventListener('click', () => {{
        if (window.speechSynthesis.speaking) {{
            window.speechSynthesis.cancel();
            btn.innerHTML = '🔊 Listen Response';
        }} else {{
            // Clean up any remaining reference block tags or emojis for smoother reading
            const cleanText = text.replace(/---[\\s\\S]*$/g, '').trim();
            utterance = new SpeechSynthesisUtterance(cleanText);
            utterance.onend = () => {{
                btn.innerHTML = '🔊 Listen Response';
            }};
            utterance.onerror = () => {{
                btn.innerHTML = '🔊 Listen Response';
            }};
            btn.innerHTML = '⏹ Stop Listening';
            window.speechSynthesis.speak(utterance);
        }}
    }});
    </script>
    </body>
    </html>
    """
    components.html(html_code, height=90)

def main():
    st.set_page_config(page_title="Saurabh Jain's StoryBot", page_icon="📖", layout="wide")

    # Sidebar Background Theme Selector
    st.sidebar.subheader("App Theme Settings")
    bg_choice = st.sidebar.selectbox(
        "Choose Background Theme:",
        [
            "🌌 Cosmic Library", 
            "✍️ Vintage Writer", 
            "🌆 Cyberpunk Narrative", 
            "🖤 Minimal Dark", 
            "📤 Upload Custom Image", 
            "🔗 Custom Image URL", 
            "🎨 Classic Moving Gradient", 
            "🌚 Solid Dark"
        ]
    )

    bg_filename = None
    if bg_choice == "🌌 Cosmic Library":
        bg_filename = "cosmic_library.png"
    elif bg_choice == "✍️ Vintage Writer":
        bg_filename = "vintage_writer.png"
    elif bg_choice == "🌆 Cyberpunk Narrative":
        bg_filename = "cyberpunk_narrative.png"
    elif bg_choice == "🖤 Minimal Dark":
        bg_filename = "minimal_dark.png"

    import base64
    img_base64 = ""
    bg_css = ""

    if bg_filename:
        # Check in backgrounds/ folder
        bg_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backgrounds", bg_filename)
        if os.path.exists(bg_image_path):
            try:
                with open(bg_image_path, "rb") as image_file:
                    img_base64 = base64.b64encode(image_file.read()).decode()
                bg_css = f"""
                background-image: linear-gradient(rgba(10, 11, 28, 0.45), rgba(10, 11, 28, 0.45)), url('data:image/png;base64,{img_base64}') !important;
                background-size: cover !important;
                background-position: center !important;
                background-attachment: fixed !important;
                background-repeat: no-repeat !important;
                """
            except Exception:
                pass

    elif bg_choice == "📤 Upload Custom Image":
        uploaded_file = st.sidebar.file_uploader(
            "Upload background image (PNG, JPG, JPEG, WEBP):", 
            type=["png", "jpg", "jpeg", "webp"]
        )
        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.read()
                img_base64 = base64.b64encode(file_bytes).decode()
                mime_type = uploaded_file.type
                bg_css = f"""
                background-image: linear-gradient(rgba(10, 11, 28, 0.45), rgba(10, 11, 28, 0.45)), url('data:{mime_type};base64,{img_base64}') !important;
                background-size: cover !important;
                background-position: center !important;
                background-attachment: fixed !important;
                background-repeat: no-repeat !important;
                """
            except Exception as e:
                st.sidebar.error(f"Error loading image: {e}")
        else:
            st.sidebar.info("Please upload an image file.")

    elif bg_choice == "🔗 Custom Image URL":
        custom_url = st.sidebar.text_input(
            "Enter Image URL:",
            placeholder="https://example.com/background.jpg"
        )
        if custom_url.strip():
            bg_css = f"""
            background-image: linear-gradient(rgba(10, 11, 28, 0.45), rgba(10, 11, 28, 0.45)), url('{custom_url.strip()}') !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
            background-repeat: no-repeat !important;
            """
        else:
            st.sidebar.info("Please enter a valid image URL.")

    # Fallback to Solid Dark or Animated Gradient if no image CSS was set
    if not bg_css:
        if bg_choice == "🌚 Solid Dark":
            bg_css = """
            background-color: #0b0f19 !important;
            background-image: none !important;
            """
        else:
            # Default or fallback to moving gradient
            bg_css = """
            background: linear-gradient(-45deg, #020617, #0b0f19, #1e1b4b, #0f172a, #020617) !important;
            background-size: 400% 400% !important;
            animation: gradientShift 25s ease infinite !important;
            background-attachment: fixed !important;
            """

    # Custom CSS for UI styling
    css_code = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .main {
        BG_CSS_PLACEHOLDER
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glassmorphic Sidebar styling - targeting main container and all nested wrappers */
    div[data-testid="stSidebar"], 
    [data-testid="stSidebar"] > div, 
    [data-testid="stSidebarUserContent"] {
        background: rgba(10, 11, 28, 0.8) !important;
        background-color: rgba(10, 11, 28, 0.8) !important;
        backdrop-filter: blur(20px) saturate(160%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    }
    
    /* Font overrides for text elements (excluding generic layout wrappers like div/span to protect icons) */
    h1, h2, h3, h4, h5, h6, p, li, .stMarkdown, .stMarkdown p, .stMarkdown li {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Main Title Styling with metallic gradient and glow */
    h1 {
        font-family: 'Outfit', sans-serif !important;
        color: #f8fafc !important;
        background: linear-gradient(135deg, #38bdf8 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 800 !important;
        text-shadow: 0 0 40px rgba(167, 139, 250, 0.25) !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 0.25rem !important;
    }
    
    /* Custom Styling for Streamlit Subheader */
    .stSubheader p {
        color: #94a3b8 !important;
        font-size: 1.1rem !important;
        font-weight: 400 !important;
        letter-spacing: 0.01em !important;
    }
    
    /* Chat message bubble styling */
    div[data-testid="stChatMessage"] {
        background-color: rgba(15, 23, 42, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 1.25rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(8px) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    div[data-testid="stChatMessage"]:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(167, 139, 250, 0.25) !important;
        box-shadow: 0 12px 40px rgba(167, 139, 250, 0.12) !important;
    }
    
    /* Differentiate assistant and user message accents */
    div[data-testid="stChatMessage"][data-testid$="assistant"] {
        border-left: 4px solid #a78bfa !important;
        background-color: rgba(88, 28, 135, 0.08) !important;
    }
    
    div[data-testid="stChatMessage"][data-testid$="user"] {
        border-left: 4px solid #38bdf8 !important;
        background-color: rgba(15, 23, 42, 0.3) !important;
    }
    
    /* Chat input box and textarea styling */
    div[data-testid="stChatInput"] {
        background-color: transparent !important;
    }
    
    div[data-testid="stChatInput"] textarea {
        background-color: rgba(15, 23, 42, 0.7) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(12px) !important;
        transition: all 0.3s ease !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    div[data-testid="stChatInput"] textarea:focus {
        border-color: #a78bfa !important;
        box-shadow: 0 0 15px rgba(167, 139, 250, 0.3) !important;
    }

    /* Make placeholder text highly readable */
    div[data-testid="stChatInput"] textarea::placeholder {
        color: rgba(255, 255, 255, 0.55) !important;
        opacity: 1 !important;
    }
    div[data-testid="stChatInput"] textarea::-webkit-input-placeholder {
        color: rgba(255, 255, 255, 0.55) !important;
    }
    div[data-testid="stChatInput"] textarea::-moz-placeholder {
        color: rgba(255, 255, 255, 0.55) !important;
        opacity: 1 !important;
    }
    
    /* Style the Chat Input send button */
    div[data-testid="stChatInput"] button {
        background-color: #a78bfa !important;
        color: #0f172a !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-testid="stChatInput"] button:hover {
        background-color: #c084fc !important;
        box-shadow: 0 0 12px rgba(192, 132, 252, 0.6) !important;
    }
    
    /* Spinner customization */
    .stSpinner > div {
        border-top-color: #a78bfa !important;
    }
    
    /* Sidebar text/about section formatting */
    div[data-testid="stSidebar"] h3 {
        color: #a78bfa !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stSidebar"] p {
        color: #cbd5e1 !important;
        font-size: 0.92rem !important;
        line-height: 1.5 !important;
    }
    

    /* Hide Streamlit default menus and buttons */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stAppDeployButton {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    

    /* Hide Streamlit default menus and buttons */
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    header { display: none !important; }
    header[data-testid='stHeader'] { display: none !important; }
    .stAppDeployButton { display: none !important; }
    .stDeployButton { display: none !important; }
    div[data-testid='stAppDeployButton'] { display: none !important; }
    </style>
    """
    css_code = css_code.replace("BG_CSS_PLACEHOLDER", bg_css)
    st.markdown(css_code, unsafe_allow_html=True)

    st.title("📖 Saurabh Jain's StoryBot")
    st.subheader("Your AI Guide through the Stories and Creative Works of Author Saurabh Jain")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Hugging Face API Token")
    if not HF_TOKEN:
        token_to_use = st.sidebar.text_input(
            "Enter HF Token:",
            type="password",
            placeholder="hf_...",
            help="Get a free token from https://huggingface.co/settings/tokens"
        )
    else:
        token_to_use = HF_TOKEN
        override_token = st.sidebar.text_input(
            "Override HF Token (Optional):",
            type="password",
            placeholder="Leave blank to use default",
            help="Override the system environment token if desired."
        )
        if override_token.strip():
            token_to_use = override_token.strip()

    st.sidebar.markdown("""
    ---
    ### About the Author
    **Saurabh Jain** is the creator of these compelling stories. 
    Use this chatbot to query character breakdowns, plot summaries, and key themes across his registered screenplays and literary works.
    """)

    # Initialize chat history in Streamlit session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_response(message["content"])
            else:
                st.markdown(message["content"])

    # Chat Input
    prompt = st.chat_input("Ask a question about Saurabh Jain's stories (e.g. 'What is the theme of 4th Idiot?', 'Summarize Astitva')...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Intercept welcome gestures
        if is_welcome_gesture(prompt):
            welcome_res = get_welcome_response()
            with st.chat_message("assistant"):
                render_assistant_response(welcome_res)
            st.session_state.messages.append({"role": "assistant", "content": welcome_res})
            return

        try:
            # Setup RAG Pipeline
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error(f"Failed to load the vector store. Ensure '{DB_FAISS_PATH}' exists and run the creation script first.")
                return

            llm = load_llm(token_to_use)
            qa_prompt = set_custom_prompt()

            # Create Custom Story Retriever
            retriever = StoryRetriever(vectorstore=vectorstore, search_kwargs={'k': 1})

            # Create the QA Chain
            combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
            qa_chain = create_retrieval_chain(
                retriever=retriever, 
                combine_docs_chain=combine_docs_chain
            )
            
            with st.spinner("Searching through Saurabh Jain's stories..."):
                response = qa_chain.invoke({"input": prompt})
                
            result = response["answer"]
            source_docs = response["context"]

            # Format the output to show story references safely without exposing server filenames
            result_to_show = f"{result}\n\n---\n**📚 Story Reference:**\n"
            if source_docs:
                story_name = source_docs[0].metadata.get('story', 'Unknown Story')
                result_to_show += f"* **Story**: {story_name}\n"
            else:
                result_to_show += "* No direct story references matched.\n"

            # Display response
            with st.chat_message("assistant"):
                render_assistant_response(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()





