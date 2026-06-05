import os
import streamlit as st
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
try:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Updated to a valid HuggingFace model repo
REPO_ID = "Qwen/Qwen2.5-7B-Instruct" 
DB_FAISS_PATH = "vectorstore/db_faiss"

class SimpleHFChatModel(BaseChatModel):
    repo_id: str
    hf_token: str
    temperature: float = 0.5
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
    """Defines the strict instructions for the LLM to prevent hallucinations."""
    # Fixed the variable from {question} to {input} to match create_retrieval_chain defaults
    custom_prompt_template = """Use the pieces of information provided in the context to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Don't provide anything out of the given context.
    
    Context: {context}
    Question: {input}
    
    Start the answer directly. No small talk please.
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
        temperature=0.5,
        max_new_tokens=512
    )
    return chat_model

def main():
    st.title("MediBot: AI Medical Chatbot")

    # Access HF_TOKEN from environment, or let user input it in sidebar
    token_to_use = HF_TOKEN
    if not token_to_use:
        st.sidebar.subheader("Hugging Face API Settings")
        token_to_use = st.sidebar.text_input("Enter HF Token (required if not in .env):", type="password")

    # Initialize chat history in Streamlit session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # Chat Input
    prompt = st.chat_input("Ask a medical question based on the textbook...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            # Setup RAG Pipeline
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error(f"Failed to load the vector store. Ensure '{DB_FAISS_PATH}' exists and run the creation script first.")
                return

            llm = load_llm(token_to_use)
            qa_prompt = set_custom_prompt()

            # Create the QA Chain
            combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
            qa_chain = create_retrieval_chain(
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}), 
                combine_docs_chain=combine_docs_chain
            )
            
            with st.spinner("Analyzing medical texts..."):
                response = qa_chain.invoke({"input": prompt})
                
            result = response["answer"]
            source_docs = response["context"]

            # Format the output to show source metadata
            result_to_show = f"{result}\n\n---\n**Source Documents:**\n"
            for i, doc in enumerate(source_docs, 1):
                page_num = doc.metadata.get('page', 'Unknown Page')
                # Optional: adding the source file name if available
                source_file = doc.metadata.get('source', 'Unknown Source')
                file_name = os.path.basename(source_file)
                result_to_show += f"* Source {i}: {file_name} (Page {page_num})\n"

            # Display response
            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()