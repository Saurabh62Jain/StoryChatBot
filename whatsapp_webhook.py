import os
from flask import Flask, request, Response
from dotenv import load_dotenv

# Import our RAG pipeline helper functions from the Streamlit app
from connect_memory_with_llm import get_vectorstore, set_custom_prompt, load_llm, StoryRetriever, is_welcome_gesture, get_welcome_response
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

app = Flask(__name__)

# Initialize the RAG components globally for efficiency
print("Loading vector database and LLM...")
vectorstore = get_vectorstore()
if vectorstore is None:
    print("WARNING: Vector database 'vectorstore/db_faiss' not found!")
llm = load_llm(HF_TOKEN)
qa_prompt = set_custom_prompt()
combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
retriever = StoryRetriever(vectorstore=vectorstore, search_kwargs={'k': 1})
qa_chain = create_retrieval_chain(
    retriever=retriever, 
    combine_docs_chain=combine_docs_chain
)
print("Initialization complete. Ready for WhatsApp messages!")

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    # 1. Get the message from WhatsApp
    incoming_msg = request.values.get("Body", "").strip()
    sender_number = request.values.get("From", "")
    
    print(f"Received message from {sender_number}: '{incoming_msg}'")
    
    if not incoming_msg:
        response_text = "Hi! I didn't receive any text. How can I help you today?"
    elif is_welcome_gesture(incoming_msg):
        response_text = get_welcome_response()
    else:
        try:
            # 2. Query the RAG Pipeline
            response = qa_chain.invoke({"input": incoming_msg})
            result = response["answer"]
            source_docs = response["context"]
            
            # Format the response with story references
            if source_docs:
                story_name = source_docs[0].metadata.get('story', 'Unknown Story')
                response_text = f"{result}\n\n*Story Reference:* {story_name}"
            else:
                response_text = result
                
        except Exception as e:
            print(f"Error processing query: {e}")
            response_text = f"An error occurred while analyzing the stories: {str(e)}"

    # 3. Create the TwiML XML response for Twilio
    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{response_text}</Message>
</Response>"""

    return Response(twiml_response, mimetype="text/xml")

if __name__ == "__main__":
    # Run Flask app on port 5000
    app.run(host="0.0.0.0", port=5000)
