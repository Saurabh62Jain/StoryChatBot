import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Import our RAG pipeline helper functions from the Streamlit app
from connect_memory_with_llm import get_vectorstore, set_custom_prompt, load_llm, StoryRetriever, is_welcome_gesture, get_welcome_response
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
INSTANCE_ID = os.getenv("GREEN_API_INSTANCE_ID")
API_TOKEN = os.getenv("GREEN_API_TOKEN")

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
print("RAG Pipeline Ready for Green-API WhatsApp messages!")

@app.route("/webhook", methods=["POST"])
def greenapi_webhook():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No JSON payload received"}), 400
    
    # We only care about incoming text messages
    type_webhook = data.get("typeWebhook")
    if type_webhook == "incomingMessageReceived":
        message_data = data.get("messageData", {})
        type_msg = message_data.get("typeMessage")
        
        # Ensure it is a text message
        if type_msg == "textMessage":
            incoming_msg = message_data.get("textMessageData", {}).get("textMessage", "").strip()
            chat_id = data.get("senderData", {}).get("chatId") # Recipient chatId (e.g. 919179660938@c.us)
            
            print(f"Received message from {chat_id}: '{incoming_msg}'")
            
            if is_welcome_gesture(incoming_msg):
                response_text = get_welcome_response()
            else:
                # Query the RAG Pipeline
                try:
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
                    print(f"RAG Error: {e}")
                    response_text = "Sorry, I encountered an error processing your query."
                
            # Send message back via Green-API HTTP request
            send_url = f"https://api.green-api.com/waInstance{INSTANCE_ID}/sendMessage/{API_TOKEN}"
            payload = {
                "chatId": chat_id,
                "message": response_text
            }
            try:
                res = requests.post(send_url, json=payload)
                print(f"Reply sent status: {res.status_code}")
            except Exception as e:
                print(f"Failed to send WhatsApp reply: {e}")

    return jsonify({"status": "success"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
