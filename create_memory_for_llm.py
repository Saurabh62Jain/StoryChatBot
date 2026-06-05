from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pypdf import PdfReader
import os
import re

# Define Paths
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    # Ensure directories exist
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)

    print("Step 1: Loading raw PDF files and grouping by story...")
    pdf_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDFs found in '{DATA_PATH}'. Please add documents and try again.")
        return
        
    all_story_chunks = []
    story_pattern = re.compile(r"Story-([^\n]+?)\s+(\d+)/(\d+)")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for filepath in pdf_files:
        print(f"Processing PDF: {os.path.basename(filepath)}")
        reader = PdfReader(filepath)
        
        current_story_name = None
        current_story_pages = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            match = story_pattern.search(text)
            
            if match:
                story_name = match.group(1).strip()
                page_idx = match.group(2).strip()
                if story_name != current_story_name or page_idx == '1':
                    # Save the previous story
                    if current_story_name and current_story_pages:
                        whole_story_text = "\n".join(current_story_pages)
                        story_doc = Document(page_content=whole_story_text, metadata={"source": filepath, "story": current_story_name})
                        chunks = text_splitter.split_documents([story_doc])
                        for chunk in chunks:
                            chunk.page_content = f"Story: {current_story_name}. {chunk.page_content}"
                            chunk.metadata["story_content"] = whole_story_text
                            all_story_chunks.append(chunk)
                    
                    current_story_name = story_name
                    current_story_pages = []
            
            if current_story_name:
                current_story_pages.append(text)
                
        # Save the last story in the file
        if current_story_name and current_story_pages:
            whole_story_text = "\n".join(current_story_pages)
            story_doc = Document(page_content=whole_story_text, metadata={"source": filepath, "story": current_story_name})
            chunks = text_splitter.split_documents([story_doc])
            for chunk in chunks:
                chunk.page_content = f"Story: {current_story_name}. {chunk.page_content}"
                chunk.metadata["story_content"] = whole_story_text
                all_story_chunks.append(chunk)

    print(f"Step 2: Created {len(all_story_chunks)} chunks from grouped stories.")

    print("Step 3: Creating vector embeddings and storing in FAISS...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Store chunks in Vector Database
    db = FAISS.from_documents(all_story_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector database saved to {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()