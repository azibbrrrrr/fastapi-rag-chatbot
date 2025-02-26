from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_together import Together
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv()

# Load API keys
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY is not set in the environment variables.")

# Initialize FastAPI
app = FastAPI()

# Initialize Together AI LLM
llm = Together(model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", together_api_key=TOGETHER_API_KEY, max_tokens=1000)

# Load knowledge base from text file
knowledge_file = os.path.join(os.path.dirname(__file__), "ecommerce_knowledge_base.txt")

try:
    # Load the text file
    loader = TextLoader(knowledge_file, encoding="utf-8")
    raw_documents = loader.load()
    print("Knowledge base loaded successfully!")
except Exception as e:
    print(f"Error loading knowledge base: {e}")

# Split text into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function=len)
documents = text_splitter.split_documents(raw_documents)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

# Create FAISS vector store from document
vector_store = FAISS.from_documents(documents, embeddings)

# Define request model
class ChatRequest(BaseModel):
    message: str

@app.post("/chat/")
async def chat(request: ChatRequest):
    try:
        user_query = request.message
        print(f"Received message: {user_query}")

        # Perform similarity search in FAISS
        faiss_results = vector_store.similarity_search(query=user_query, k=2)

        # Extract relevant content
        retrieved_docs = [doc.page_content for doc in faiss_results]
        context = " ".join(retrieved_docs) if retrieved_docs else "No relevant context found."
        print(f"Generated context: {context}")

        # Generate response using Together AI
        response = llm.predict(f"Use only the given context to answer briefly.\nContext: {context}\nUser: {user_query}\nAssistant:")

        print("reply:", response)

        return {"reply": response}

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))