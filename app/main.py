from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from together import Together
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import asyncio
import time

# Load environment variables
load_dotenv()

# Load API keys
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize FastAPI
app = FastAPI()

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Together AI client
client = Together(api_key=TOGETHER_API_KEY)

# Global cache for product data
product_data_cache = {"data": None, "last_updated": 0}
CACHE_EXPIRY = 300  # 5 minutes

# Load knowledge base at startup
def load_knowledge_base():
    try:
        knowledge_file = os.path.join(os.path.dirname(__file__), "knowledge_base.txt")
        with open(knowledge_file, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return ""

# Preload knowledge base and FAISS vector store at startup
knowledge_base = load_knowledge_base()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function=len)
documents = text_splitter.split_text(knowledge_base)

# Load embeddings and FAISS once
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
vector_store = FAISS.from_texts(documents, embeddings)

# Define request model
class ChatRequest(BaseModel):
    message: str

async def fetch_product_data():
    """
    Fetch product data from Supabase, caching it every 5 minutes.
    Runs in the background to keep cache fresh without blocking responses.
    """
    try:
        current_time = time.time()
        if product_data_cache["data"] and (current_time - product_data_cache["last_updated"] < CACHE_EXPIRY):
            return product_data_cache["data"]

        # Fetch products and variations in a single batch call
        response = supabase.rpc("fetch_all_products_with_variations").execute()
        products = response.data if response.data else []

        product_texts = [
            f"Product: {p['name']}\nDescription: {p['description']}\nPrice: RM{p['priceincents'] / 100}\nAvailability: {'Available' if p['isavailableforpurchase'] else 'Out of stock'}\nVariants: {p['variants']}"
            for p in products
        ]

        result_text = "\n\n".join(product_texts) if product_texts else "No product data available."
        product_data_cache["data"] = result_text
        product_data_cache["last_updated"] = current_time
        print("🔹 Product data updated in cache.")
        return result_text

    except Exception as e:
        print(f"⚠️ Error fetching product data: {e}")
        return "No product data available."

@app.on_event("startup")
async def preload_data():
    """
    Fetch product data at startup to reduce initial delay for users.
    """
    await fetch_product_data()

@app.post("/chat/")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        user_query = request.message
        print(f"🔹 Received message: {user_query}")

        # Fetch product data & perform similarity search in parallel
        product_data_task = asyncio.create_task(fetch_product_data())
        faiss_results = vector_store.similarity_search(query=user_query, k=2)

        retrieved_docs = [doc.page_content for doc in faiss_results]
        context = " ".join(retrieved_docs) if retrieved_docs else ""

        # Wait for product data retrieval
        product_data = await product_data_task

        # Combine product data with knowledge base
        full_context = f"{product_data}\n\n{context}" if product_data else context
        print(f"Final context used for response:\n{full_context}")

        system_prompt = (
            "You are an AI customer support assistant for Rusholme Rendunks E-Commerce. "
            "Answer customer inquiries clearly and concisely without mentioning 'context' or 'provided information'. "
            "If you don't know the answer, politely suggest contacting support without making up facts."
        )

        # Generate response using Together API
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            messages=[
                {"role": "system", "content": system_prompt},
                # Instead of "Context: {full_context}", add a natural assistant role entry
                {"role": "assistant", "content": full_context},  
                {"role": "user", "content": user_query}
            ],
            max_tokens=800
        )

        reply_text = response.choices[0].message.content
        print(f"🗨️ Reply: {reply_text}")

        # Schedule background cache update
        background_tasks.add_task(fetch_product_data)

        return {"reply": reply_text}

    except Exception as e:
        print(f"⚠️ Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
