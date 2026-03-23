import os
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic


#initialization
encoder_model = SentenceTransformer("all-MiniLM-L6-v2")
chromadb_client = chromadb.PersistentClient(path="./chroma_db")
load_dotenv()
model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_tokens=3000
)

# embed queries
def generate_embeddings(user_message):
    embeddings = encoder_model.encode(user_message)
    return embeddings.tolist()

#find top-k-chunk
def find_top_k_chunk(k,user_message):
    collection = chromadb_client.get_collection(name="my-collection")
    results = collection.query(
        query_embeddings=[generate_embeddings(user_message)],
        n_results= k
        )
    return results["documents"][0]

def query_rag(user_message,k=5):
    chunks = find_top_k_chunk(k,user_message)
    context="\n\n".join(chunks)

    messages=[
        (
            "system",
            "You are a helpful assistant that answers user questions when given relevant context. If you can't find relevant information, reply you don't know"
        ),
        (
            "human",
            f"""here are relevant documents:
            {context}
            and here is user question:
            {user_message}"""
        )
    ]
    return model.invoke(messages).content