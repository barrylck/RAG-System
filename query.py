import os
from sentence_transformers import SentenceTransformer
import chromadb
import anthropic 
from dotenv import load_dotenv

#initialization
encoder_model = SentenceTransformer("all-MiniLM-L6-v2")
chromadb_client = chromadb.PersistentClient(path="./chroma_db")
load_dotenv()
claude_client = anthropic.Anthropic() 

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

def construct_prompt(chunks, user_message):
    context="\n\n".join(chunks)
    prompt = f""" here are relevant documents:

    {context}

    based only on the exercepts above, answer the following quesiton:

    {user_message}

    If the answer cannot be found in the excerpts, say "I could not find this information in the provided documents." """
    return prompt

def query_rag(user_message,k=5):
    chunks = find_top_k_chunk(k,user_message)
    prompt = construct_prompt(chunks, user_message)
    message = claude_client.messages.create(
        model = "claude-haiku-4-5-20251001",
        max_tokens = 1000,
        messages=[
            {
            "role":"user",
            "content": prompt
            }
        ]
    )
    return message.content[0].text

print(query_rag("calculate apple's return on equity in recent 3 years", k=10))

