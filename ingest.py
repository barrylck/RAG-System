from pypdf import PdfReader
import re
from sentence_transformers import SentenceTransformer
import chromadb 

# read pdf document and extract text (filepath is the path to the pdf document)
def read_pdf(filepath):
    reader = PdfReader(filepath)
    full_text=""
    for page in reader.pages:
        full_text+=page.extract_text()
    return full_text

# split text into paragraphs

def split_by_paragraphs(text):
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs

# split text into chunks by words, with a specified chunk size and overlap

def split_by_tokens(text,chunk_size=250,overlap=50):
    words = text.split()
    chunks = []
    start=0
    while start<len(words):
        end = start+chunk_size
        chunks.append(" ".join(words[start:end]))
        start+=chunk_size-overlap
    return chunks

# split document into chunks, first by paragraphs, then by tokens if the paragraph is too long

def split_document(text, chunk_size = 250, overlap = 50):
    paragraphs = split_by_paragraphs(text)
    final_chunks = []
    for paragraph in paragraphs:
        if len(paragraph.split())<=chunk_size:
            final_chunks.append(paragraph)
        else:
            sub_chunk = split_by_tokens(paragraph)
            final_chunks.extend(sub_chunk)
    return final_chunks

# generate embeddings for the chunks using sentence transformer model

def generate_embeddings(input):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(input)
    return embeddings


# calling the function to read pdf document

text = read_pdf(filepath)

# calling the function to split document into chunks

chunks = split_document(text)
print(f"the number of chunks in this document is {len(chunks)}")

# calling the function to generate embeddings

embeddings = generate_embeddings(chunks)
print(f"the shape of the embdding is {embeddings.shape}")


# calling the functions to storing chunks and embeddings in vector database

def store_embeddings(chunks, embeddings, doc_name):
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="document")
    collection.add(
        ids = [f"{doc_name}_{i}" for i in range(len(chunks))],
        documents  = chunks,
        embeddings = embeddings.tolist()
    )
    print(f"stored {len(chunks)} chunks in vector database")
    return collection



