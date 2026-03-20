from pypdf import PdfReader
import re
from sentence_transformers import SentenceTransformer
import chromadb 

#initialization
encoder_model = SentenceTransformer("all-MiniLM-L6-v2")
chromadb_client = chromadb.PersistentClient(path="./chroma_db")

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
    embeddings = encoder_model.encode(input)
    return embeddings

# store embeddings in chromadb
def store_in_chromadb(chunks, embeddings, doc_name, collection):
    collection.add(
        ids = [f"{doc_name}_{i}" for i in range(len(chunks))],
        documents  = chunks,
        embeddings = embeddings.tolist()
    )

#specify filepaths for pdfs
pdf_paths = [
     "./data/AnnualReport2023.pdf",
     "./data/AnnualReport2024.pdf",
     "./data/AnnualReport2025.pdf",
]

#create collection
collection = chromadb_client.get_or_create_collection(name="my-collection")

for pdf_path in pdf_paths:
    doc_name = pdf_path.split("/")[-1].replace(".pdf","")
    text = read_pdf(pdf_path)
    chunks= split_document(text)
    embeddings = generate_embeddings(chunks)
    store_in_chromadb(chunks, embeddings, doc_name, collection)
    print(f"for {doc_name},{len(chunks)} of chunks are stored")

print(f"total number of chunk loaded is {collection.count()}")



