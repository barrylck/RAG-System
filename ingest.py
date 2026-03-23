import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb 
import re

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


# split document into chunks, first by paragraphs, then by tokens if the paragraph is too long
def split_document(text, chunk_size = 256, overlap = 26):

    def split_by_paragraphs(text):
        paragraphs = re.split(r'\n\s*\n',text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap, 
        length_function=lambda text: len(text.split())
    )

    paragraphs = split_by_paragraphs(text)
    final_chunks = []

    for paragraph in paragraphs:
        if len(paragraph.split())<=chunk_size:
            final_chunks.append(paragraph)
        else:
            sub_chunk = text_splitter.split_text(paragraph)
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




