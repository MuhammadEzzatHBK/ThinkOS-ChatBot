import PyPDF2
import os
import re
from typing import List, Optional
from io import StringIO
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np


def read_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_parts = []
            for page in pdf_reader.pages:
                text_parts.append(page.extract_text())
            return ''.join(text_parts)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None

def preprocess_text(text):
    if not text:
        return None
    
    text = re.sub(r'\n\s*\d+\s*\n|\s+\d+\s*$|^.*?Chapter\s+\d+.*?\n|^.*?©.*?\n', 
                 '\n', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    return text

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    max_chunk_size = 1000
    overlap = 100
    
    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = current_chunk[-overlap:] if overlap > 0 else ""
        
        current_chunk += para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def create_embedding(chunks: List[str]):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return embeddings

def save_index_and_chunks(index, chunks, index_path="data/index.faiss", chunks_path="data/chunks.pkl"):
    faiss.write_index(index, index_path)
    
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"Saved index to {index_path} and chunks to {chunks_path}")

def load_index_and_chunks(index_path="data/index.faiss", chunks_path="data/chunks.pkl"):
    index = faiss.read_index(index_path)
    
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    
    return index, chunks

document = read_pdf("data/ThinkOS.pdf")
chunks = chunk_text(preprocess_text(document))
embeddings = create_embedding(chunks)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)

save_index_and_chunks(index, chunks)