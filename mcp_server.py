#!/usr/bin/env python3
"""
Offline LangChain Agent with ThinkOS search tool using Ollama (LLaMA 3)
"""

import os
import logging
import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.llms import Ollama

# ------------------------
# Logging
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ThinkOS-Agent")

# ------------------------
# Globals
# ------------------------
index = None
chunks = None
model = None

# ------------------------
# Load resources
# ------------------------
def load_resources():
    global index, chunks, model

    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "data", "index.faiss")
    chunks_path = os.path.join(base_dir, "data", "chunks.pkl")

    if not os.path.exists(index_path):
        logger.error(f"Missing FAISS index: {index_path}")
        return False

    if not os.path.exists(chunks_path):
        logger.error(f"Missing chunks file: {chunks_path}")
        return False

    try:
        logger.info("Loading FAISS index...")
        index = faiss.read_index(index_path)

        logger.info("Loading chunks...")
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)

        logger.info("Loading cached embedding model...")
        model = SentenceTransformer("all-mpnet-base-v2")

        logger.info(f"Resources loaded: {index.ntotal} vectors, {len(chunks)} chunks.")
        return True
    except Exception as e:
        logger.exception("Failed to load resources")
        return False

# ------------------------
# ThinkOS search tool
# ------------------------
def search_thinkos(query: str, k: int = 5) -> str:
    if index is None or chunks is None or model is None:
        if not load_resources():
            return "Failed to load ThinkOS resources."

    try:
        embedding = model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = index.search(embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(chunks):
                results.append(f"Result {i+1} (score={distances[0][i]:.2f}):\n{chunks[idx]}")

        return "\n\n".join(results) if results else "No relevant content found."
    except Exception as e:
        logger.exception("Search failed")
        return f"Search failed: {str(e)}"

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    if not load_resources():
        exit(1)

    thinkos_tool = Tool(
        name="SearchThinkOS",
        func=search_thinkos,
        description="Useful for answering questions from the Think OS book"
    )

    llm = Ollama(model="mistral")  # or use "mistral", "phi", etc.

    agent = initialize_agent(
        tools=[thinkos_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    while True:
        query = input("\n🧠 Ask about ThinkOS (or type 'exit'): ")
        if query.strip().lower() in ["exit", "quit"]:
            break
        response = agent.run(query)
        print(f"\n🤖 {response}")
