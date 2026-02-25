# FAISS = similarity search engine
import faiss
import numpy as np
import os
import pickle

def create_faiss_index(embeddings):
    """
    Create faiss index from embeddings
    """
    dimension = embeddings.shapep[1]

    # IndexFlatL2 = simple L2 distance index
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index

def save_index(index, path="vectorstore/faiss.index"):
    """
    Saves faiss index to disk
    """
    faiss.write_index(index, path)

def load_index(path="vectorstore/faiss.index"):
    """
    Loads faiss index from disk
    """
    return faiss.read_index(path)