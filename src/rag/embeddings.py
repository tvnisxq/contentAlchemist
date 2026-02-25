from sentence_transformers import SentenceTransformer

# Load embedding model once(avoid reloading per request)
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts:list):
    """
    Converts list of text chunks into vector embeddings.
    Returns numpy array of embeddings.
    """
    # Generate embeddings
    embeddings = model.encode(texts)
    return embeddings

def embed_query(query:str):
    """
    Converts user qeury into embedding vector. 
    """
    return model.encode([query])