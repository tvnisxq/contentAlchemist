# LLMS Cannot handle large PDFs directly
def chunk_text(text:str, chunk_size: int=1000, overlap: int=200):
    """
    Splits text into overlapping chunks.

    chunk_size: size of each chunk(character)
    overlap: overlapping characters between chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        # Define end of chunk
        end = start + chunk_size

        # Extract chunk
        chunk = text[start:end]

        chunks.append(chunk)

        # Move start forward but keep overlap
        start = end - overlap
    
    return chunks    