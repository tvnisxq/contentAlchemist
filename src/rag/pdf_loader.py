import fitz # pymupdf libary

def load_pdf(file_path: str):
    """
    Loads a pdf file and extracts all textual context.
    Returns the full extracted text as a single string.
    """

    # Open the pdf file
    doc = fitz.open(file_path)

    full_text = ""

    # Iterate through all pages
    for page in doc:
        # Extract text from each page
        text = page.get_text()

        # Append to full text
        full_text += text

    # Close the doc to free memory
    doc.close()

    return full_text

