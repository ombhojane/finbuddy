import re

def remove_unwanted_symbols(text):
    """
    Remove unwanted symbols like bullets and special characters from the text.
    """
    # Remove bullet points (•) and other common unwanted symbols
    text = re.sub(r'[•]', '', text)  # Remove bullet points
    # Add other unwanted symbols if necessary, for example:
    # text = re.sub(r'[^\w\s]', '', text)  # Remove all punctuation except for words and whitespace
    return text

def clean_text(text):
    """
    Clean the text by removing unwanted symbols and unnecessary spaces.
    """
    text = remove_unwanted_symbols(text)
    text = text.strip()
    return text

def preserve_content_length(original_text, cleaned_text, retention_factor=0.8):
    """
    Preserve the length of the original text by keeping a proportion of the cleaned text.
    """
    original_length = len(original_text.split())
    target_length = int(original_length * retention_factor)
    
    # Generate a word list from the cleaned text
    words = cleaned_text.split()
    
    # Keep the text length within the target range
    if len(words) > target_length:
        words = words[:target_length]  # Truncate if necessary
    
    return ' '.join(words)

def filter_unwanted_text(text):
    """
    Filter out unwanted parts of the text such as department names, titles, acronyms, and symbols.
    """
    # Define patterns to remove
    patterns = [
        r"\bDEPARTMENT OF.*\b",  # Department names
        r"\bProf\. [A-Za-z ]*\b",  # Titles and names
        r"\b[A-Z]+\b",  # Acronyms
        r"[-|]",  # Symbols
        r"\b[A-Z]+\b"  # Extra uppercase words (e.g., AI&ML)
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    
    # Remove multiple spaces and trim text
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
