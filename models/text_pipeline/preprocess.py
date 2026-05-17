import string
import re

def clean_text(text):
    """
    Cleans raw transcript text by converting to lowercase, 
    removing punctuation, and stripping extra whitespace.
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra spaces
    text = re.sub(' +', ' ', text).strip()
    
    return text