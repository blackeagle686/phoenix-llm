import re
import unicodedata

def clean_html(text: str) -> str:
    """Removes HTML tags and common boilerplate."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(text: str) -> str:
    """Normalizes Unicode characters."""
    return unicodedata.normalize('NFKC', text)

def quality_filter(text: str) -> bool:
    """
    Returns True if the text passes quality heuristics.
    Filters out text that is too short, has too many special characters, 
    or looks like junk/boilerplate.
    """
    if len(text) < 100:
        return False
    
    # Check for excessive punctuation/special characters (common in junk)
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s.,!?]', text))
    if special_chars / len(text) > 0.1:
        return False
    
    # Check for 'lorem ipsum' or other common placeholders
    if "lorem ipsum" in text.lower():
        return False
        
    return True

def exact_deduplicate(text: str, seen_hashes: set) -> bool:
    """Returns True if the text is unique (not in seen_hashes)."""
    import hashlib
    h = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if h in seen_hashes:
        return False
    seen_hashes.add(h)
    return True
