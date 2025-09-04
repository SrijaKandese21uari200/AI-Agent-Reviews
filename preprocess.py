import re

def clean_text(text: str, max_len=300) -> str:
    """
    Clean review text: lowercase, remove links/special chars, normalize spaces.
    Truncate to max_len chars to save tokens.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)   # remove links
    text = re.sub(r"[^a-z\s]", " ", text) # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]
