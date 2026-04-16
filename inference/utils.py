"""
Utility functions for the NL2SIM pipeline.
Includes JSON handling, token counting, and other helpers.
"""
import json
from typing import Dict, Any, Optional

def safe_json_parse(json_str: str, default: Optional[Dict] = None) -> Dict:
    """
    Safely parse a JSON string with fallback to default.
    
    Args:
        json_str: String to parse
        default: Default value if parsing fails (empty dict if not provided)
        
    Returns:
        Parsed JSON dict or default
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[Warning] JSON parse error: {e}")
        if default is None:
            return {}
        return default

def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """
    Safely convert object to JSON string.
    
    Args:
        obj: Object to serialize
        indent: Indentation level
        
    Returns:
        JSON string or error representation
    """
    try:
        return json.dumps(obj, indent=indent)
    except (TypeError, ValueError) as e:
        return f"/* JSON serialization error: {e} */"

def count_tokens_approx(text: str) -> int:
    """
    Approximate token count (useful for prompt sizing).
    Uses simple word-based heuristic: ~1 token per 4 characters.
    
    Args:
        text: Text to count
        
    Returns:
        Approximate token count
    """
    return len(text) // 4

def truncate_text(text: str, max_tokens: int, tokenizer=None) -> str:
    """
    Truncate text to approximate token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        tokenizer: Optional tokenizer for accurate counting
        
    Returns:
        Truncated text
    """
    if tokenizer:
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_tokens:
            return tokenizer.convert_tokens_to_string(tokens[:max_tokens])
    else:
        max_chars = max_tokens * 4
        return text[:max_chars]
    return text

def extract_code_block(text: str, language: str = "mx3") -> Optional[str]:
    """
    Extract code block from markdown-formatted text.
    
    Args:
        text: Text potentially containing code block
        language: Language identifier to look for
        
    Returns:
        Extracted code or None if not found
    """
    import re
    pattern = rf"```{language}\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try without language marker
    pattern = r"```\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def normalize_filename(name: str) -> str:
    """
    Normalize a string for use as a filename.
    
    Args:
        name: Original name
        
    Returns:
        Sanitized filename
    """
    import re
    # Remove invalid characters and replace spaces
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.replace(' ', '_')
    return name.lower()

def format_error_message(error_type: str, message: str, context: str = "") -> str:
    """
    Format an error message with type and optional context.
    
    Args:
        error_type: Type of error (e.g., "VALIDATION_ERROR")
        message: Error message
        context: Additional context
        
    Returns:
        Formatted error message
    """
    msg = f"[{error_type}] {message}"
    if context:
        msg += f"\n  Context: {context}"
    return msg

def merge_dicts(base: Dict, updates: Dict, recursive: bool = True) -> Dict:
    """
    Merge updates dict into base dict.
    
    Args:
        base: Base dictionary
        updates: Updates to apply
        recursive: Whether to merge nested dicts
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in updates.items():
        if recursive and isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_dicts(result[key], value, recursive=True)
        else:
            result[key] = value
    return result
