import re
import json
from typing import Union

def json_from_response(response: str, verbose: bool=False) -> Union[list, dict]:
    """
    Extract the JSON result from a response.

    It finds the bottom most json block and parses it.
    It ignores json blocks that are not the last one.
    """

    # search from the bottom of the text
    end = response.rfind('```')
    if end == -1:
        raise ValueError("No closing code block found in response.")
    
    # remove the last N characters
    text_until_end = response[:end]
    if verbose:
        print("text_until_end --- BEFORE")
        print(text_until_end)
        print("text_until_end --- AFTER")

    start = text_until_end.rfind('```')
    if start == -1:
        raise ValueError("No opening code block found in response.")
    text = response[start + 3:end]

    # Remove either ^``` or ^```json from text
    text = re.sub(r'^json\s*', '', text.strip(), flags=re.IGNORECASE)
    if verbose:
        print(f"start: {start}, end: {end}")
        print("text --- BEFORE")
        print(text)
        print("text --- AFTER")

    # Parse the JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON")
