import json
import re
from bs4 import BeautifulSoup
import textwrap

def normalize_section_title(raw_title):
    """
    Remove HTML, markdown, and standardize section titles.
    """
    if not raw_title:
        return ""
    # Remove HTML tags
    text = BeautifulSoup(raw_title, "html.parser").get_text()
    # Remove asterisks, extra whitespace, title case
    text = re.sub(r"\*+", "", text).strip()
    return text.title()

def process_structure_and_sections(data, wrap_width=100):
    """
    Processes both 'structure' and 'section_summaries' to ensure matching keys.
    Returns:
        - Cleaned structure list
        - Cleaned section_summaries with normalized keys and formatted paragraphs
    """
    raw_structure = data.get("structure", [])
    raw_sections = data.get("section_summaries", {})

    cleaned_structure = []
    cleaned_sections = {}

    for raw_key in raw_structure:
        norm_key = normalize_section_title(raw_key)
        cleaned_structure.append(norm_key)

        # Match section summary using original key (fallback to None)
        for k in raw_sections.keys():
            if normalize_section_title(k) == norm_key:
                content = raw_sections[k]
                # Clean markdown and format into paragraphs
                clean_text = re.sub(r"\*+", "", content)
                clean_text = re.sub(r"\s+", " ", clean_text).strip()
                # paragraphs = [textwrap.fill(p.strip(), width=wrap_width) 
                #               for p in clean_text.split("\n\n") if p.strip()]
                paragraphs = clean_text.strip()
                cleaned_sections[norm_key] = paragraphs
                break
        else:
            # If no match found in summaries
            cleaned_sections[norm_key] = ["Summary not available."]

    return {
        "structure": cleaned_structure,
        "section_summaries": cleaned_sections
    }

def process_overall_summary(data):
    """
    Cleans and formats the overall summary:
    - Removes asterisks (*)
    - Normalizes whitespace
    - Wraps text into UI-friendly paragraphs
    """
    summary = data.get("overall_summary", "")

    if not summary.strip():
        return {"overall_summary": "Summary not available."}

    # Remove *, ** and other markdown artifacts
    summary_cleaned = re.sub(r"\*+", "", summary)
    summary_cleaned = re.sub(r"\s+", " ", summary_cleaned).strip()

    # Split by paragraph (if separated by newlines)
    paragraphs = [p.strip() for p in summary_cleaned.split("\n\n") if p.strip()]

    # Optionally wrap lines for frontend
    # formatted_paragraphs = [textwrap.fill(p, width=wrap_width) for p in paragraphs]

    return {"overall_summary": paragraphs}

def get_json_from_user():
    """Gets a JSON file path from the user and prints it."""
    # json_path = input("Enter the path to your JSON file: ").strip()
    json_path = "AphthousStomatitis.pdf_analysis.json"
    json_data = None
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data


def clean_text(text):
    """
    Cleans raw text by removing HTML, markdown, asterisks, and extra whitespace.
    """
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove markdown asterisks
    text = re.sub(r"\*+", "", text)
    # Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()

def split_into_points(text):
    """
    Splits text into points using sentence boundaries.
    Falls back to \n\n if that structure exists.
    # """
    # if "\n\n" in text:
    #     chunks = text.split("\n\n")
    # else:
    #     # Basic sentence splitting using periods (with spacing logic)
    #     # chunks = re.split(r'(?<=[.?!])\s+', text)
        
    chunks = re.split(r'(?<=[.!?])\s+', text)    
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def process_section_summaries(data):
    """
    Processes 'section_summaries' into bullet-point-friendly content.
    Handles both raw string and preprocessed list values.
    """
    raw_sections = data.get("section_summaries", {})
    processed_sections = {}
    print(raw_sections)
    for raw_key, raw_val in raw_sections.items():
        # print(raw_key, raw_val)
        if isinstance(raw_val, list):
            # Already processed, clean each item just in case
            cleaned_list = [clean_text(item) for item in raw_val if isinstance(item, str)]
            processed_sections[raw_key] = cleaned_list
        elif isinstance(raw_val, str):
            clean_val = clean_text(raw_val)
            bullet_points = split_into_points(clean_val)
            processed_sections[raw_key] = bullet_points
        else:
            # Unexpected format
            processed_sections[raw_key] = ["Summary not available."]

    return {"section_summaries": processed_sections}

def preprocess_json(json_data):
    processed_data = process_structure_and_sections(json_data)
    json_data.update(processed_data)
    json_data.update(process_overall_summary(json_data))
    json_data.update(process_section_summaries(json_data))
    return json_data

    
    
if __name__ == "__main__":
    json_data = get_json_from_user()
    print("JSON Data:")
    print(json.dumps(json_data, indent=4))
    preprocess_json(json_data)
    print("Cleaned JSON Data:")
    print(json.dumps(json_data, indent=4))


