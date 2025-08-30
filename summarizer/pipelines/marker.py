import os
import re
import requests
import subprocess
from pymongo import MongoClient
from openai import OpenAI
from preprocess_json import preprocess_json
from groq_api import summarize_groq
from mistral_api import summarize_mistral

# Add import for pandoc pipeline
from utils.pandoc_api import pdf_to_markdown_pandoc
from utils.pdf_ocr import pdf_to_markdown_ocr
# from grobid_pipeline import tei_xml_to_markdown


# --- CONFIG ---
CONFIG = {
    'summarizer': 'groq',  # Options: 'deepseek', 'claude', groq, mistral
    'mongodb_uri': 'mongodb+srv://Person:Password123@cluster1.0ukuklg.mongodb.net/?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE',
    'mongodb_db': 'papers',
    'mongodb_collection': 'summaries',
    'openrouter_api_key': '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',  # Changed to OpenRouter
    'claude_api_key': os.getenv('CLAUDE_API_KEY'),
    'site_url': os.getenv('SITE_URL', 'http://localhost'),
    'site_name': os.getenv('SITE_NAME', 'PDF Pipeline'),
    'marker_path': os.getenv('MARKER_PATH', ''),

    'use_ocr': os.getenv('USE_OCR', 'false').lower() == 'true',
    'use_pandoc': os.getenv('USE_PANDOC', 'false').lower() == 'true',

    # NEW: prefer the marker Python library instead of calling the CLI
    'use_marker_python': True,

    'optimized_mode': False,
    'section_batch_size': 3, # Custom batch size for section summaries
}

# Initialize OpenRouter client
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=CONFIG['openrouter_api_key'],
)

# --- PDF TO MARKDOWN CONVERSION ---    
def pdf_to_markdown(pdf_path):
    """Convert PDF to markdown using Marker Python API (preferred) or fallback to CLI/Pandoc/OCR."""
    # Prefer Python API when enabled
    if CONFIG.get('use_marker_python', False):
        print("Converting PDF to markdown using Marker Python API...")
        markdown_content = pdf_to_markdown_python(pdf_path)
        if markdown_content:
            os.makedirs('output', exist_ok=True)
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            markdown_path = f"output/{pdf_name}.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"Wrote markdown to {markdown_path}")
            return markdown_content
        print("Marker Python API failed, falling back to other converters...")

    if CONFIG.get('use_pandoc', False):
        print("Converting PDF to markdown using Pandoc...")
        return pdf_to_markdown_pandoc(pdf_path)
    if CONFIG.get('use_ocr', False):
        print("Converting PDF to markdown using OCR...")
        return pdf_to_markdown_ocr(pdf_path)

    try:
        os.makedirs('output', exist_ok=True)
        print("Converting PDF to markdown using Marker CLI...", CONFIG['marker_path'])
        # pass the PDF file (not its directory)
        result = subprocess.run([
            CONFIG['marker_path'],
            pdf_path,
            '--output_dir',
            'output'
        ], capture_output=True, text=True, check=True)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        markdown_path = f"output/{pdf_name}.md"
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        return markdown_content
    except subprocess.CalledProcessError as e:
        print(f"Error converting PDF to markdown (CLI) exit {e.returncode}")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        return None
    except FileNotFoundError:
        print("Marker CLI not found. Trying Python API as fallback...")
        return pdf_to_markdown_python(pdf_path)

def pdf_to_markdown_python(pdf_path):
    """Use marker Python API to convert a PDF to markdown text with debug info and output saved."""
    try:
        import traceback
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
        except Exception as imp_e:
            print("Import error for marker modules:", imp_e)
            raise

        # print marker package info if available
        try:
            import marker
            print("marker package file:", getattr(marker, '__file__', None))
            print("marker package version:", getattr(marker, '__version__', None))
        except Exception:
            pass

        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(pdf_path)
        text, _, images = text_from_rendered(rendered)

        text = text.replace('\r\n', '\n').strip()

        # Save for inspection
        try:
            os.makedirs('output', exist_ok=True)
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            markdown_path = f"output/{pdf_name}.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Marker Python API wrote markdown to {markdown_path}")
        except Exception as e_save:
            print("Failed to save markdown output:", e_save)

        return text
    except ImportError:
        print("Could not import marker modules. Please run: pip3 install marker-pdf (inside your venv).")
        return None
    except Exception:
        import traceback
        print("Error using Marker Python API. Full traceback:")
        traceback.print_exc()
        return None

# --- MARKDOWN PARSING ---
def parse_markdown_sections(markdown_content):
    """Parse markdown content into sections and structure"""
    lines = markdown_content.split('\n')
    sections = {}
    structure = []
    current_section = None
    current_content = []
    
    for line in lines:
        # Check for headers (# ## ### etc.)
        if line.startswith('#'):
            # Save previous section
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Start new section
            current_section = line.strip('#').strip()
            structure.append(current_section)
            current_content = []
        else:
            current_content.append(line)
    
    # Save the last section
    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections, structure

# --- SUMMARIZATION ---
def summarize_deepseek_openrouter(text, prompt):
    """Use DeepSeek via OpenRouter with optional system instructions"""
    try:
        messages = []
        system_instructions = """
You are an expert scientific summarizer.

Your task is to read academic markdown text (extracted from a research paper) and produce summaries that:
- Use the same **formal, academic tone** as the original paper.
- Group related concepts into **logically coherent paragraphs**.
- Separate each paragraph with **two newline characters (`\\n\\n`)**.
- **Do not copy text verbatim**, unless quoting technical terms or definitions.
- **Clean up formatting artifacts** like markdown symbols (`*`, `**`, headers) and OCR noise (e.g., broken sentences or page numbers).

Your output should be clear, precise, and professional—suitable for a research audience and frontend presentation.
"""
        messages.append({"role": "system", "content": system_instructions})
        messages.append({"role": "user", "content": f"{prompt}\n\nText:\n{text}"})

        completion = openrouter_client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": CONFIG['site_url'],
                "X-Title": CONFIG['site_name'],
            },
            model="deepseek/deepseek-chat-v3-0324:free",
            # model="qwen/qwen3-coder:free",
            messages=messages,
            max_tokens=1024,
            temperature=0.2
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with OpenRouter DeepSeek API: {e}")
        return "Error generating summary"

def summarize_claude(text, prompt):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": CONFIG['claude_api_key'],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": f"{prompt}\n\nText:\n{text}"}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['content'][0]['text'].strip()
    except Exception as e:
        print(f"Error with Claude API: {e}")
        return "Error generating summary"

def generate_summary(text, prompt):
    if CONFIG['summarizer'] == 'deepseek':
        return summarize_deepseek_openrouter(text, prompt)
    elif CONFIG['summarizer'] == 'groq':
        return summarize_groq(text, prompt)
    elif CONFIG['summarizer'] == 'mistral':
        return summarize_mistral(text, prompt)
    else:
        return summarize_claude(text, prompt)

# --- ANALYSIS FUNCTIONS ---
def analyze_paper(markdown_content, sections):
    """Generate comprehensive analysis of the paper, supporting batched or legacy mode."""
    analysis = {}

    # Overall summary (always single call)
    print("Generating overall summary...")
    overall_prompt = "Provide a comprehensive summary of this research paper in 2-3 paragraphs."
    analysis['overall_summary'] = generate_summary(markdown_content[:3000], overall_prompt)

    if CONFIG.get('optimized_mode', False):
        # --- Optimized (batched) mode ---
        print("Generating section-wise summaries (batched)...")
        analysis['section_summaries'] = batch_section_summaries(sections)

        print("Generating contributions, methodology, and key findings (batched)...")
        batch_results = batch_contributions_methodology_findings(markdown_content)
        analysis['contributions'] = batch_results.get('contributions', "Not available.")
        analysis['methodology'] = batch_results.get('methodology', "Not available.")
        analysis['key_findings'] = batch_results.get('key_findings', "Not available.")
    else:
        # --- Legacy (per-section/per-field) mode ---
        print("Generating section-wise summaries...")
        section_summaries = {}
        for title, content in sections.items():
            if content.strip() and len(content) > 100:
                section_prompt = f"Summarize this section '{title}' in 1-2 paragraphs."
                section_summaries[title] = generate_summary(content[:2000], section_prompt)
        analysis['section_summaries'] = section_summaries

        print("Identifying contributions...")
        contributions_prompt = (
            "List the main contributions of this research paper. "
            "Format as bullet points and be specific about what new knowledge or methods are introduced."
        )
        analysis['contributions'] = generate_summary(markdown_content[:3000], contributions_prompt)

        print("Analyzing methodology...")
        methodology_prompt = (
            "Describe the methodology used in this research paper. "
            "Include the approach, techniques, and experimental setup if applicable."
        )
        analysis['methodology'] = generate_summary(markdown_content[:3000], methodology_prompt)

        print("Extracting key findings...")
        findings_prompt = (
            "What are the key findings and results of this research? "
            "Summarize the main outcomes and their significance."
        )
        analysis['key_findings'] = generate_summary(markdown_content[:3000], findings_prompt)

    return analysis

def batch_section_summaries(sections):
    """
    Summarize sections in batches to avoid overloading the LLM.
    Returns a dict mapping section titles to summaries.
    """
    import json

    batch_size = CONFIG.get('section_batch_size', 3)
    filtered_sections = [
        (title, content)
        for title, content in sections.items()
        if content.strip() and len(content) > 100
    ]
    all_summaries = {}

    for i in range(0, len(filtered_sections), batch_size):
        batch = filtered_sections[i:i+batch_size]
        prompt = (
            "For each section below, provide a 1-2 paragraph summary. "
            "Return your answer as a JSON object mapping section titles to their summaries. "
            "IMPORTANT: Only output the JSON object. Do NOT include any explanations, markdown, or code blocks. "
            "Do NOT wrap the JSON in ``` or any other formatting. Only output valid JSON.\n\n"
        )
        for title, content in batch:
            prompt += f"Section: {title}\nText:\n{content}\n\n"

        response = generate_summary("", prompt)
        response = re.sub(r"``[json|](http://_vscodecontentref_/5)``", "", response).strip()

        match = re.search(r'({[\s\S]*})', response)
        if match:
            json_str = match.group(1)
        else:
            json_str = response  # fallback

        try:
            batch_summaries = json.loads(json_str)
            all_summaries.update(batch_summaries)
        except Exception as e:
            print("Error parsing LLM response as JSON:", e)
            # Fallback: summarize each section individually in this batch
            for title, content in batch:
                section_prompt = f"Summarize this section '{title}' in 1-2 paragraphs."
                all_summaries[title] = generate_summary(content[:2000], section_prompt)

    return all_summaries

def batch_contributions_methodology_findings(markdown_content):
    """
    Ask for contributions, methodology, and key findings in a single LLM call.
    Returns a dict with keys: 'contributions', 'methodology', 'key_findings'.
    """
    prompt = (
        "Given the following research paper content, answer the following:\n"
        "1. List the main contributions as bullet points.\n"
        "2. Describe the methodology (approach, techniques, experimental setup).\n"
        "3. Summarize the key findings and results.\n\n"
        "Return your answer as a JSON object with keys: 'contributions', 'methodology', 'key_findings'. "
        "IMPORTANT: Only output the JSON object. Do NOT include any explanations, markdown, or code blocks. "
        "IMPORTANT: Only output the JSON object. Do NOT include any explanations, markdown, or code blocks. Do NOT wrap the JSON in ``` or any other formatting. Only output valid JSON."
        "Do NOT wrap the JSON in ``` or any other formatting. Only output valid JSON.\n\n"
        f"Text:\n{markdown_content[:3000]}"
    )
    response = generate_summary("", prompt)
    import json
    try:
        return json.loads(response)
    except Exception as e:
        print("Error parsing LLM response as JSON:", e)
        return {
            "contributions": "Not available.",
            "methodology": "Not available.",
            "key_findings": "Not available."
        }


# --- MONGODB ---
def store_in_mongodb(pdf_filename, structure, analysis):
    try:
        client = MongoClient(CONFIG['mongodb_uri'])
        db = client[CONFIG['mongodb_db']]
        collection = db[CONFIG['mongodb_collection']]
        
        doc = {
            'filename': pdf_filename,
            'structure': structure,
            'overall_summary': analysis['overall_summary'],
            'section_summaries': analysis['section_summaries'],
            'contributions': analysis['contributions'],
            'methodology': analysis['methodology'],
            'key_findings': analysis['key_findings'],
            'timestamp': __import__('datetime').datetime.now()
        }
        processed_doc = preprocess_json(doc)
        result = collection.insert_one(processed_doc)
        print(f"Document inserted with ID: {result.inserted_id}")
    except Exception as e:
        print(f"MongoDB error: {e}")
        # Fallback to JSON file
        print("Saving to JSON file instead...")
        save_to_json_file(pdf_filename, structure, analysis)

def save_to_json_file(pdf_filename, structure, analysis):
    """Save analysis to JSON file as fallback"""
    import json
    from datetime import datetime
    output_file = f"output/{pdf_filename}_analysis_test_local_mistral.json"
    doc = {
        'filename': pdf_filename,
        'structure': structure,
        'overall_summary': analysis['overall_summary'],
        'section_summaries': analysis['section_summaries'],
        'contributions': analysis['contributions'],
        'methodology': analysis['methodology'],
        'key_findings': analysis['key_findings'],
        'timestamp': datetime.now().isoformat()  # <-- fix here
    }
    processed_doc = preprocess_json(doc)
    with open(output_file, 'w') as f:
        json.dump(processed_doc, f, indent=2, ensure_ascii=False)
    print(f"Analysis saved to {output_file}")

# --- MAIN PIPELINE ---
import time
def process_pdf_with_marker(pdf_path):
    pdf_filename = os.path.basename(pdf_path)
    print(f"Processing {pdf_filename}...")

    # Step 1: Convert PDF to Markdown
    print("Converting PDF to markdown...")
    t0 = time.time()
    markdown_content = pdf_to_markdown(pdf_path)
    t1 = time.time()
    print(f"PDF to markdown took {t1 - t0:.2f} seconds")
    if not markdown_content:
        print("Failed to convert PDF to markdown")
        return

    # Step 2: Parse sections
    print("Parsing sections...")
    t2 = time.time()
    sections, structure = parse_markdown_sections(markdown_content)
    t3 = time.time()
    print(f"Parsing sections took {t3 - t2:.2f} seconds")
    print(f"Found {len(sections)} sections: {structure}")

    # Step 3: Generate analysis
    print("Generating analysis...")
    t4 = time.time()
    analysis = analyze_paper(markdown_content, sections)
    t5 = time.time()
    print(f"Generating analysis took {t5 - t4:.2f} seconds")

    # Step 4: Store results
    print("Storing results...")
    t6 = time.time()
    # store_in_mongodb(pdf_filename, structure, analysis)
    save_to_json_file(pdf_filename, structure, analysis)
    t7 = time.time()
    print(f"Storing results took {t7 - t6:.2f} seconds")

    print("Processing complete!")
    print(f"Total time: {t7 - t0:.2f} seconds")

if __name__ == "__main__":
    pdf_path = input("Enter PDF file path: ").strip()
    process_pdf_with_marker(pdf_path)

