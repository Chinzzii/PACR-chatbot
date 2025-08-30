import os
import re
import json
import time
import subprocess
import requests
from pdf2image import convert_from_path
import pytesseract
from openai import OpenAI

from ..utils.preprocess_json import preprocess_json
from  ..utils.groq_api import summarize_groq
from ..utils.mistral_api import summarize_mistral
import glob


# --- CONFIG ---
CONFIG = {
    'summarizer': 'groq',  # Options: 'deepseek', 'claude', groq, mistral
    'mongodb_uri': 'mongodb+srv://Person:Password123@cluster1.0ukuklg.mongodb.net/?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE',
    'mongodb_db': 'papers',
    'mongodb_collection': 'summaries',
    'openrouter_api_key': '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',  # Changed to OpenRouter
    'claude_api_key': os.getenv('CLAUDE_API_KEY'),
    'site_url': os.getenv('SITE_URL', 'http://localhost'),  # Optional
    'site_name': os.getenv('SITE_NAME', 'PDF Pipeline'),  
    'marker_path': os.getenv('MARKER_PATH', ''),  
    'use_ocr': os.getenv('USE_OCR', 'false').lower() == 'true', 
    'output_dir': 'output', 
    # Optional
    'optimized_mode': False,  # Toggle this to switch modes
    'section_batch_size': 3, # Custom batch size for section summaries
    'max_retries': 3,  # Add retry configuration
}


def pdf_to_markdown_tesseract(pdf_path, output_dir="output"):
    """
    Convert scanned PDF to markdown using pytesseract OCR.
    Each page is OCR'd and added as a markdown section.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    markdown_path = os.path.join(output_dir, f"{pdf_name}_ocr.md")
    try:
        images = convert_from_path(pdf_path)
        md_content = ""
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img)
            md_content += f"\n\n# Page {i+1}\n\n{text.strip()}\n"
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"OCR Markdown saved to {markdown_path}")
        return md_content
    except Exception as e:
        print(f"OCR conversion failed: {e}")
        return None

def parse_markdown_sections_pagewise(markdown_content):
    """
    Parse markdown content into sections using page headers and common section titles.
    Returns sections dict and structure list.
    """
    pages = re.split(r'\n#+\s*Page \d+\s*\n', markdown_content)
    sections = {}
    structure = []
    # Expanded and flexible pattern
    known_headers = [
        "Abstract", "Aim", "Aims", "Objective", "Objectives", "Introduction", "Background",
        "Methods", "Materials", "Materials and Methods", "Methodology", "Experimental Setup",
        "Patients and Methods", "Results", "Discussion", "Conclusion", "Conclusions",
        "References", "Acknowledgements", "Related Work", "Future Work", "Limitations",
        "Supplementary Data", "Appendix"
    ]
    header_pattern = r'^(%s)\b' % "|".join([re.escape(h) for h in known_headers])
    flexible_pattern = r'^([A-Z][A-Za-z0-9 &/-]{2,60})\n'
    section_pattern = re.compile(f'{header_pattern}|{flexible_pattern}', re.IGNORECASE | re.MULTILINE)

    for page_num, page in enumerate(pages[1:], 1):
        found = False
        for match in section_pattern.finditer(page):
            title = match.group(1) or match.group(2)
            if not title:
                continue
            start = match.start()
            end = match.end()
            next_match = section_pattern.search(page, end)
            section_text = page[start:next_match.start()] if next_match else page[start:]
            norm_title = title.strip().title()
            if norm_title not in sections:
                sections[norm_title] = section_text.strip()
                structure.append(norm_title)
                found = True
        if not found:
            sections[f"Page {page_num}"] = page.strip()
            structure.append(f"Page {page_num}")
    return sections, structure

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

# def analyze_paper(markdown_content, sections):
#     """Generate analysis similar to pipeline_test.py, but with dummy summaries."""
#     analysis = {}

#     # Overall summary
#     print("Generating overall summary...")
#     t0 = time.time()
#     overall_prompt = "Provide a comprehensive summary of this research paper in 2-3 paragraphs."
#     analysis['overall_summary'] = generate_summary(markdown_content[:3000], overall_prompt)
#     t1 = time.time()
#     print(f"Overall summary took {t1 - t0:.2f} seconds")

#     # Section summaries
#     print("Generating section-wise summaries...")
#     t2 = time.time()
#     section_summaries = {}
#     for title, content in sections.items():
#         if content.strip() and len(content) > 50:
#             section_prompt = f"Summarize this section '{title}' in 1-2 paragraphs."
#             section_summaries[title] = dummy_generate_summary(content[:2000], section_prompt)
#     analysis['section_summaries'] = section_summaries
#     t3 = time.time()
#     print(f"Section summaries took {t3 - t2:.2f} seconds")

#     # Contributions
#     print("Identifying contributions...")
#     t4 = time.time()
#     contributions_prompt = (
#         "List the main contributions of this research paper. "
#         "Format as bullet points and be specific about what new knowledge or methods are introduced."
#     )
#     analysis['contributions'] = dummy_generate_summary(markdown_content[:3000], contributions_prompt)
#     t5 = time.time()
#     print(f"Contributions took {t5 - t4:.2f} seconds")

#     # Methodology
#     print("Analyzing methodology...")
#     t6 = time.time()
#     methodology_prompt = (
#         "Describe the methodology used in this research paper. "
#         "Include the approach, techniques, and experimental setup if applicable."
#     )
#     analysis['methodology'] = dummy_generate_summary(markdown_content[:3000], methodology_prompt)
#     t7 = time.time()
#     print(f"Methodology took {t7 - t6:.2f} seconds")

#     # Key findings
#     print("Extracting key findings...")
#     t8 = time.time()
#     findings_prompt = (
#         "What are the key findings and results of this research? "
#         "Summarize the main outcomes and their significance."
#     )
#     analysis['key_findings'] = dummy_generate_summary(markdown_content[:3000], findings_prompt)
#     t9 = time.time()
#     print(f"Key findings took {t9 - t8:.2f} seconds")

#     return analysis

def save_to_json_file(pdf_filename, structure, analysis):
    """Save analysis to JSON file as fallback"""
    output_file = f"{CONFIG['output_dir']}/{pdf_filename}_analysis_tesseract.json"
    doc = {
        'filename': pdf_filename,
        'structure': structure,
        'overall_summary': analysis['overall_summary'],
        'section_summaries': analysis['section_summaries'],
        'contributions': analysis['contributions'],
        'methodology': analysis['methodology'],
        'key_findings': analysis['key_findings'],
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    processed_doc = preprocess_json(doc)
    with open(output_file, 'w') as f:
        json.dump(processed_doc, f, indent=2, ensure_ascii=False)
    print(f"Analysis saved to {output_file}")

def save_raw_llm_response(pdf_filename, raw_response):
    """Save raw LLM response directly to JSON file"""
    output_file = f"{CONFIG['output_dir']}/{pdf_filename}_raw_llm_response.json"
    
    # Clean up response (remove any markdown formatting)
    import re
    cleaned_response = re.sub(r'```json\s*', '', raw_response)
    cleaned_response = re.sub(r'```\s*', '', cleaned_response)
    cleaned_response = cleaned_response.strip()
    
    try:
        # Parse to ensure it's valid JSON, then save
        import json
        parsed_json = json.loads(cleaned_response)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, indent=2, ensure_ascii=False)
        
        print(f"Raw LLM response saved to {output_file}")
        return True
    except json.JSONDecodeError as e:
        print(f"Failed to save raw response as JSON: {e}")
        # Save as text file instead
        text_file = f"{CONFIG['output_dir']}/{pdf_filename}_raw_llm_response.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(raw_response)
        print(f"Raw LLM response saved as text to {text_file}")
        return False

def analyze_paper_single_call(markdown_content):
    """
    Analyze entire paper with ONE LLM call to get sections, summaries, and analysis.
    This minimizes API costs and latency.
    """
    # Truncate content if too large (keep first portion which usually has most sections)
    max_chars = 15000  # Adjust based on your model's context window
    content = markdown_content[:max_chars] if len(markdown_content) > max_chars else markdown_content

    comprehensive_prompt = f"""
Analyze this research paper and provide a comprehensive analysis in the following JSON format. 
IMPORTANT: Return ONLY valid JSON, no explanation or markdown formatting.

Identify ALL sections present in the paper (do not assume specific section names). 
For each section you find, provide a 1-2 paragraph summary.

{{
  "overall_summary": "2-3 paragraph summary of the entire paper",
  "sections_identified": ["list", "of", "ALL", "section", "names", "found", "in", "the", "paper"],
  "section_summaries": {{
    "Section Name 1": "1-2 paragraph summary of this section",
    "Section Name 2": "1-2 paragraph summary of this section",
    "Section Name N": "1-2 paragraph summary of this section"
  }},
  "contributions": "Bullet points of main contributions and novel aspects",
  "methodology": "Description of research approach and techniques used",
  "key_findings": "Summary of main results and their significance"
}}

Instructions:
- Find ALL sections in the paper, regardless of their names
- Include every section that has substantial content (more than just a few sentences)
- Use the exact section names as they appear in the paper
- If sections have variations in naming (e.g., "Materials and Methods", "Methodology"), use the exact title from the paper
- Include sections like Abstract, Introduction, Background, Literature Review, Methods, Results, Discussion, Conclusion, References, Acknowledgments, Appendix, etc. - but ONLY if they actually exist in the paper
- Do not invent sections that don't exist

Paper text:
{content}
"""

    print("Generating comprehensive analysis with single LLM call...")
    t0 = time.time()

    # Add retry logic
    for attempt in range(CONFIG['max_retries']):
        try:
            print(f"Attempt {attempt + 1}/{CONFIG['max_retries']}")
            response = generate_summary("", comprehensive_prompt)  # Empty text since prompt contains everything

            # Log the raw summary response
            print("\n--- Raw LLM Summary Response ---")
            print(response[:2000])  # Print first 2000 chars for brevity

            # Clean up response (remove any markdown formatting)
            import re
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            response = response.strip()

            # Parse JSON response
            import json
            analysis_data = json.loads(response)

            # Log the parsed JSON
            print("\n--- Parsed JSON from LLM ---")
            print(json.dumps(analysis_data, indent=2, ensure_ascii=False)[:2000])  # Print first 2000 chars

            # Extract structure from identified sections
            structure = analysis_data.get('sections_identified', [])
            print("Sections Identified", structure)
            # Build analysis dict in expected format
            analysis = {
                'overall_summary': analysis_data.get('overall_summary', 'Summary not available'),
                'section_summaries': analysis_data.get('section_summaries', {}),
                'contributions': analysis_data.get('contributions', 'Contributions not available'),
                'methodology': analysis_data.get('methodology', 'Methodology not available'),
                'key_findings': analysis_data.get('key_findings', 'Key findings not available')
            }

            t1 = time.time()
            print(f"Single-call analysis took {t1 - t0:.2f} seconds")
            print(f"Identified sections: {structure}")

            return structure, analysis, response  # Return raw response too

        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1} failed: JSON parsing error: {e}")
            print(f"Raw response: {response[:500]}...")
            
            if attempt == CONFIG['max_retries'] - 1:  # Last attempt
                print("All retry attempts failed. Using fallback.")
                # Fallback: extract what we can from malformed response
                fallback_prompt = "Summarize the following research paper text."
                return [], {
                    'overall_summary': generate_summary(content, fallback_prompt),
                    'section_summaries': {},
                    'contributions': '',
                    'methodology': '',
                    'key_findings': ''
                }, response  # Return raw response even on error
            else:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)  # Brief delay before retry

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: Unexpected error: {e}")
            
            if attempt == CONFIG['max_retries'] - 1:  # Last attempt
                print("All retry attempts failed. Using fallback.")
                fallback_prompt = "Summarize the following research paper text."
                return [], {
                    'overall_summary': generate_summary(content, fallback_prompt),
                    'section_summaries': {},
                    'contributions': '',
                    'methodology': '',
                    'key_findings': ''
                }, ""
            else:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)  # Brief delay before retry

def analyze_paper_two_calls(markdown_content):
    """
    Analyze paper with TWO LLM calls for better reliability:
    1. Structure and section summaries
    2. Contributions, methodology, and key findings
    """
    # Truncate content if too large
    max_chars = 15000
    content = markdown_content[:max_chars] if len(markdown_content) > max_chars else markdown_content

    print("Generating analysis with two LLM calls...")
    t0 = time.time()

    # CALL 1: Structure and Section Summaries
    structure_prompt = f"""
Analyze this research paper and provide detailed structure analysis in the following JSON format.
IMPORTANT: Return ONLY valid JSON, no explanation or markdown formatting.

{{
  "overall_summary": "3-4 paragraph comprehensive summary covering: research problem, methodology approach, key findings, and significance/implications",
  "sections_identified": ["list", "of", "ALL", "section", "names", "found", "in", "the", "paper"],
  "section_summaries": {{
    "Section Name 1": "Detailed 2-3 paragraph summary covering: main points, key concepts, methodological details, findings, and relevance to overall research",
    "Section Name 2": "Detailed 2-3 paragraph summary covering: main points, key concepts, methodological details, findings, and relevance to overall research",
    "Section Name N": "Detailed 2-3 paragraph summary covering: main points, key concepts, methodological details, findings, and relevance to overall research"
  }}
}}

Instructions for section summaries:
- Provide 2-3 detailed paragraphs for each section (not just 1-2 sentences)
- Include specific details, numbers, methods, or findings mentioned
- Explain the significance of each section to the overall research
- Cover key concepts, methodologies, results, or arguments presented
- Use academic language but make it accessible
- If it's a Methods section, include specific techniques, tools, or procedures
- If it's a Results section, include key findings, statistics, or outcomes
- If it's a Discussion section, include interpretations, implications, and limitations

Find ALL sections in the paper, regardless of their names:
- Include every section that has substantial content (more than just a few sentences)
- Use the exact section names as they appear in the paper
- Include sections like Abstract, Introduction, Background, Literature Review, Methods, Results, Discussion, Conclusion, References, Acknowledgments, Appendix, etc. - but ONLY if they actually exist

Paper text:
{content}
"""

    print("Call 1: Getting structure and section summaries...")
    structure_data = None
    for attempt in range(CONFIG['max_retries']):
        try:
            print(f"Structure call attempt {attempt + 1}/{CONFIG['max_retries']}")
            response1 = generate_summary("", structure_prompt)
            
            # Clean and parse response
            import re, json
            cleaned = re.sub(r'```json\s*', '', response1)
            cleaned = re.sub(r'```\s*', '', cleaned).strip()
            structure_data = json.loads(cleaned)
            
            print("✓ Structure call successful")
            break
            
        except json.JSONDecodeError as e:
            print(f"Structure call attempt {attempt + 1} failed: {e}")
            if attempt == CONFIG['max_retries'] - 1:
                print("Structure call failed completely")
                structure_data = {
                    "overall_summary": "Summary not available",
                    "sections_identified": [],
                    "section_summaries": {}
                }
            else:
                time.sleep(2)

    # CALL 2: Analysis (Contributions, Methodology, Key Findings)
    analysis_prompt = f"""
Analyze this research paper and provide detailed analysis in the following JSON format.
IMPORTANT: Return ONLY valid JSON, no explanation or markdown formatting.

{{
  "contributions": "• Bullet point 1: Detailed description of first contribution\\n• Bullet point 2: Detailed description of second contribution\\n• Bullet point 3: Detailed description of third contribution",
  "methodology": "Detailed 2-3 paragraph description covering: research design, data collection methods, analytical techniques, tools/software used, sample size, experimental setup, validation methods, and any limitations",
  "key_findings": "• Key finding 1: Detailed description with specific results, numbers, or outcomes\\n• Key finding 2: Detailed description with specific results, numbers, or outcomes\\n• Key finding 3: Detailed description with specific results, numbers, or outcomes"
}}

Instructions:
- For contributions: Use bullet points (•) and describe what new knowledge, methods, techniques, or insights this research provides
- For methodology: Provide comprehensive description of how the research was conducted, including specific methods, tools, and approaches
- For key findings: Use bullet points (•) and include specific results, statistics, outcomes, or discoveries
- Be specific and include numbers, percentages, or quantitative results where available
- Focus on what makes this research novel and significant
- Include both theoretical and practical implications

Paper text:
{content}
"""

    print("Call 2: Getting contributions, methodology, and key findings...")
    analysis_data = None
    for attempt in range(CONFIG['max_retries']):
        try:
            print(f"Analysis call attempt {attempt + 1}/{CONFIG['max_retries']}")
            response2 = generate_summary("", analysis_prompt)
            
            # Clean and parse response
            import re, json
            cleaned = re.sub(r'```json\s*', '', response2)
            cleaned = re.sub(r'```\s*', '', cleaned).strip()
            analysis_data = json.loads(cleaned)
            
            print("✓ Analysis call successful")
            break
            
        except json.JSONDecodeError as e:
            print(f"Analysis call attempt {attempt + 1} failed: {e}")
            if attempt == CONFIG['max_retries'] - 1:
                print("Analysis call failed completely")
                analysis_data = {
                    "contributions": "• Contributions not available",
                    "methodology": "Methodology not available", 
                    "key_findings": "• Key findings not available"
                }
            else:
                time.sleep(2)

    # Combine results
    structure = structure_data.get('sections_identified', [])
    analysis = {
        'overall_summary': structure_data.get('overall_summary', 'Summary not available'),
        'section_summaries': structure_data.get('section_summaries', {}),
        'contributions': analysis_data.get('contributions', '• Contributions not available'),
        'methodology': analysis_data.get('methodology', 'Methodology not available'),
        'key_findings': analysis_data.get('key_findings', '• Key findings not available')
    }

    # Combine raw responses for saving
    raw_response = {
        "call_1_structure": response1 if 'response1' in locals() else "",
        "call_2_analysis": response2 if 'response2' in locals() else ""
    }

    t1 = time.time()
    print(f"Two-call analysis took {t1 - t0:.2f} seconds")
    print(f"Identified sections: {structure}")

    return structure, analysis, raw_response

def process_pdf_with_tesseract(pdf_path):
    """
    Process PDF and return analysis as JSON object.
    This is the main function to be called from the chat app.
    """
    pdf_filename = os.path.basename(pdf_path)
    print(f"Processing {pdf_filename}...")

    # Step 1: OCR PDF to Markdown
    print("Converting PDF to markdown with Tesseract OCR...")
    t0 = time.time()
    markdown_content = pdf_to_markdown_tesseract(pdf_path, CONFIG['output_dir'])
    t1 = time.time()
    print(f"OCR to markdown took {t1 - t0:.2f} seconds")
    if not markdown_content:
        print("Failed to convert PDF to markdown")
        return None

    # Step 2: Parse sections (as fallback only)
    print("Parsing sections as fallback...")
    t2 = time.time()
    fallback_sections, fallback_structure = parse_markdown_sections_pagewise(markdown_content)
    t3 = time.time()
    print(f"Parsing sections took {t3 - t2:.2f} seconds")
    print(f"Found {len(fallback_sections)} fallback sections: {fallback_structure}")

    # Step 3: Generate analysis with LLM (TWO CALLS)
    print("Generating analysis...")
    t4 = time.time()
    structure_from_llm, analysis, raw_response = analyze_paper_two_calls(markdown_content=markdown_content)
    t5 = time.time()
    print(f"Generating analysis took {t5 - t4:.2f} seconds")

    # Step 4: Use the structure from LLM, fallback to parsed structure only if LLM failed
    final_structure = structure_from_llm if structure_from_llm else fallback_structure
    print(f"Using structure: {final_structure} (from {'LLM' if structure_from_llm else 'fallback parser'})")
    
    # Step 5: Build final JSON response
    json_result = {
        'filename': pdf_filename,
        'structure': final_structure,
        'overall_summary': analysis['overall_summary'],
        'section_summaries': analysis['section_summaries'],
        'contributions': analysis['contributions'],
        'methodology': analysis['methodology'],
        'key_findings': analysis['key_findings'],
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
        'processing_time': f"{t5 - t0:.2f} seconds"
    }

    # Step 6: Save to file (optional - for backup)
    save_to_json_file(pdf_filename, final_structure, analysis)
    
    # Step 7: Save raw LLM response (optional - for debugging)
    if raw_response:
        save_raw_llm_response(pdf_filename, json.dumps(raw_response, indent=2))

    print("Processing complete!")
    print(f"Total time: {t5 - t0:.2f} seconds")
    
    # Return the JSON result to be used by the chat app
    return json_result

if __name__ == "__main__":
    pdf_path = input("Enter PDF file path: ").strip()
    process_pdf_with_tesseract(pdf_path)