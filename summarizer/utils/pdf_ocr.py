import os
from pdf2image import convert_from_path
import pytesseract

def pdf_to_markdown_ocr(pdf_path, output_dir="output"):
    """
    Convert a scanned PDF to markdown using OCR (pytesseract).
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

if __name__ == "__main__":
    pdf_path = input("Enter PDF file path: ").strip()
    md = pdf_to_markdown_ocr(pdf_path)
    if md:
        print("OCR Markdown conversion successful. First 500 chars:")
        print(md[:500])
    else:
        print("OCR Markdown conversion failed.")