import os
import subprocess

def pdf_to_markdown_pandoc(pdf_path, output_dir="output"):
    """
    Convert PDF to markdown using Pandoc.
    Requires Pandoc to be installed and available in PATH.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    markdown_path = os.path.join(output_dir, f"{pdf_name}.md")
    try:
        result = subprocess.run(
            ["pandoc", pdf_path, "-o", markdown_path],
            capture_output=True, text=True, check=True
        )
        with open(markdown_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
        return markdown_content
    except subprocess.CalledProcessError as e:
        print(f"Pandoc conversion failed: {e}")
        print(f"Pandoc stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Pandoc is not installed or not found in PATH.")
        return None

if __name__ == "__main__":
    pdf_path = input("Enter PDF file path: ").strip()
    md = pdf_to_markdown_pandoc(pdf_path)
    if md:
        print("Markdown conversion successful. First 500 chars:")
        print(md[:500])
    else:
        print("Markdown conversion failed.")