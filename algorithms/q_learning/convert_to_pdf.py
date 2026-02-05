#!/usr/bin/env python3
"""
Convert Q-Learning Master Report from Markdown to PDF
"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path
import os

def markdown_to_pdf(md_file_path, output_pdf_path=None):
    """
    Convert markdown file to PDF using WeasyPrint
    
    Args:
        md_file_path (str): Path to the markdown file
        output_pdf_path (str): Path for the output PDF file (optional)
    """
    
    # Read the markdown file
    with open(md_file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['codehilite', 'fenced_code', 'tables', 'toc'])
    html_content = md.convert(md_content)
    
    # Add CSS styling for better PDF appearance
    css_style = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        line-height: 1.6;
        color: #2d3748;
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
        font-size: 11pt;
    }
    
    h1 {
        color: #1a202c;
        font-weight: 700;
        font-size: 24pt;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #4299e1;
        padding-bottom: 0.5rem;
    }
    
    h2 {
        color: #2d3748;
        font-weight: 600;
        font-size: 18pt;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.25rem;
    }
    
    h3 {
        color: #4a5568;
        font-weight: 600;
        font-size: 14pt;
        margin-top: 1.25rem;
        margin-bottom: 0.5rem;
    }
    
    h4 {
        color: #718096;
        font-weight: 600;
        font-size: 12pt;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    p {
        margin-bottom: 1rem;
        text-align: justify;
    }
    
    code {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        background-color: #f7fafc;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        border: 1px solid #e2e8f0;
        font-size: 10pt;
    }
    
    pre {
        background-color: #2d3748;
        color: #f7fafc;
        padding: 1rem;
        border-radius: 6px;
        overflow-x: auto;
        margin: 1rem 0;
        border-left: 4px solid #4299e1;
    }
    
    pre code {
        background: transparent;
        border: none;
        padding: 0;
        color: inherit;
        font-size: 10pt;
    }
    
    ul, ol {
        margin-bottom: 1rem;
        padding-left: 1.5rem;
    }
    
    li {
        margin-bottom: 0.25rem;
    }
    
    strong {
        font-weight: 600;
        color: #1a202c;
    }
    
    em {
        font-style: italic;
        color: #4a5568;
    }
    
    blockquote {
        border-left: 4px solid #4299e1;
        margin: 1rem 0;
        padding-left: 1rem;
        color: #4a5568;
        font-style: italic;
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 10pt;
    }
    
    th, td {
        border: 1px solid #e2e8f0;
        padding: 0.5rem;
        text-align: left;
    }
    
    th {
        background-color: #f7fafc;
        font-weight: 600;
        color: #2d3748;
    }
    
    .page-break {
        page-break-before: always;
    }
    
    @page {
        size: A4;
        margin: 1in;
        @bottom-right {
            content: "Page " counter(page);
        }
    }
    """
    
    # Create complete HTML document
    html_document = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Q-Learning Master Report</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Determine output path
    if output_pdf_path is None:
        md_path = Path(md_file_path)
        output_pdf_path = md_path.with_suffix('.pdf')
    
    # Convert to PDF
    try:
        html = HTML(string=html_document)
        css = CSS(string=css_style)
        html.write_pdf(output_pdf_path, stylesheets=[css])
        
        print(f"✅ Successfully converted to PDF: {output_pdf_path}")
        print(f"📄 File size: {os.path.getsize(output_pdf_path) / 1024:.1f} KB")
        
    except Exception as e:
        print(f"❌ Error converting to PDF: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    # Convert the Q-Learning Master Report
    md_file = "Q_Learning_Master_Report.md"
    pdf_file = "Q_Learning_Master_Report.pdf"
    
    if not os.path.exists(md_file):
        print(f"❌ Markdown file not found: {md_file}")
        exit(1)
    
    print("🔄 Converting Q-Learning Master Report to PDF...")
    success = markdown_to_pdf(md_file, pdf_file)
    
    if success:
        print(f"🎉 Conversion complete! PDF saved as: {pdf_file}")
    else:
        print("❌ Conversion failed!")