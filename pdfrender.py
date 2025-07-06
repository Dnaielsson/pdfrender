import requests
from io import BytesIO
from pdfminer.high_level import extract_text
import re
from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse

app = FastAPI()

def pdf_url_to_text(url):
    response = requests.get(url)
    response.raise_for_status()
    with BytesIO(response.content) as pdf_file:
        text = extract_text(pdf_file)
    return text

def clean_text(text):
    cleaned_lines = []
    for line in text.splitlines():
        # Remove lines that are just numbers (page numbers)
        if re.fullmatch(r'\s*\d+\s*', line):
            continue
        # Add more patterns here if needed (e.g., headers/footers)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

@app.get("/extract", response_class=PlainTextResponse)
def extract_pdf_text(url: str = Query(..., description="PDF file URL")):
    plain_text = pdf_url_to_text(url)
    cleaned = clean_text(plain_text)
    return cleaned