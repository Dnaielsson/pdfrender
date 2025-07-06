import requests
from io import BytesIO
from pdfminer.high_level import extract_text
import re
from fastapi import FastAPI, Query, Header
from fastapi.responses import PlainTextResponse
import openai
import os
import tiktoken

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

def summarize_text(text, api_key, percent=75):
    client = openai.OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    prompt = (
        f"Summarize the following text to about {percent}% of its original length:\n\n{text}"
    )
    response = client.chat.completions.create(
        model="llama3-70b-8192",  # or another Groq-supported model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,  # adjust as needed
        temperature=0.5,
    )
    return response.choices[0].message.content

def count_tokens(text, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

@app.get("/extract", response_class=PlainTextResponse)
def extract_pdf_text(url: str = Query(..., description="PDF file URL")):
    print(f"API called with URL: {url}")
    plain_text = pdf_url_to_text(url)
    cleaned = clean_text(plain_text)
    return cleaned

@app.get("/")
def root():
    return {"message": "PDF Render API is running. No root url exists... (Dnaielsson)"}

@app.get("/summarize", response_class=PlainTextResponse)
def summarize_pdf(
    url: str = Query(..., description="PDF file URL"),
    percent: int = Query(75, description="Percent of original length"),
):
    plain_text = pdf_url_to_text(url)
    token_count = count_tokens(plain_text)
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        return "Groq API key not set in environment."
    summary = summarize_text(plain_text, groq_api_key, percent)
    return f"Original tokens: {token_count}\n\nSummary:\n{summary}"