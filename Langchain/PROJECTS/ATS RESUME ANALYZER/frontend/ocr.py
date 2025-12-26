import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from pdf2image import convert_from_path
import pytesseract

def extractTextPypdf(pdfPath):
    loader = PyPDFLoader(pdfPath)
    docs = loader.load()
    return "\n".join(d.page_content for d in docs)

def extractTextOcr(pdfPath):
    images = convert_from_path(pdfPath)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

def extractResumeText(uploadedFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploadedFile.read())
        path = tmp.name

    text = extractTextPypdf(path)

    if len(text.strip()) < 300:
        text = extractTextOcr(path)

    os.remove(path)
    return text