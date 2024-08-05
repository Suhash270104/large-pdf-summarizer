import streamlit as st
import os
import tempfile
import re
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader
import torch

nltk.download('punkt')

def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    full_text = ""
    for i in range(len(reader.pages)):
        page = reader.pages[i]
        full_text += page.extract_text()
    return full_text

def load_and_split_pdf_by_chapters(text):
    chapters = re.split(r'(Chapter|CHAPTER) (\d+|[IVXLCDM]+)', text)
    chapters = [chapter.strip() for chapter in chapters if chapter.strip()]
    return chapters

def clean_text(text):
    text = re.sub(r'\bPage \d+\b', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\b[I|V|X|L|C|D|M]+\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(documents, max_tokens, tokenizer):
    sentences = [sent_tokenize(doc) for doc in documents]
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence_list in sentences:
        for sentence in sentence_list:
            sentence_length = len(tokenizer.encode(sentence, add_special_tokens=False))
            if current_length + sentence_length > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def ensure_full_stop(text):
    if text and text[-1] not in ".!?":
        text += "."
    return text

def summarize_texts(chapters, model, tokenizer):
    summaries = []
    for chapter in chapters:
        chapter_summary = ""
        chunks = chunk_text([chapter], 1024, tokenizer)
        for chunk in chunks:
            inputs = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=1024, truncation=True)
            inputs = inputs.to(model.device)  # Ensure inputs are on the same device as the model
            summary_ids = model.generate(inputs, max_length=1024, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
            chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            chunk_summary = ensure_full_stop(chunk_summary)
            chapter_summary += chunk_summary + " "

        chapter_summary = clean_text(chapter_summary)
        summaries.append(chapter_summary.strip())

    return summaries

def summarize_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        temp_path = temp_file.name

    try:
        full_text = load_pdf_text(temp_path)
        chapters = load_and_split_pdf_by_chapters(full_text)
        tokenizer = AutoTokenizer.from_pretrained("pszemraj/long-t5-tglobal-base-16384-book-summary")
        model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/long-t5-tglobal-base-16384-book-summary")
        # Ensure the model is on the appropriate device (GPU if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        chapter_summaries = summarize_texts(chapters, model, tokenizer)

        st.markdown("## Chapter Summaries:")
        for i, summary in enumerate(chapter_summaries, start=1):
            st.markdown(f"### Chapter {i} Summary:")
            st.write(summary)
            st.write("\n\n")

        full_text_summary = " ".join(chapter_summaries)
        full_summary_chunks = chunk_text([full_text_summary], 1024, tokenizer)
        final_summary = ""
        for chunk in full_summary_chunks:
            inputs = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=8192, truncation=True)
            inputs = inputs.to(model.device)  # Ensure inputs are on the same device as the model
            summary_ids = model.generate(inputs, max_length=1024, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
            chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            chunk_summary = ensure_full_stop(chunk_summary)
            final_summary += chunk_summary + " "

        final_summary = clean_text(final_summary)

        st.markdown("## Final Summary:")
        st.write(final_summary.replace('\n', '\n\n'))

    finally:
        os.remove(temp_path)

st.title("Book PDF Summarizer")

pdf_file = st.file_uploader("Upload a PDF book", type="pdf")

if pdf_file:
    if st.button("Generate Summary"):
        st.write("Summarizing PDF, please wait...")
        summarize_pdf(pdf_file)
        st.write("Completed")
