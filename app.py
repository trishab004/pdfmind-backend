from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
from transformers import pipeline
import torch

app = Flask(__name__)
CORS(app)

# Load free AI models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

pdf_text_cache = ""

@app.route('/summarize', methods=['POST'])
def summarize_pdf():
    global pdf_text_cache
    file = request.files['file']
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    pdf_text_cache = text

    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summary = ""
    for chunk in chunks:
        try:
            result = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
            summary += result[0]['summary_text'] + "\n"
        except Exception as e:
            print("Summarizer error:", e)
            continue

    return jsonify({'summary': summary})

@app.route('/ask', methods=['POST'])
def ask_pdf():
    global pdf_text_cache
    question = request.json['question']
    if not pdf_text_cache.strip():
        return jsonify({'answer': "Please upload a PDF first."})
    try:
        result = qa_pipeline({
            'question': question,
            'context': pdf_text_cache[:4000]
        })
        return jsonify({'answer': result['answer']})
    except Exception as e:
        print("QnA error:", e)
        return jsonify({'answer': "Something went wrong answering your question."})

if __name__ == '__main__':
    app.run(debug=True)
