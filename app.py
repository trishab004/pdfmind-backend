from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import requests

app = Flask(__name__)
CORS(app)

pdf_text_cache = ""

# Hugging Face endpoints (FREE)
SUMMARIZER_API = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
QNA_API = "https://api-inference.huggingface.co/models/distilbert-base-cased-distilled-squad"

headers = {
    "Authorization": "Bearer YOUR_HF_TOKEN"  # Optional — you can skip if you don't have one
}

@app.route('/summarize', methods=['POST'])
def summarize_pdf():
    global pdf_text_cache
    file = request.files['file']
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    pdf_text_cache = text

    # Trim text to 1000 chars (API limit)
    chunk = text[:1000]

    payload = {
        "inputs": chunk
    }

    response = requests.post(SUMMARIZER_API, headers=headers, json=payload)

    if response.status_code == 200:
        summary = response.json()[0]['summary_text']
        return jsonify({'summary': summary})
    else:
        return jsonify({'summary': "Error generating summary."})

@app.route('/ask', methods=['POST'])
def ask_pdf():
    global pdf_text_cache
    question = request.json['question']

    if not pdf_text_cache.strip():
        return jsonify({'answer': "Please upload a PDF first."})

    payload = {
        "inputs": {
            "question": question,
            "context": pdf_text_cache[:2000]  # limited for API
        }
    }

    response = requests.post(QNA_API, headers=headers, json=payload)

    if response.status_code == 200:
        answer = response.json()['answer']
        return jsonify({'answer': answer})
    else:
        return jsonify({'answer': "Error processing your question."})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
