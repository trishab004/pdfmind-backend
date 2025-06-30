from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import requests
import os

app = Flask(__name__)
CORS(app)

pdf_text_cache = ""

# Hugging Face public endpoints (no token)
SUMMARIZER_API = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
QNA_API = "https://api-inference.huggingface.co/models/distilbert-base-cased-distilled-squad"

headers = {
    "Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"
}

@app.route('/summarize', methods=['POST'])
def summarize_pdf():
    global pdf_text_cache
    try:
        file = request.files['file']
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        pdf_text_cache = text

        # Limit the input for HuggingFace API
        chunk = text[:1000]
        print("📄 Extracted Text (first 1000 chars):\n", chunk[:300], "...")  # Print only first 300 chars for sanity

        payload = { "inputs": chunk }
        response = requests.post(SUMMARIZER_API, headers=headers, json=payload)

        # Raise error for anything that's not 200
        response.raise_for_status()
        summary = response.json()[0]['summary_text']
        return jsonify({'summary': summary})

    except requests.exceptions.HTTPError as e:
        print("🛑 Hugging Face API HTTP error:", response.status_code)
        print("🔍 Details:", response.text)
        return jsonify({'summary': f"Error generating summary: {response.status_code}"})
    except Exception as e:
        print("🔥 Unexpected summarization error:", str(e))
        return jsonify({'summary': "Unexpected error occurred during summarization."})

@app.route('/ask', methods=['POST'])
def ask_pdf():
    global pdf_text_cache
    question = request.json.get('question', '')

    if not pdf_text_cache.strip():
        return jsonify({'answer': "Please upload a PDF first."})

    payload = {
        "inputs": {
            "question": question,
            "context": pdf_text_cache[:2000]  # Limit context for speed
        }
    }

    try:
        response = requests.post(QNA_API, headers=headers, json=payload)
        response.raise_for_status()
        answer = response.json().get('answer', "Sorry, I couldn’t find an answer.")
        return jsonify({'answer': answer})

    except requests.exceptions.HTTPError as e:
        print("🛑 QnA API HTTP error:", response.status_code)
        print("🔍 Details:", response.text)
        return jsonify({'answer': f"Error processing your question: {response.status_code}"})
    except Exception as e:
        print("🔥 Unexpected QnA error:", str(e))
        return jsonify({'answer': "Unexpected error occurred while answering your question."})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
