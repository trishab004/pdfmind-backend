# 🧠 PDFMind.AI - Backend

📡 *Flask-powered backend API for PDFMind.AI — Your AI PDF Assistant*

This is the backend of **PDFMind.AI**, a smart AI-driven platform that summarizes uploaded PDFs and enables users to chat with them. The backend uses Flask and integrates with Hugging Face's transformer models for summarization and question answering.

Deployed seamlessly via [Render](https://render.com), this service powers the frontend app hosted on GitHub Pages.

---

## 🚀 Live Backend URL

🌐 [https://pdfmind-backend-fy4f.onrender.com](https://pdfmind-backend-fy4f.onrender.com)

> Note: You won't see a home page here. The backend only responds to `/summarize` and `/ask` POST requests.

---

## ⚙️ Features

🔁 Accepts PDF file uploads  
🧠 Extracts text from PDFs using PyMuPDF  
📝 Summarizes text using Hugging Face’s `distilbart-cnn-12-6` model  
❓ Answers user questions using `distilbert-base-cased-distilled-squad` model  
💬 Caches PDF context for continuous Q&A  
🛡️ Uses environment variable for Hugging Face API token  
🧑‍💻 Deployed using Render

---

## 📂 API Endpoints

### `POST /summarize`

Accepts a PDF file and returns an AI-generated summary.

#### Request
- `Content-Type: multipart/form-data`
- `file`: PDF file

#### Response
```json
{
  "summary": "This is the summary of the uploaded document."
}
