# 🛡️ Veritas AI – Precision Misinformation Detection

![Veritas AI Mockup](https://ik.imagekit.io/manjumindverse/images/veritas_ai_mockup_1774571683493.png)

> **Empowering Truth with Offline-First Intelligence.** Veritas AI is a state-of-the-art misinformation detection platform that uses advanced linguistic pattern matching and sentiment analysis to identify deceptive content, even without an internet connection.

---

## ✨ Key Features

- 🧠 **Offline AI Engine**: Powered by `spaCy` and `TextBlob`, detecting clickbait, bias, and misinformation patterns locally.
- 🖼️ **Multimodal Analysis**: Verify both text claims and image-based misinformation (OCR and visual integrity).
- 📱 **WhatsApp Integration**: A ready-to-use Twilio-powered bot for real-time verification in messaging apps.
- 🎨 **Liquid Glass UI**: A premium, responsive web interface with high-performance animations and glassmorphism.
- 📊 **Detailed Breakdown**: Get granular scores on Sentiment, Language Complexity, Fact-Match proxy, and Source Quality.

---

## 🚀 Technology Stack

### Backend
- **Core**: Python / Flask
- **NLP**: `spaCy` (en_core_web_sm), `TextBlob`
- **Integrations**: Twilio (WhatsApp API)
- **Deployment**: Configured for Render (Procfile, render.yaml included)

### Frontend
- **Design**: Vanilla JS / CSS3 (Liquid Glassmorphism)
- **Typography**: Inter (Google Fonts)
- **Architecture**: Single Page Application (SPA) with dynamic state management.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.9+
- Node.js (for local preview, though frontend is vanilla)

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Environment Variables
Create a `.env` file in the `backend/` directory:
```env
PORT=5000
FLASK_ENV=development
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_PHONE_NUMBER=your_number
```

### 4. Running the App
**Start Backend:**
```bash
python app.py
```
**View Frontend:**
Open `frontend/index.html` in your browser or use a live server.

---

## 📁 Project Architecture

```text
misinformation-detection/
├── backend/               # Flask Application
│   ├── app.py             # Main API Endpoints
│   ├── local_engine.py    # Offline NLP Logic
│   ├── scoring.py         # Advanced Scoring Algorithms
│   └── integrations/      # Twilio & External APIs
├── frontend/              # Web Interface
│   ├── index.html         # Main UI (Glassmorphic)
│   ├── src/               # Component Logic
│   └── styles/            # CSS Design System
└── README.md
```

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Built with ❤️ for the Hackathon.*
