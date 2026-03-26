import os
import json
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from scoring import calculate_credibility_score
from integrations.twilio_whatsapp import handle_whatsapp_message

load_dotenv()
app = Flask(__name__)
CORS(app)


@app.route('/api/score', methods=['POST'])
def score_content():
    """
    POST /api/score
    Supports text and image (base64) analysis.

    Body (JSON):
    {
      "text": "...",            // for text analysis
      "type": "text",           // "text" or "image"
      "image": "<base64>",      // for image analysis
      "mime_type": "image/jpeg" // optional, default image/jpeg
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        content_type = data.get('type', 'text')

        # ── Image analysis ──────────────────────────────────────────────────
        if content_type == 'image':
            image_b64 = data.get('image', '').strip()
            if not image_b64:
                return jsonify({"error": "image field (base64) is required for type=image"}), 400
            mime = data.get('mime_type', 'image/jpeg')
            result = calculate_credibility_score(text='', image_data=image_b64, image_mime=mime)
            return jsonify(result), 200

        # ── Text analysis ───────────────────────────────────────────────────
        text = data.get('text', '').strip()
        if not text:
            return jsonify({"error": "text field is required and cannot be empty"}), 400
        if len(text) > 5000:
            return jsonify({"error": "Text exceeds 5000 characters"}), 400

        result = calculate_credibility_score(text)
        return jsonify(result), 200

    except Exception as e:
        print(f"Error in /api/score: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/api/whatsapp', methods=['POST'])
def whatsapp_webhook():
    """
    Twilio WhatsApp webhook — handles both text and image messages.
    """
    try:
        from twilio.twiml.messaging_response import MessagingResponse
        resp = MessagingResponse()

        form_data = request.form
        incoming_msg  = form_data.get('Body', '').strip()
        num_media     = int(form_data.get('NumMedia', 0))
        media_url     = form_data.get('MediaUrl0', '')
        media_type    = form_data.get('MediaContentType0', 'image/jpeg')

        print(f"📥 RECEIVED: {incoming_msg[:50]}...", flush=True)

        if num_media > 0 and media_url:
            response_text = handle_whatsapp_message(
                text=incoming_msg or '[Image sent]',
                media_url=media_url,
                media_type=media_type
            )
        elif incoming_msg or num_media == 0:
            response_text = handle_whatsapp_message(text=incoming_msg)
        else:
            response_text = "👋 Hi! Send me something to check!"

        print(f"📤 REPLYING: {response_text[:60]}...", flush=True)

        resp.message(response_text)
        return str(resp), 200, {'Content-Type': 'text/xml'}



    except Exception as e:
        print(f"WhatsApp webhook error: {e}")
        from twilio.twiml.messaging_response import MessagingResponse
        r = MessagingResponse()
        r.message("⚠️ Error processing your request. Please try again.")
        return str(r), 500, {'Content-Type': 'text/xml'}


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "credibility-checker", "ai": "gemini-2.5-flash"}), 200


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "Credibility Checker API (Gemini-powered)",
        "version": "2.5.0",

        "endpoints": {
            "POST /api/score": "Score text or image credibility",
            "POST /api/whatsapp": "WhatsApp webhook (text + images)",
            "GET /health": "Health check"
        }
    }), 200


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)
