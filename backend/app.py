import os
import json
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
    POST endpoint to score content credibility.
    
    Request body:
    {
      "text": "string content to analyze",
      "type": "text|image|video"  (optional, default: "text")
    }
    
    Response:
    {
      "score": 0-100,
      "breakdown": {
        "fact_match": 0-100,
        "language": 0-100,
        "sentiment": 0-100
      },
      "claims": [{"text": "...", "type": "...", "verified": bool}],
      "flags": ["flag1", "flag2"],
      "summary": "string explanation"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400
        
        text = data.get('text', '').strip()
        content_type = data.get('type', 'text')
        
        if not text:
            return jsonify({"error": "Text field is required and cannot be empty"}), 400
        
        if len(text) > 5000:
            return jsonify({"error": "Text exceeds 5000 characters"}), 400
        
        if content_type == 'text':
            result = calculate_credibility_score(text)
            return jsonify(result), 200
        else:
            return jsonify({"error": f"Content type '{content_type}' not yet implemented"}), 501
    
    except Exception as e:
        print(f"Error in /api/score: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/whatsapp', methods=['POST'])
def whatsapp_webhook():
    """
    Twilio WhatsApp webhook endpoint.
    Receives incoming messages and returns credibility score.
    """
    try:
        form_data = request.form
        incoming_msg = form_data.get('Body', '').strip()
        
        if not incoming_msg:
            response_text = "Please send me some text to check!"
        else:
            response_text = handle_whatsapp_message(incoming_msg)
        
        # Build Twilio XML response
        from twilio.twiml.messaging_response import MessagingResponse
        resp = MessagingResponse()
        resp.message(response_text)
        return str(resp), 200
    
    except Exception as e:
        print(f"WhatsApp webhook error: {str(e)}")
        from twilio.twiml.messaging_response import MessagingResponse
        resp = MessagingResponse()
        resp.message("Error processing your request. Please try again.")
        return str(resp), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "service": "credibility-checker"}), 200

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API info"""
    return jsonify({
        "name": "Credibility Checker API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/score": "Score content credibility",
            "POST /api/whatsapp": "WhatsApp webhook",
            "GET /health": "Health check"
        }
    }), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)
