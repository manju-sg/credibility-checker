import os
from scoring import calculate_credibility_score

def handle_whatsapp_message(incoming_text):
    """
    Process WhatsApp incoming message and return credibility assessment.
    
    Args:
        incoming_text: Raw message text from user
    
    Returns:
        Response message to send back to user
    """
    if not incoming_text or len(incoming_text.strip()) < 5:
        return """Hi there! 👋
        
Send me any text, headline, or claim you want me to verify.

I'll analyze it for credibility and let you know if it's likely true, false, or uncertain.

Example: "Send a news headline"
        """
    
    try:
        result = calculate_credibility_score(incoming_text)
        score = result['score']
        breakdown = result['breakdown']
        flags = result['flags']
        claims = result['claims']
        
        # Build response based on score
        if score >= 75:
            emoji = "🟢"
            verdict = "Likely CREDIBLE"
        elif score >= 50:
            emoji = "🟡"
            verdict = "UNCERTAIN"
        else:
            emoji = "🔴"
            verdict = "Likely FALSE"
        
        # Format response
        response = f"{emoji} *{verdict}*\n"
        response += f"Score: {score}/100\n\n"
        
        response += f"📊 *Breakdown:*\n"
        response += f"  • Fact match: {breakdown['fact_match']}%\n"
        response += f"  • Language quality: {breakdown['language']}%\n"
        response += f"  • Sentiment: {breakdown['sentiment']}%\n\n"
        
        if flags:
            response += f"⚠️ *Red Flags:*\n"
            for flag in flags[:2]:
                response += f"  • {flag}\n"
            response += "\n"
        
        if claims:
            response += f"📌 *{len(claims)} claim(s) detected:*\n"
            for i, claim in enumerate(claims[:2], 1):
                response += f"  {i}. {claim['text'][:50]}...\n"
            if len(claims) > 2:
                response += f"  ... and {len(claims)-2} more\n"
        
        return response
    
    except Exception as e:
        print(f"WhatsApp handler error: {str(e)}")
        return "⚠️ Error analyzing text. Please try a different input or shorter text."
