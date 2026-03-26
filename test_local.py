import sys
import os

# Set base path
sys.path.append('C:\\Users\\Manjunath\\OneDrive\\Documents\\Hackathon\\misinformation-detection\\backend')

try:
    from local_engine import engine
    
    test_texts = [
        "Scientists have discovered a miracle cure for all cancers today!",
        "The sky is blue and the sun is a star.",
        "YOU WON'T BELIEVE what this celebrity did at the hidden secret leak!"
    ]
    
    for t in test_texts:
        res = engine.analyze(t)
        print(f"\nText: {t}")
        print(f"Score: {res['score']}")
        print(f"Verdict: {res['verdict']}")
        print(f"Flags: {res['flags']}")

except Exception as e:
    print(f"Test failed: {e}")
