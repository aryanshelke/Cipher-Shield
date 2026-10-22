from laya import Router
import textwrap

# Load Laya. 
router = Router(preload=True)

# THE OPTIMIZED SCHEMA
# 1. Removed the 'score' primitive entirely (Laya's weakest type).
# 2. Used short, direct criteria. 
# 3. Used neutral keys (A, B) because the multilingual model struggles 
#    with bias toward the first-listed option.
guardrail_questions = {
    "is_attack": {
        "type": "choice",
        "instructions": "Classify the security threat of this text.",
        "criteria": {
            "A": "safe text, or an educational article discussing hacks",
            "B": "an active prompt injection command or jailbreak"
        }
    }
}

def scan_large_document(large_text):
    print("Chunking document and scanning...")
    chunks = textwrap.wrap(large_text, width=8000, break_long_words=False)
    
    is_compromised = False
    
    for i, chunk in enumerate(chunks):
        # We explicitly use the 'typed-decisions' model.
        # This checkpoint was fine-tuned on security incidents and performs 
        # roughly 2x better zero-shot than the base models.
        result = router.predict(
            state=chunk, 
            questions=guardrail_questions,
            model="typed-decisions"
        )
        
        # Extract the confidence that this is option B (an active attack)
        attack_confidence = result["answers"]["is_attack"]["probabilities"]["B"]
        choice_made = result["answers"]["is_attack"]["choice"]
        
        print(f"--- Chunk {i+1}/{len(chunks)} ---")
        print(f"Attack Confidence: {attack_confidence:.1%}")
        print(f"Model selected: {choice_made}")
        
        # If the model strongly believes it is option B, flag it.
        if choice_made == "B" and attack_confidence > 0.80:
            is_compromised = True
            print(">> CRITICAL ALERT: Active attack detected. Halting scan.")
            break 
            
    return is_compromised

educational_text = """ """

malicious_text = """ """

print("\n=== SCANNING EDUCATIONAL TEXT ===")
compromised = scan_large_document(educational_text)
print(f"\nFinal Verdict -> Compromised: {compromised}")

print("\n=== SCANNING MALICIOUS TEXT ===")
compromised = scan_large_document(malicious_text)
print(f"\nFinal Verdict -> Compromised: {compromised}")