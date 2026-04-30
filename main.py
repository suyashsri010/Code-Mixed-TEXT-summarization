import torch
import os
import sys
from transformers import AutoTokenizer, MT5ForConditionalGeneration

# --- PATH SETUP ---
base_path = os.path.dirname(os.path.abspath(__file__))
# Priority: Manual path -> Auto-detected path
model_path = os.path.join(base_path, "BhashaBlend_mT5_Base", "checkpoint-630") 

if not os.path.exists(model_path):
    print("Searching for model folder...")
    for root, dirs, files in os.walk("."):
        if "config.json" in files and "model.safetensors" in files:
            model_path = root
            break

print(f"📂 Using Model: {model_path}")

# --- LOAD ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base", use_fast=False)
model = MT5ForConditionalGeneration.from_pretrained(model_path).to(device)
print("✅ BhashaBlend is Ready!")

def generate_summary(text):
    print("⏳ [1/3] Tokenizing...")
    # Using a more natural prompt that DialogSum models usually prefer
    input_text = "Summarize the following conversation: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    print(f"⏳ [2/3] Generating on {device}...")
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=6,               # Higher beams to track multiple characters
            max_length=100,            # Allow a much longer summary
            min_length=45,             # FORCE it to write at least 45 tokens (more info)
            repetition_penalty=2.0,    # Moderate penalty to stop 'Susan Susan Susan' loops
            no_repeat_ngram_size=3,    
            length_penalty=2.0,        # Positive length penalty ENCOURAGES longer outputs
            early_stopping=True,
            do_sample=False            
        )
    
    print("⏳ [3/3] Decoding...")
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- BETTER INPUT LOOP ---
print("\n" + "="*40)
print("INSTRUCTIONS: Paste your text below.")
print("When finished, press ENTER then Ctrl+D (on Mac) to submit.")
print("="*40)

while True:
    print("\nPaste text + Enter + Ctrl+D:")
    try:
        # This captures everything including newlines until you hit Ctrl+D
        user_input = sys.stdin.read() 
        if not user_input.strip():
            break
            
        summary = generate_summary(user_input)
        print(f"\n✨ SUMMARY:\n{summary}")
        print("\n" + "-"*40)
        
        # Reset for next input
        print("\n(To run another, paste again and hit Ctrl+D)")
    except EOFError:
        break
    except KeyboardInterrupt:
        break