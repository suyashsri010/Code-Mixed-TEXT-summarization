import os
import time
import torch
import warnings
from faster_whisper import WhisperModel
from transformers import MT5ForConditionalGeneration, AutoTokenizer, AutoConfig
from safetensors.torch import load_file

# Suppress noisy warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. LOAD THE MODELS (Repair & Patch Mode)
# ==========================================
print("⏳ [1/2] Loading Local Whisper ASR (CPU)...")
asr_model = WhisperModel("base", device="cpu", compute_type="int8")

print("⏳ [2/2] Repairing & Loading Fine-Tuned mT5 Model (MPS/GPU)...")

# Path to your weights
target_folder = "BhashaBlend_mT5_Base/checkpoint-945"
model_path = os.path.abspath(target_folder)
weights_path = os.path.join(model_path, "model.safetensors")
device = "mps" if torch.backends.mps.is_available() else "cpu"

# STEP A: Load the standard tokenizer and a fresh 'body' for the model
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base", legacy=False)
print("   -> Fetching base dictionary weights...")
summary_model = MT5ForConditionalGeneration.from_pretrained(
    "google/mt5-base",
    local_files_only=True # <--- THE MAGIC SHIELD
)

# STEP B: Load YOUR fine-tuned 'brain' weights
if os.path.exists(weights_path):
    print(f"   -> Patching weights from {target_folder}...")
    fine_tuned_weights = load_file(weights_path)
    # strict=False is the key: it fills in the MISSING embeddings from the base model
    summary_model.load_state_dict(fine_tuned_weights, strict=False)
else:
    print(f"❌ ERROR: Weights not found at {weights_path}")
    exit()

summary_model.to(device)
summary_model.eval() # Set to evaluation mode for consistent output

print("\n✅ SYSTEM REPAIRED & READY!")

# ==========================================
# 2. CORE PROCESSING LOGIC
# ==========================================

def generate_summary(text_input):
    """Core logic to turn any Hinglish text into a summary."""
    print("🧠 Generating Summary...")
    
    # FIX 1: Back to the exact prompt structure your model was trained on
    input_text = "Summarize the following conversation: " + text_input 
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        summary_ids = summary_model.generate(
            inputs["input_ids"],
            num_beams=5,               # Increased for better semantic search
            max_length=60,
            min_length=0,              # FIX 2: Let it be short if the answer is short!
            repetition_penalty=1.5,    # Forces it to use new words rather than repeating the input
            length_penalty=0.8,        # Slightly penalizes rambling
            early_stopping=True,
            decoder_start_token_id=tokenizer.pad_token_id # Safer start token
        )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def run_audio_pipeline():
    """Lists local wav files, transcribes, and then summarizes."""
    audio_files = [f for f in os.listdir('.') if f.endswith('.wav')]
    if not audio_files:
        print("❌ No .wav files found in this folder.")
        return

    print(f"\nAvailable audio files: {audio_files}")
    filename = input("Enter filename or press Enter for default: ").strip()
    if not filename:
        filename = "WhatsApp Audio 2026-04-26 at 21.31.44.wav"

    if not os.path.exists(filename):
        print(f"❌ File '{filename}' not found.")
        return

    print(f"🎙️ Transcribing {filename}...")
    start_time = time.time()
    
    # Transcription with Hinglish nudge
    prompt = "Hello, bhai lab project ka kaam hogaya hain?"
    segments, _ = asr_model.transcribe(filename, initial_prompt=prompt)
    transcript = "".join([s.text for s in segments]).strip()
    
    print(f"\n--- 📝 TRANSCRIPT ---\n{transcript}")
    
    summary = generate_summary(transcript)
    print(f"\n--- ✨ FINAL SUMMARY ---\n{summary}")
    print(f"✅ Total Process Time: {time.time() - start_time:.2f}s")

def run_text_pipeline():
    """Direct text input for instant summarization."""
    print("\n--- ⌨️ TEXT MODE ---")
    user_text = input("Paste your conversation here:\n> ")
    
    if len(user_text.strip()) < 5:
        print("❌ Input too short.")
        return

    start_time = time.time()
    summary = generate_summary(user_text)
    print(f"\n--- ✨ FINAL SUMMARY ---\n{summary}")
    print(f"✅ Completed in {time.time() - start_time:.2f}s")

# ==========================================
# 3. INTERACTIVE MAIN LOOP
# ==========================================

if __name__ == "__main__":
    while True:
        print("\n" + "="*20)
        print("   BHASHABLEND AI ")
        print("="*20)
        print("1. Audio Input (WAV to Summary)")
        print("2. Text Input (Direct Summary)")
        print("3. Exit")
        
        choice = input("\nSelect an option (1/2/3): ").strip()

        if choice == "1":
            run_audio_pipeline()
        elif choice == "2":
            run_text_pipeline()
        elif choice == "3":
            print("👋 System shutting down. Good luck with the demo!")
            break
        else:
            print("❌ Invalid input. Choose 1, 2, or 3.")