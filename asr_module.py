from faster_whisper import WhisperModel
import time

print("⏳ Loading Whisper (Base Model)...")
# We switched back to 'base' to match your previous code exactly
model = WhisperModel("base", device="cpu", compute_type="int8")

audio_file = "test_audio.wav" 

print(f"🎙️ Transcribing {audio_file}...\n")
start_time = time.time()

# THE MAGIC FIX: We remove language="hi". 
# We give it an "initial_prompt" written in Hinglish. 
# This tricks Whisper's brain into thinking, "Oh, we are speaking Romanized Hinglish today!"
hinglish_prompt = "Hello, mera code nahi chal raha hai, error aa raha hai."

segments, info = model.transcribe(
    audio_file, 
    initial_prompt=hinglish_prompt,
    condition_on_previous_text=False # Keeps it from hallucinating
)

print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

final_hinglish = ""
for segment in segments:
    final_hinglish += segment.text + " "

print("\n--- FINAL OUTPUT ---")
print(final_hinglish.strip())

print(f"\n✅ Pipeline finished in {time.time() - start_time:.2f} seconds")