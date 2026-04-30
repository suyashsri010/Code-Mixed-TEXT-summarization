from datasets import load_dataset
from openai import OpenAI
import pandas as pd
import json
import os
import concurrent.futures
import threading

# 1. Initialize OpenAI (Add your API key)
client = OpenAI(api_key="YOUR_OPENAI_API_KEY_HERE")

print("Downloading DialogSum dataset...")
dataset = load_dataset("knkarthick/dialogsum")
english_data = dataset["train"] 

num_rows_to_generate = 5000
csv_filename = "hinglish_training_data_5000_.csv"
WORKERS = 10 # This means 10 rows are translating at the exact same time!

# A lock to prevent file corruption when multiple threads try to save at once
csv_lock = threading.Lock()

if not os.path.exists(csv_filename):
    pd.DataFrame(columns=["transcript", "summary"]).to_csv(csv_filename, index=False)
    print(f"Created fresh database: {csv_filename}")

print(f"Starting HYPER-SPEED generation for {num_rows_to_generate} rows using {WORKERS} workers...")

def process_row(i):
    eng_dialogue = english_data[i]["dialogue"]
    eng_summary = english_data[i]["summary"]
    
    prompt = f"""
    Translate the following English dialogue and its summary into natural 'Hinglish' (Hindi words written in English Roman script mixed with English technical terms).
    
    English Dialogue:
    {eng_dialogue}
    
    English Summary:
    {eng_summary}
    
    You MUST respond strictly in valid JSON format with exactly two keys: "transcript" and "summary".
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output pure JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        result_dict = json.loads(response.choices[0].message.content)
        
        new_row = pd.DataFrame([{
            "transcript": result_dict.get("transcript", "").strip(),
            "summary": result_dict.get("summary", "").strip()
        }])
        
        # Lock the file, save the row, unlock the file
        with csv_lock:
            new_row.to_csv(csv_filename, mode='a', header=False, index=False)
            print(f"Row {i+1} secured!")
            
    except Exception as e:
        # If one row hits a rate limit, we just print the error and move on
        print(f"API Error on row {i+1}: {e}")

# The Multithreading Engine
with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
    # This blasts all the rows to OpenAI simultaneously
    executor.map(process_row, range(num_rows_to_generate))

print(f"\n✅ All done! {num_rows_to_generate} rows generated at hyper-speed.")