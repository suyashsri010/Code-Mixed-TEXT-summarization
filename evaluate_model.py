import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score
import warnings
from asr_mt5_pipeline import generate_summary

warnings.filterwarnings("ignore")

# 1. LOAD YOUR DATASET
print("⏳ Loading Dataset...")
df = pd.read_csv("hinglish_training_data_5000_.csv") 

# Sample 300 rows so it takes 10 minutes instead of 2 hours
df = df.sample(n=300, random_state=42).reset_index(drop=True)

# 2. GENERATE SUMMARIES
model_outputs = []
print("🧠 Generating Summaries via mT5...")

# We use the 'transcript' column for the input text
for text in tqdm(df['transcript']):
    # Convert to string just in case Pandas loaded a blank row as a mathematical NaN
    safe_text = str(text)
    summary = generate_summary(safe_text) 
    model_outputs.append(summary)

df['model_summary'] = model_outputs

# 3. CALCULATE ROUGE
print("⏳ Calculating ROUGE Scores...")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

rouge1_f1, rouge2_f1, rougel_f1 = [], [], []

for index, row in df.iterrows():
    # Compare the generated summary against your 'summary' column
    scores = scorer.score(str(row['summary']), str(row['model_summary'])) 
    rouge1_f1.append(scores['rouge1'].fmeasure)
    rouge2_f1.append(scores['rouge2'].fmeasure)
    rougel_f1.append(scores['rougeL'].fmeasure)

df['ROUGE-1'] = rouge1_f1
df['ROUGE-2'] = rouge2_f1
df['ROUGE-L'] = rougel_f1

# 4. CALCULATE BERTSCORE
print("⏳ Calculating BERTScore...")
# BERTScore takes the whole list at once. Convert all targets to strings first.
target_summaries = [str(t) for t in df['summary'].tolist()]
P, R, F1 = score(df['model_summary'].tolist(), target_summaries, lang="en", verbose=True)
df['BERTScore'] = F1.numpy()

# 5. PRINT FINAL METRICS
print("\n" + "="*50)
print(" 📊 FINAL SYSTEM AVERAGES (Based on 300 rows)")
print("="*50)
print(f"Average ROUGE-1:   {df['ROUGE-1'].mean():.4f}")
print(f"Average ROUGE-2:   {df['ROUGE-2'].mean():.4f}")
print(f"Average ROUGE-L:   {df['ROUGE-L'].mean():.4f}")
print(f"Average BERTScore: {df['BERTScore'].mean():.4f}")