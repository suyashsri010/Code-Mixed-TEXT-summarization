import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a clean, academic style for the plots
sns.set_theme(style="whitegrid", palette="muted")

# 1. LOAD THE DATA
print("📊 Loading evaluation data...")
try:
    df = pd.read_csv("evaluation_checkpoint.csv")
except FileNotFoundError:
    print("❌ Could not find evaluation_checkpoint.csv. Make sure it's in the same folder.")
    exit()

# ==========================================
# GRAPH 1: ROUGE vs BERTSCORE (Bar Chart)
# ==========================================
print("🎨 Generating Semantic vs Literal Bar Chart...")
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore']
averages = [df['ROUGE-1'].mean(), df['ROUGE-2'].mean(), df['ROUGE-L'].mean(), df['BERTScore'].mean()]

plt.figure(figsize=(8, 5))
# Create bar plot (Multiply ROUGE by 100 or keep as decimals. Here we keep as decimals for scale)
# Wait, ROUGE is usually 0-1 or 0-100. Assuming your script outputted 0-1 decimals:
ax = sns.barplot(x=metrics, y=averages, palette=['#ff9999', '#ffcc99', '#ffb366', '#66b3ff'])

plt.title('mT5 Summarization Metrics: Keyword Match vs. Semantic Intent', fontsize=14, fontweight='bold')
plt.ylabel('Score (0.0 to 1.0)', fontsize=12)
plt.ylim(0, 1.0)

# Add the exact numbers on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.savefig('documentation_bar_chart.png', dpi=300)
plt.close()

# ==========================================
# GRAPH 2: BERTSCORE DENSITY (The Consistency Curve)
# ==========================================
print("🎨 Generating BERTScore Density Curve...")
plt.figure(figsize=(8, 5))

# Create a smooth density curve
sns.kdeplot(df['BERTScore'], fill=True, color="blue", alpha=0.5, linewidth=2)

plt.title('Model Consistency: Distribution of BERTScores (300 Samples)', fontsize=14, fontweight='bold')
plt.xlabel('BERTScore', fontsize=12)
plt.ylabel('Density (Frequency of Score)', fontsize=12)

# Add a vertical line for the mean
mean_score = df['BERTScore'].mean()
plt.axvline(mean_score, color='red', linestyle='--', label=f'Mean Score: {mean_score:.2f}')
plt.legend()

plt.tight_layout()
plt.savefig('documentation_density_curve.png', dpi=300)
plt.close()

# ==========================================
# GRAPH 3: SYSTEM LATENCY (Stacked Bar)
# ==========================================
print("🎨 Generating Latency Breakdown Chart...")
# Hardcoded averages based on your terminal logs (Whisper ~10s, mT5 ~1.5s)
processes = ['Audio Pipeline (Whisper CPU)', 'NLP Pipeline (mT5 GPU)']
times = [10.5, 1.6] 

plt.figure(figsize=(6, 4))
plt.barh(['End-to-End Latency'], [times[0]], color='#ff9999', label=f'ASR Phase ({times[0]}s)')
plt.barh(['End-to-End Latency'], [times[1]], left=[times[0]], color='#66b3ff', label=f'NLP Phase ({times[1]}s)')

plt.title('Heterogeneous Compute: Inference Latency', fontsize=14, fontweight='bold')
plt.xlabel('Seconds', fontsize=12)
plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig('documentation_latency_chart.png', dpi=300)
plt.close()

print("✅ DONE! Check your folder for 3 new high-quality .png files.")