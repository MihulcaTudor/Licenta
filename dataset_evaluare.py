import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. CONFIGURATION
# ============================================================
CSV_PATH = "mimic_complete_master_dataset.csv"

# Excludem "No Finding" și păstrăm doar cele 13 patologii de interes
LABEL_COLS = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion',
    'Fracture', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Other'
]


def generate_class_distribution():
    print(f"Loading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    # ============================================================
    # 2. COUNT TOTAL POSITIVE CASES (1.0)
    # ============================================================
    print("Calculating class distribution for the entire dataset...")
    disease_counts = {}

    for col in LABEL_COLS:
        if col in df.columns:
            # Numărăm doar cazurile clar pozitive (1.0)
            count = (df[col] == 1.0).sum()
            disease_counts[col] = count
        else:
            print(f"Warning: Column '{col}' not found in the dataset.")

    # Convertim într-o serie Pandas și o sortăm crescător
    counts_series = pd.Series(disease_counts).sort_values(ascending=True)

    # ============================================================
    # 3. PLOTTING
    # ============================================================
    plt.figure(figsize=(12, 8))

    # Setăm un stil curat
    sns.set_style("whitegrid")

    # Folosim exact paleta din imaginea ta (viridis - mov închis spre verde deschis)
    # Este o paletă foarte elegantă și subtilă, standard în publicațiile științifice.
    colors = sns.color_palette("viridis", len(counts_series))

    # Creăm graficul cu bare orizontale, adăugând și un contur subțire elegant
    bars = plt.barh(counts_series.index, counts_series.values, color=colors, edgecolor='black', linewidth=0.8)

    # Adăugăm valorile numerice la capătul fiecărei bare
    max_val = max(counts_series.values)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + (max_val * 0.01),
                 bar.get_y() + bar.get_height() / 2,
                 f'{int(width):,}',
                 va='center', ha='left', fontsize=11, fontweight='bold', color='black')

    # Personalizarea textelor în Engleză
    # plt.title("Total Class Distribution (Positive Cases) Across Entire Dataset", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Number of Positive Cases", fontsize=14, fontweight='bold')
    plt.ylabel("Pathology", fontsize=14, fontweight='bold')

    # Ajustăm dimensiunea fontului pe axe
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Spațiu extra în dreapta pentru ca textele să nu se taie
    plt.xlim(0, max_val * 1.15)

    # ============================================================
    # 4. SAVING
    # ============================================================
    plt.tight_layout()
    save_name = "total_class_distribution_viridis.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"✅ Successfully saved plot as: {save_name}")

    plt.show()


if __name__ == '__main__':
    generate_class_distribution()