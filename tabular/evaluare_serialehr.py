import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, \
    average_precision_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIGURĂRI INIȚIALE
# ============================================================
CSV_PATH = "cehr_bert_ready_to_train.csv"
MODEL_WEIGHTS = "best_cehr_bert_tabular.pth"
MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
MAX_SEQ_LEN = 128
BATCH_SIZE = 16

TARGET_PATHOLOGIES = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
    'Support Devices'
]
NUM_LABELS = len(TARGET_PATHOLOGIES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. DEFINIREA CLASEI DATASET
# ============================================================
class EHRTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = self.df['text_sequence'].fillna('').values
        self.labels = self.df[TARGET_PATHOLOGIES].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }


# ============================================================
# 3. FUNCȚIA DE EVALUARE
# ============================================================
def evaluate_model():
    print(f"--- Rulăm evaluarea pe: {device} ---")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=TARGET_PATHOLOGIES)

    # Recreem split-urile pentru a obține exact același set de testare
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    print(f"Set de testare izolat: {len(test_df)} vizite la urgență.")

    print("Reconstruim Vocabularul (Tokenizer)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Extragem tokenii personalizați din setul de antrenare (pentru a avea aceeași dimensiune a vocabularului)
    all_text = " ".join(train_df['text_sequence'].dropna().tolist())
    custom_tokens = list(set(re.findall(r'\[.*?\]', all_text)))
    existing_special_tokens = tokenizer.all_special_tokens
    new_tokens = [tok for tok in custom_tokens if tok not in existing_special_tokens]
    tokenizer.add_tokens(new_tokens)

    print("Încărcăm Modelul și Greutățile (Weights)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )
    model.resize_token_embeddings(len(tokenizer))

    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
        print(">>> Succes: Greutățile modelului au fost încărcate!")
    except Exception as e:
        print(f"Eroare critică la încărcarea modelului: {e}")
        return

    model = model.to(device)
    model.eval()

    test_loader = DataLoader(EHRTextDataset(test_df, tokenizer, MAX_SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False)

    all_labels, all_preds = [], []

    print("Generăm predicțiile pe setul de testare...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Aplicăm Sigmoid pentru a transforma logits în probabilități (0.0 - 1.0)
            preds = torch.sigmoid(outputs.logits)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    # Prag de decizie: dacă probabilitatea > 0.5, prezicem 1 (Patologie prezentă)
    binary_preds = (all_preds > 0.5).astype(int)

    results = []

    # --- PREGĂTIRE GRAFIC (Grid 4x4 pentru 13 clase) ---
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 18))
    fig.suptitle('Matrici de Confuzie - CEHR-BERT (Analiză Semne Vitale și Medicamente)', fontsize=22, y=0.92,
                 fontweight='bold')
    axes = axes.flatten()

    for i, class_name in enumerate(TARGET_PATHOLOGIES):
        y_true = all_labels[:, i]
        y_pred_prob = all_preds[:, i]
        y_pred_bin = binary_preds[:, i]

        # Calcul metrici per clasă
        try:
            auc = roc_auc_score(y_true, y_pred_prob)
            pr_auc = average_precision_score(y_true, y_pred_prob)
        except ValueError:
            auc, pr_auc = 0.0, 0.0  # Se întâmplă dacă o clasă nu are exemple pozitive în setul de test

        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        prec = precision_score(y_true, y_pred_bin, zero_division=0)
        rec = recall_score(y_true, y_pred_bin, zero_division=0)

        results.append({
            'Patologie': class_name,
            'ROC-AUC': auc,
            'PR-AUC': pr_auc,
            'F1-Score': f1,
            'Precision': prec,
            'Recall': rec
        })

        # --- CONSTRUIREA MATRICEI DE CONFUZIE ---
        cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        annot_text = np.array([[f"TN\n{tn}", f"FP\n{fp}"],
                               [f"FN\n{fn}", f"TP\n{tp}"]])

        ax = axes[i]
        sns.heatmap(cm, annot=annot_text, fmt='', cmap='Blues', cbar=False,
                    xticklabels=['Negativ(0)', 'Pozitiv(1)'],
                    yticklabels=['Negativ(0)', 'Pozitiv(1)'], ax=ax,
                    annot_kws={"size": 13, "weight": "bold"})

        ax.set_title(f"{class_name}\nAUC: {auc:.2f} | PR-AUC: {pr_auc:.2f}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicția Modelului')
        ax.set_ylabel('Realitatea (Ground Truth)')

    # Ascundem ultimele 3 grafice goale (din grila de 16 avem doar 13 patologii)
    for j in range(13, 16):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.4, wspace=0.3)

    # Salvare grafic și tabel
    grafic_filename = "../matrici_confuzie_cehr_bert.png"
    plt.savefig(grafic_filename, dpi=300, bbox_inches='tight')

    results_df = pd.DataFrame(results)
    tabel_filename = "../rezultate_evaluare_cehr_bert.csv"
    results_df.to_csv(tabel_filename, index=False)

    # Calculăm media MACRO globală
    macro_metrics = results_df.mean(numeric_only=True)

    print("\n" + "=" * 90)
    print("REZULTATE EVALUARE (PER PATOLOGIE)")
    print("=" * 90)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 90)
    print("METRICI GLOBALE (MACRO AVERAGE)")
    print("=" * 90)
    for metric, value in macro_metrics.items():
        print(f"Macro {metric}: {value:.4f}")

    print(f"\n[SUCCES] Graficul a fost salvat ca: {grafic_filename}")
    print(f"[SUCCES] Tabelul a fost salvat ca: {tabel_filename}")


if __name__ == "__main__":
    evaluate_model()