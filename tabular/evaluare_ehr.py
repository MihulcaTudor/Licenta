import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, \
    average_precision_score
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIGURARE
# ============================================================
CSV_PATH = "cehr_bert_ready_to_train.csv"
MODEL_WEIGHTS = "best_cher_bert_tabular.pth"  # Actualizat la numele default salvat din scriptul de train
MAX_SEQ_LEN = 256
BATCH_SIZE = 32

LABEL_COLS = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion',
    'Fracture', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Other'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. DATASET ȘI VOCABULAR
# ============================================================
class EHRDataset(Dataset):
    def __init__(self, df, vocab, max_len=256):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        concepts = str(row['concept_seq']).split(',')
        times = [float(x) for x in str(row['time_seq']).split(',')]
        types = [int(x) for x in str(row['type_seq']).split(',')]
        positions = [int(x) for x in str(row['position_seq']).split(',')]

        concepts = concepts[:self.max_len]
        times = times[:self.max_len]
        types = types[:self.max_len]
        positions = positions[:self.max_len]

        concept_ids = [self.vocab.get(c, self.vocab['[UNK]']) for c in concepts]
        pad_len = self.max_len - len(concept_ids)
        concept_ids += [0] * pad_len
        times += [0.0] * pad_len
        types += [0] * pad_len
        positions += [0] * pad_len

        attention_mask = [1] * (self.max_len - pad_len) + [0] * pad_len
        labels = row[LABEL_COLS].values.astype(np.float32)

        return {
            'concept_ids': torch.tensor(concept_ids, dtype=torch.long),
            'type_ids': torch.tensor(types, dtype=torch.long),
            'time_vals': torch.tensor(times, dtype=torch.float),
            'pos_ids': torch.tensor(positions, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float)
        }


def build_vocab(df):
    unique_tokens = set(["[PAD]", "[UNK]", "[CLS]", "[VS]", "[VE]", "[REG]"])
    for seq in df['concept_seq']:
        for token in str(seq).split(','):
            unique_tokens.add(token.strip())
    return {token: idx for idx, token in enumerate(sorted(unique_tokens))}


class Time2Vec(nn.Module):
    def __init__(self, time_emb_size=32):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.b0 = nn.Parameter(torch.randn(1, 1))
        self.w = nn.Parameter(torch.randn(1, time_emb_size - 1))
        self.b = nn.Parameter(torch.randn(1, time_emb_size - 1))

    def forward(self, tau):
        tau = tau.unsqueeze(-1)
        v1 = tau * self.w0 + self.b0
        v2 = torch.sin(tau * self.w + self.b)
        return torch.cat([v1, v2], dim=-1)


# --- CLASĂ ACTUALIZATĂ SĂ CORESPUNDĂ CU CEA DE ANTRENARE ---
class CEHRBERTModel(nn.Module):
    def __init__(self, vocab_size, num_classes=13):
        super().__init__()
        d_model = 768
        self.concept_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(10, d_model, padding_idx=0)  # MĂRIT LA 10
        self.pos_emb = nn.Embedding(512, d_model)  # MĂRIT LA 512
        self.time2vec = Time2Vec(time_emb_size=32)
        self.time_proj = nn.Linear(32, d_model)
        self.LayerNorm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(0.1)

        config = BertConfig(
            hidden_size=d_model, num_hidden_layers=6, num_attention_heads=12,
            intermediate_size=3072, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
            attn_implementation="eager"  # ADAUGAT PENTRU A EVITA WARNING
        )
        self.encoder = BertEncoder(config)
        self.pooler = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.classifier_dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(d_model, num_classes)

    def init_classifier_bias(self, pos_counts, neg_counts):
        # O păstrăm aici chiar dacă nu o apelăm direct în evaluare,
        # pentru ca structura clasei să fie identică.
        pass

    def forward(self, concept_ids, type_ids, time_vals, pos_ids, attention_mask):
        e_c = self.concept_emb(concept_ids)
        e_ty = self.type_emb(type_ids)
        e_p = self.pos_emb(pos_ids)
        e_ti = self.time_proj(self.time2vec(time_vals))

        x = e_c + e_ty + e_p + e_ti
        x = self.LayerNorm(x)
        x = self.embedding_dropout(x)

        ext_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        ext_mask = ext_mask.to(dtype=next(self.parameters()).dtype)
        ext_mask = (1.0 - ext_mask) * -10000.0

        encoder_outputs = self.encoder(x, attention_mask=ext_mask)
        cls_output = encoder_outputs[0][:, 0, :]
        pooled_output = self.pooler(cls_output)
        logits = self.classifier(self.classifier_dropout(pooled_output))
        return logits


# ============================================================
# 3. EVALUARE ȘI MATRICE DE CONFUZIE
# ============================================================
def evaluate_model():
    print(f"--- Rulăm evaluarea pe: {device} ---")
    df = pd.read_csv(CSV_PATH)
    vocab = build_vocab(df)
    vocab_size = len(vocab)

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    print(f"Testăm pe {len(test_df)} pacienți nevăzuți de model...")

    test_loader = DataLoader(EHRDataset(test_df, vocab, MAX_SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False)

    model = CEHRBERTModel(vocab_size=vocab_size, num_classes=13).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        print("Greutățile modelului au fost încărcate cu succes!")
    except Exception as e:
        print(f"Eroare la încărcare: {e}")
        return

    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for batch in test_loader:
            c_ids = batch['concept_ids'].to(device)
            ty_ids = batch['type_ids'].to(device)
            tm_vals = batch['time_vals'].to(device)
            p_ids = batch['pos_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(c_ids, ty_ids, tm_vals, p_ids, mask)
            preds = torch.sigmoid(logits)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    # Binarizăm predicțiile la pragul standard de 0.5
    binary_preds = (all_preds > 0.5).astype(int)

    results = []

    # --- PREGĂTIRE GRAFIC (Grid 4x4 pentru 13 clase) ---
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18, 18))
    fig.suptitle('Matrici de Confuzie pe Patologii (CEHR-BERT pe Urgențe)', fontsize=20, y=0.92)
    axes = axes.flatten()

    for i, class_name in enumerate(LABEL_COLS):
        y_true = all_labels[:, i]
        y_pred_prob = all_preds[:, i]
        y_pred_bin = binary_preds[:, i]

        # Calcul metrici textuale
        try:
            auc = roc_auc_score(y_true, y_pred_prob)
            # PR-AUC este mult mai relevantă pentru clase rare
            pr_auc = average_precision_score(y_true, y_pred_prob)
        except ValueError:
            auc = np.nan
            pr_auc = np.nan

        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        prec = precision_score(y_true, y_pred_bin, zero_division=0)
        rec = recall_score(y_true, y_pred_bin, zero_division=0)

        results.append({
            'Patologie': class_name,
            'ROC-AUC': auc,
            'PR-AUC': pr_auc,  # Metrica nouă adăugată
            'F1': f1,
            'Precision': prec,
            'Recall': rec
        })

        # --- CALCUL ȘI PLOTARE MATRICE DE CONFUZIE ---
        cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        annot_text = np.array([[f"TN\n{tn}", f"FP\n{fp}"],
                               [f"FN\n{fn}", f"TP\n{tp}"]])

        ax = axes[i]
        sns.heatmap(cm, annot=annot_text, fmt='', cmap='Blues', cbar=False,
                    xticklabels=['Negativ (0)', 'Pozitiv (1)'],
                    yticklabels=['Negativ (0)', 'Pozitiv (1)'], ax=ax,
                    annot_kws={"size": 12, "weight": "bold"})

        ax.set_title(f"{class_name}\n(AUC: {auc:.2f})", fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicția Modelului')
        ax.set_ylabel('Realitatea (Ground Truth)')

    # Ascundem ultimele 3 grafice goale
    for j in range(13, 16):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # Salvare grafic și tabel
    grafic_filename = "../matrici_confuzie_patologii.png"
    plt.savefig(grafic_filename, dpi=300, bbox_inches='tight')

    results_df = pd.DataFrame(results)
    tabel_filename = "../rezultate_evaluare_tabular.csv"
    results_df.to_csv(tabel_filename, index=False)

    print("\n" + "=" * 80)
    print("REZULTATE METRICI")
    print("=" * 80)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[SUCCES] Graficul a fost salvat ca: {grafic_filename}")


if __name__ == "__main__":
    evaluate_model()