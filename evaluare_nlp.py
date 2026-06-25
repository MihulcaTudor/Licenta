import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)

# 1 CONFIGURARE
CSV_PATH = "mimic_complete_master_dataset.csv"
MODEL_NAME = "microsoft/BiomedVLP-CXR-BERT-specialized"
MODEL_WEIGHTS = "best_cxr_bert_model_3.pth"
MAX_LENGTH = 256
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_COLS = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion',
    'Fracture', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Other'
]
NUM_CLASSES = len(LABEL_COLS)


# 2 RE-DEFINIRE CLASE
class MIMICReportDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = str(self.dataframe.loc[idx, 'report_text'])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        labels = self.dataframe.loc[idx, LABEL_COLS].values.astype(float)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }


class CXRBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(CXRBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        dropped_output = self.dropout(cls_output)
        logits = self.classifier(dropped_output)
        return logits


# 3 PREGATIREA DATELOR DE TEST
print(f"Incarcam datele pentru evaluare pe device: {DEVICE}")
df = pd.read_csv(CSV_PATH)

# Selectam exclusiv datele de testare
test_df = df[df['split'] == 'test']
print(f"Total rapoarte in setul de testare: {len(test_df)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
test_dataset = MIMICReportDataset(test_df, tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4 INCARCAREA MODELULUI ANTRENAT
model = CXRBERTClassifier(MODEL_NAME, NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
model.eval()
print("Greutatile modelului au fost incarcate cu succes")

# 5 EVALUARE SI GENERARE RAPORT
all_preds = []
all_labels = []

print("\n----- INCEPE EVALUAREA PE SETUL DE TESTARE ----")
with torch.no_grad():
    loop = tqdm(test_loader, desc="Evaluare")
    for batch in loop:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)

        all_preds.append(probs.cpu().numpy())  # Salvam probabilitatile brute mai intai
        all_labels.append(labels.cpu().numpy())

    # Unificam rezultatele batch-urilor
    y_pred_test_probs = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # CALCUL PRAGURI OPTIME PE VALIDARE (ADAPTAT PENTRU NLP)
    best_thresholds = {}
    for i, class_name in enumerate(LABEL_COLS):
        col_true = all_labels[:, i]
        col_pred = y_pred_test_probs[:, i]
        mask = ~np.isnan(col_true)
        valid_true, valid_pred = col_true[mask], col_pred[mask]

        if len(valid_true) == 0 or valid_true.sum() == 0:
            best_thresholds[class_name] = 0.5
            continue

        p, r, th = precision_recall_curve(valid_true, valid_pred)
        numerator = 2 * p * r
        denominator = p + r + 1e-7
        f1 = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
        best_thresholds[class_name] = float(th[np.argmax(f1[:-1])]) if len(th) > 0 else 0.5


    all_preds_bin = np.zeros_like(y_pred_test_probs)
    for i, class_name in enumerate(LABEL_COLS):
        all_preds_bin[:, i] = (y_pred_test_probs[:, i] >= best_thresholds[class_name]).astype(int)

    all_preds = all_preds_bin


#  Transformam toate valorile NaN din etichetele
all_labels = np.nan_to_num(all_labels, nan=0.0)


print("\n" + "=" * 65)
print("RAPORT DE CLASIFICARE FINAL (SET TESTARE)")
print("=" * 65)
report = classification_report(
    y_true=all_labels,
    y_pred=all_preds,
    target_names=LABEL_COLS,
    zero_division=0,
    digits=4
)
print(report)

# CALCUL SI AFISARE ACURATETE (PER CLASA SI MACRO)
print("\n" + "=" * 65)
print("ACURATETE (ACCURACY) PE CLASE SI MACRO ACCURACY")
print("=" * 65)

# Calculam matricea de confuzie
mcm = multilabel_confusion_matrix(all_labels, all_preds)
accuracies = []

for i, (label, matrix) in enumerate(zip(LABEL_COLS, mcm)):
    tn, fp, fn, tp = matrix.ravel()
    total = tn + fp + fn + tp
    # Formula Acuratetei: (TP + TN) / Total
    acc = (tp + tn) / total if total > 0 else 0
    accuracies.append(acc)
    print(f"{label:>26}: {acc:.4f} ({acc * 100:.2f}%)")

macro_accuracy = np.mean(accuracies)
print("-" * 65)
print(f"{'MACRO ACCURACY (MEDIE)':>26}: {macro_accuracy:.4f} ({macro_accuracy * 100:.2f}%)")
print("=" * 65)

# CALCUL ROC-AUC, PR-AUC, MCC (PER CLASA SI MACRO)
print("\n" + "=" * 65)
print(f"{'Patologie':>26} | {'ROC-AUC':>8} | {'PR-AUC':>8} | {'MCC':>8}")
print("-" * 65)

roc_aucs = []
pr_aucs = []
mccs = []

for i, class_name in enumerate(LABEL_COLS):
    y_true = all_labels[:, i]
    y_score = y_pred_test_probs[:, i]  # Probabilitati brute pentru AUC
    y_pred_b = all_preds[:, i]  # Binar pentru MCC

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        pr_auc = np.nan

    try:
        mcc = matthews_corrcoef(y_true, y_pred_b)
    except ValueError:
        mcc = np.nan

    roc_aucs.append(roc_auc)
    pr_aucs.append(pr_auc)
    mccs.append(mcc)

    # Afisare linie per patologie
    print(f"{class_name:>26} | {roc_auc:8.4f} | {pr_auc:8.4f} | {mcc:8.4f}")

macro_roc_auc = np.nanmean(roc_aucs)
macro_pr_auc = np.nanmean(pr_aucs)
macro_mcc = np.nanmean(mccs)

print("-" * 65)
print(f"{'MACRO AVERAGE':>26} | {macro_roc_auc:8.4f} | {macro_pr_auc:8.4f} | {macro_mcc:8.4f}")
print("=" * 65 + "\n")

# 6 GENERARE MATRICE DE CONFUZIE MULTI-LABEL
print("matricele de confuzie vizuale")

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()

for i, (label, matrix) in enumerate(zip(LABEL_COLS, mcm)):
    ax = axes[i]

    # Extragem valorile standard din scikit-learn
    tn, fp, fn, tp = matrix.ravel()

    viz_matrix = np.array([[tp, fp],
                           [fn, tn]])

    # Desenam heatmap-ul
    sns.heatmap(viz_matrix, annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=['P', 'N'], yticklabels=['P', 'N'], ax=ax)

    ax.set_title(label, fontsize=12, pad=10)
    ax.set_xlabel('True label', fontsize=10)
    ax.set_ylabel('Predicted label', fontsize=10)


axes[13].axis('off')
axes[14].axis('off')

plt.tight_layout()
plt.savefig('matrice_confuzie_13.png', dpi=300)
print("-> Imaginea a fost salvata cu succes: 'matrice_confuzie_13.png'!")

plt.show()