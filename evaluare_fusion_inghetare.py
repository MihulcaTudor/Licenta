import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torchxrayvision as xrv


from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_fscore_support,
    cohen_kappa_score, matthews_corrcoef
)



# 1 CONFIGURARE

MODEL_PATH = "best_end2end_frozen.pth"
MASTER_CSV_PATH = Path("mimic_complete_master_dataset.csv")


MODEL_NAME_NLP = "microsoft/BiomedVLP-CXR-BERT-specialized"
CNN_WEIGHTS = "mimic_cnn_model_best_1.pth"
NLP_WEIGHTS = "best_cxr_bert_model.pth"

BATCH_SIZE = 32
MAX_LENGTH = 256
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_COLS = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion',
    'Fracture', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Other'
]
NUM_CLASSES = len(LABEL_COLS)


# 2 DEFINITIE DATASET & MODEL
class MultimodalEndToEndDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df.dropna(subset=['report_text']).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.image_paths = self.df['image_path'].values
        self.texts = self.df['report_text'].values
        self.labels = self.df[LABEL_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Imagine
        path = self.image_paths[idx]
        try:
            image = np.array(Image.open(path).convert('L'))
        except Exception:
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        # Resize manual / simplu pt evaluare (fara augmentari)
        if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
            img_pil = Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE))
            image = np.array(img_pil)

        image = torch.tensor(image).float()
        if image.ndim == 2: image = image.unsqueeze(0)
        image = image / 255.0 * 2048 - 1024

        # Text
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


class JointMultimodalModel(nn.Module):
    def __init__(self, cnn_weights_path, nlp_weights_path, num_classes=13):
        super(JointMultimodalModel, self).__init__()

        self.cnn = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
        self.cnn.op_threshs = None
        self.cnn.classifier = nn.Linear(1024, num_classes)

        try:
            cnn_state_dict = torch.load(cnn_weights_path, map_location=DEVICE, weights_only=False)
            clean_cnn_dict = {k.replace("densenet.", "").replace("_orig_mod.", ""): v for k, v in
                              cnn_state_dict.items()}
            self.cnn.load_state_dict(clean_cnn_dict, strict=False)
        except:
            pass

        self.cnn.classifier = nn.Identity()

        self.bert = AutoModel.from_pretrained(MODEL_NAME_NLP, trust_remote_code=True)

        try:
            nlp_state_dict = torch.load(nlp_weights_path, map_location=DEVICE, weights_only=True)
            clean_nlp_dict = {}
            for k, v in nlp_state_dict.items():
                if k.startswith("bert."):
                    clean_nlp_dict[k[5:]] = v
            self.bert.load_state_dict(clean_nlp_dict, strict=False)
        except:
            pass

        input_dim = 1024 + 768
        hidden_dim = 512

        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        with torch.no_grad():
            cnn_features = self.cnn(image)
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            nlp_features = bert_outputs.last_hidden_state[:, 0, :]

        fused_vector = torch.cat((cnn_features, nlp_features), dim=1)
        return self.fusion_mlp(fused_vector)


# 3 FUNCTIE PREDICTII
def get_predictions(model, loader):
    all_probs = []
    all_labels = []

    print("------------ Generare predictii ------", end="")
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(DEVICE, non_blocking=True)
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            labels = batch['labels'].numpy()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images, input_ids, attention_mask)
                probs = torch.sigmoid(outputs)

            all_probs.append(probs.float().cpu().numpy())
            all_labels.append(labels)

    print(" Done")
    return np.vstack(all_labels), np.vstack(all_probs)


# 4 main
if __name__ == '__main__':
    print(f"Incarcare date din {MASTER_CSV_PATH}...")
    df_master = pd.read_csv(MASTER_CSV_PATH)

    df_val = df_master[df_master['split'] == 'val'].copy()
    df_test = df_master[df_master['split'] == 'test'].copy()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_NLP, trust_remote_code=True)

    val_dataset = MultimodalEndToEndDataset(df_val, tokenizer, MAX_LENGTH)
    test_dataset = MultimodalEndToEndDataset(df_test, tokenizer, MAX_LENGTH)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- INCARCARE MODEL ---
    print(f"Initializare arhitectura")
    model = JointMultimodalModel(CNN_WEIGHTS, NLP_WEIGHTS, num_classes=NUM_CLASSES).to(DEVICE)

    print(f"Incarcare ponderi finale de fuziune din {MODEL_PATH}")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        print(" -> Model fuziune End-to-End incarcat cu succes!")
    except Exception as e:
        print(f"EROARE la incarcarea modelului: {e}")
        sys.exit()

    model.eval()

    print("\nCalculare praguri optime pe Validare ")
    y_true_val, y_pred_val_probs = get_predictions(model, val_loader)

    # --- Calcul Praguri (F1 Maxim) ---
    best_thresholds = {}
    threshold_data = []

    for i, class_name in enumerate(LABEL_COLS):
        col_true = y_true_val[:, i]
        col_pred = y_pred_val_probs[:, i]

        mask = ~np.isnan(col_true)
        valid_true = col_true[mask]
        valid_pred = col_pred[mask]

        if len(valid_true) == 0 or valid_true.sum() == 0:
            best_thresholds[class_name] = 0.5
            continue

        p, r, th = precision_recall_curve(valid_true, valid_pred)
        numerator = 2 * p * r
        denominator = p + r + 1e-7
        f1 = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

        if len(th) == 0:
            best_thresholds[class_name] = 0.5
        else:
            best_idx = np.argmax(f1[:-1])
            best_thresholds[class_name] = float(th[best_idx])

        threshold_data.append([class_name, best_thresholds[class_name]])

    print(pd.DataFrame(threshold_data, columns=['Patologie', 'Best Threshold']).to_string(index=False))

    # --- Predictii Finale pe Test ---
    print("\nEvaluare pe Test (IGNORAND NaN)")
    y_true_test, y_pred_test_probs = get_predictions(model, test_loader)

    y_pred_test_bin = np.zeros_like(y_pred_test_probs)
    for i, class_name in enumerate(LABEL_COLS):
        # Varianta 1 prag custom
        y_pred_test_bin[:, i] = (y_pred_test_probs[:, i] >= best_thresholds[class_name]).astype(int)
        # VARIANTA 2: Prag Simplu de 0.5
        #  y_pred_test_bin[:, i] = (y_pred_test_probs[:, i] >= 0.5).astype(int)

    # CALCUL METRICI CU FILTRARE (MASKING)
    results_data = []
    confusion_matrices = []

    micro_true = []
    micro_pred = []
    micro_prob = []

    for i, label in enumerate(LABEL_COLS):
        col_true = y_true_test[:, i]
        col_pred = y_pred_test_bin[:, i]
        col_prob = y_pred_test_probs[:, i]

        mask = ~np.isnan(col_true)
        valid_true = col_true[mask]
        valid_pred = col_pred[mask]
        valid_prob = col_prob[mask]

        micro_true.extend(valid_true)
        micro_pred.extend(valid_pred)
        micro_prob.extend(valid_prob)

        if len(valid_true) > 0:
            labels_present = [0, 1]
            cm = confusion_matrix(valid_true, valid_pred, labels=labels_present)
            tn, fp, fn, tp = cm.ravel()
            confusion_matrices.append(cm)

            p, r, f1, _ = precision_recall_fscore_support(valid_true, valid_pred, average='binary', zero_division=0)

            try:
                auc = roc_auc_score(valid_true, valid_prob)
            except:
                auc = 0.5
            try:
                pr_auc = average_precision_score(valid_true, valid_prob)
            except:
                pr_auc = 0.0

            kappa = cohen_kappa_score(valid_true, valid_pred)
            mcc = matthews_corrcoef(valid_true, valid_pred)

            if np.isnan(kappa): kappa = 0
            if np.isnan(mcc): mcc = 0
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            p, r, f1, auc, pr_auc, kappa, mcc = 0, 0, 0, 0, 0, 0, 0
            confusion_matrices.append(np.zeros((2, 2)))

        results_data.append({
            'Patologie': label, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
            'Precision': p, 'Recall': r, 'F1-Score': f1,
            'ROC-AUC': auc, 'PR-AUC': pr_auc, 'Kappa': kappa, 'MCC': mcc
        })

    df_results = pd.DataFrame(results_data)

    # --- MACRO / MICRO AVERAGES --
    macro_avg = df_results[['Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC', 'Kappa', 'MCC']].mean()
    macro_row = {
        'Patologie': 'MACRO AVERAGE', 'TN': '-', 'FP': '-', 'FN': '-', 'TP': '-',
        'Precision': macro_avg['Precision'], 'Recall': macro_avg['Recall'], 'F1-Score': macro_avg['F1-Score'],
        'ROC-AUC': macro_avg['ROC-AUC'], 'PR-AUC': macro_avg['PR-AUC'], 'Kappa': macro_avg['Kappa'],
        'MCC': macro_avg['MCC']
    }

    micro_true, micro_pred, micro_prob = np.array(micro_true), np.array(micro_pred), np.array(micro_prob)
    p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(micro_true, micro_pred, average='binary', zero_division=0)
    try:
        auc_mic = roc_auc_score(micro_true, micro_prob)
    except:
        auc_mic = 0.5
    try:
        pr_mic = average_precision_score(micro_true, micro_prob)
    except:
        pr_mic = 0
    kappa_mic = cohen_kappa_score(micro_true, micro_pred)
    mcc_mic = matthews_corrcoef(micro_true, micro_pred)

    micro_row = {
        'Patologie': 'MICRO AVERAGE', 'TN': '-', 'FP': '-', 'FN': '-', 'TP': '-',
        'Precision': p_mic, 'Recall': r_mic, 'F1-Score': f1_mic,
        'ROC-AUC': auc_mic, 'PR-AUC': pr_mic, 'Kappa': kappa_mic, 'MCC': mcc_mic
    }

    df_final = pd.concat([df_results, pd.DataFrame([macro_row, micro_row])], ignore_index=True)

    print("\n" + "=" * 160)
    print(f"{'TABEL FINAL: RAPORT FUZIUNE END-TO-END PE SET TEST':^160}")
    print("=" * 160)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(df_final.to_string(index=False, float_format="%.4f"))
    print("=" * 160)


    df_final.to_csv("rezultate_fuziune_end2end.csv", index=False)
    print(" Raport CSV salvat: rezultate_fuziune_end2end.csv")

    # VIZUALIZARE HEATMAP GRID
    cols = 4
    rows = (len(LABEL_COLS) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.ravel()

    for i, label in enumerate(LABEL_COLS):
        cm = confusion_matrices[i]
        group_names = ['TN', 'FP', 'FN', 'TP']
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]

        total_valid = np.sum(cm)
        if total_valid > 0:
            group_percentages = ["{0:.2%}".format(value / total_valid) for value in cm.flatten()]
        else:
            group_percentages = ["0%", "0%", "0%", "0%"]

        labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)

        sns.heatmap(cm, annot=labels, fmt='', cmap='Purples', cbar=False, ax=axes[i],
                    annot_kws={"size": 11, "weight": "bold"})

        axes[i].set_title(f"{label}", fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Prediction (Fusion)')
        axes[i].set_ylabel('True Label')
        axes[i].set_xticklabels(['Neg', 'Pos'])
        axes[i].set_yticklabels(['Neg', 'Pos'])

    for j in range(len(LABEL_COLS), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle(f"Confusion Matrix - Fusion Image + Text", fontsize=20, fontweight='bold')


    plt.savefig("grid_matrici_fuziune_end2end.png", dpi=300)
    print(" Imagine salvata: grid_matrici_fuziune_end2end.png")
    plt.show()