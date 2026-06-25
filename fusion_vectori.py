import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import warnings

# Importam metricile
from torchmetrics import AUROC, Accuracy, Precision, Recall

warnings.filterwarnings("ignore")

# 1 CONFIGURARE PARAMETRI
CSV_PATH = "mimic_complete_master_dataset.csv"
CNN_EMBEDDINGS_PATH = "cnn_embeddings_dict.pt"
NLP_EMBEDDINGS_PATH = "nlp_embeddings_dict.pt"

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_COLS = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion',
    'Fracture', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Other'
]
NUM_CLASSES = len(LABEL_COLS)


# 2 DATASET PENTRU FUZIUNE (OFFLINE)
class MultimodalFusionDataset(Dataset):
    def __init__(self, df, cnn_dict, nlp_dict, label_cols):
        self.valid_data = []
        for _, row in df.iterrows():
            s_id = str(row['study_id'])
            # Verificam daca pacientul are ambele tipuri de vectori salvati
            if s_id in cnn_dict and s_id in nlp_dict:
                self.valid_data.append({
                    'cnn_feat': cnn_dict[s_id]['image_features_1024'],
                    'nlp_feat': nlp_dict[s_id]['text_features_768'],
                    'labels': row[label_cols].values.astype(np.float32)
                })

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        cnn_feat = torch.tensor(item['cnn_feat'], dtype=torch.float32)
        nlp_feat = torch.tensor(item['nlp_feat'], dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.float32)
        return cnn_feat, nlp_feat, labels


# 3 ARHITECTURA MLP DE FUZIUNE
class FusionMLP(nn.Module):
    def __init__(self, cnn_dim=1024, nlp_dim=768, hidden_dim=512, num_classes=13, drop_rate=0.4):
        super(FusionMLP, self).__init__()

        input_dim = cnn_dim + nlp_dim  # 1792

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_rate),

            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, cnn_features, nlp_features):
        # Concatenam vectorii
        fused_vector = torch.cat((cnn_features, nlp_features), dim=1)
        logits = self.mlp(fused_vector)
        return logits


class MaskedBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        mask = ~torch.isnan(targets)
        safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
        loss = self.bce(logits, safe_targets)
        loss = loss * mask.float()
        return loss.sum() / (mask.float().sum() + 1e-8)


# 4 EXECUTIE SI ANTRENARE
if __name__ == '__main__':
    print(f"Folosim {DEVICE}")
    print("-----Incarcare Master CSV si Dictionare de Vectori-----")
    df_master = pd.read_csv(CSV_PATH)

    # Incarcam vectorii gata extrasi din CNN si BERT
    cnn_embeddings = torch.load(CNN_EMBEDDINGS_PATH, map_location='cpu', weights_only=False)
    nlp_embeddings = torch.load(NLP_EMBEDDINGS_PATH, map_location='cpu', weights_only=False)

    df_train = df_master[df_master['split'] == 'train']
    df_val = df_master[df_master['split'] == 'val']

    print("--------Pregatire Dataset-uri de Fuziune------")
    train_dataset = MultimodalFusionDataset(df_train, cnn_embeddings, nlp_embeddings, LABEL_COLS)
    val_dataset = MultimodalFusionDataset(df_val, cnn_embeddings, nlp_embeddings, LABEL_COLS)

    print(f"  -> Rapoarte pentru Train: {len(train_dataset)}")
    print(f"  -> Rapoarte pentru Validare: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = FusionMLP(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = MaskedBCE()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Initializare metrici
    metrics = {
        'AUC': AUROC(task="multilabel", num_labels=NUM_CLASSES, average='macro').to(DEVICE),
        'Acc': Accuracy(task="binary").to(DEVICE),
        'Prec': Precision(task="binary").to(DEVICE),
        'Rec': Recall(task="binary").to(DEVICE)
    }

    best_auc_overall = 0.0

    print("\n------- INCEPE ANTRENAREA MLP ULUI DE FUZIUNE OFFLINE ------")
    for epoch in range(1, EPOCHS + 1):

        # ==================== TRAIN ====================
        model.train()
        train_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoca {epoch}/{EPOCHS} [Train]", ncols=120)
        for cnn_feat, nlp_feat, labels in train_loop:
            cnn_feat, nlp_feat, labels = cnn_feat.to(DEVICE), nlp_feat.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(cnn_feat, nlp_feat)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loop.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # ==================== VALIDARE ====================
        model.eval()
        val_loss = 0.0


        for m in metrics.values(): m.reset()

        val_loop = tqdm(val_loader, desc=f"Epoca {epoch}/{EPOCHS} [Valid]", ncols=120)
        with torch.no_grad():
            for cnn_feat, nlp_feat, labels in val_loop:
                cnn_feat, nlp_feat, labels = cnn_feat.to(DEVICE), nlp_feat.to(DEVICE), labels.to(DEVICE)

                logits = model(cnn_feat, nlp_feat)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                # Calcul metrici (ignorand NaN-urile)
                probs = torch.sigmoid(logits)
                mask = ~torch.isnan(labels)

                metrics['AUC'].update(probs, torch.nan_to_num(labels, nan=0).int())
                if mask.sum() > 0:
                    metrics['Acc'].update(probs[mask], labels[mask].int())
                    metrics['Prec'].update(probs[mask], labels[mask].int())
                    metrics['Rec'].update(probs[mask], labels[mask].int())

                val_loop.set_postfix({'val_loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)

        # Extragem valorile finale ale metricilor pentru aceasta epoca
        v_auc = metrics['AUC'].compute().item()
        v_acc = metrics['Acc'].compute().item()
        v_prec = metrics['Prec'].compute().item()
        v_rec = metrics['Rec'].compute().item()

        print(f" -> Rezultate Epoca {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"    Metrici Validare  : AUC: {v_auc:.4f} | Acc: {v_acc:.4f} | Prec: {v_prec:.4f} | Rec: {v_rec:.4f}")

        # Salvam modelul daca AUC-ul este cel mai bun de pana acum
        if v_auc > best_auc_overall:
            best_auc_overall = v_auc
            torch.save(model.state_dict(), "best_fusion_mlp_offline.pth")
            print(f"  ->>> Model salvat! (Nou  AUC: {best_auc_overall:.4f})")

    print("\n Antrenament Fuziune Finalizat!")