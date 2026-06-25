import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# ============================================================
# 1. CONFIGURARE PARAMETRI (Conform articolului)
# ============================================================
CSV_PATH = "nlp_ed_master_dataset.csv"
MODEL_NAME = "microsoft/BiomedVLP-CXR-BERT-specialized"
MAX_LENGTH = 256  # Conform testelor din articol
BATCH_SIZE = 32
EPOCHS = 8  # Conform articolului
LEARNING_RATE = 6e-6  # Conform articolului
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Folosim device-ul: {DEVICE}")

LABEL_COLS = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion',
    'Fracture', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Other'
]
NUM_CLASSES = len(LABEL_COLS)


# ============================================================
# 2. DEFINIRE DATASET PYTORCH
# ============================================================
class MIMICReportDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = str(self.dataframe.loc[idx, 'report_text'])

        # Folosim apelarea directă a tokenizer-ului (metoda __call__)
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
            # Scoatem dimensiunea extra adăugată de return_tensors='pt'
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }

# ============================================================
# 3. DEFINIRE ARHITECTURĂ MODEL
# ============================================================
class CXRBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(CXRBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extragem token-ul [CLS] care reprezinta tot textul
        cls_output = outputs.last_hidden_state[:, 0, :]
        dropped_output = self.dropout(cls_output)
        logits = self.classifier(dropped_output)
        return logits


# ============================================================
# 4. PREGĂTIREA DATELOR ȘI A MODELULUI
# ============================================================
df = pd.read_csv(CSV_PATH)

# Impartim dataset-ul folosind coloana noastra 'split'
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'validate']

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

train_dataset = MIMICReportDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset = MIMICReportDataset(val_df, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = CXRBERTClassifier(MODEL_NAME, NUM_CLASSES).to(DEVICE)

# Funcția de loss pentru Multi-Label Classification
criterion = nn.BCEWithLogitsLoss()
# Optimizatorul Adam
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)




# ============================================================
# 5. BUCLA DE ANTRENARE (TRAINING LOOP)
# ============================================================
best_val_loss = float('inf')

print("\n--- INCEPE ANTRENAREA ---")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    # Bara de progres pentru antrenare
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
    for batch in loop:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)

    # Validare
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]")
        for batch in val_loop:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"\nRezultate Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

    # Salvăm cel mai bun model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_cxr_bert_model_2.pth")
        print("-> Model salvat (Val Loss a scazut!)")

print("\nAntrenare finalizata!")