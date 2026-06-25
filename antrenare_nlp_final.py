import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# 1 CONFIGURARE
CSV_PATH = "mimic_complete_master_dataset.csv"
MODEL_NAME = "microsoft/BiomedVLP-CXR-BERT-specialized"
MAX_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 6e-6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Folosim device ul: {DEVICE}")

LABEL_COLS = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion',
    'Fracture', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Other'
]
NUM_CLASSES = len(LABEL_COLS)


# 2 DEFINIRE DATASET
class MIMICReportDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = str(self.dataframe.loc[idx, 'report_text'])
        study_id = str(self.dataframe.loc[idx, 'study_id'])  # Adaugat pt identificare

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
            'study_id': study_id,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }


# 3 DEFINIRE ARHITECTURA MODEL SI MASKED BCE
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


class CXRBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(CXRBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extragem token-ul [CLS] (penultimul strat)
        cls_output = outputs.last_hidden_state[:, 0, :]
        dropped_output = self.dropout(cls_output)
        # Ultimul strat (Logits)
        logits = self.classifier(dropped_output)
        return logits

    def extract_features(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

            # PENULTIMUL STRAT (Trasaturile textului)
            # Token-ul [CLS] comprima sensul intregului raport intr-un vector de 768 valori
            penultimate_vector = outputs.last_hidden_state[:, 0, :]

            # ULTIMUL STRAT (Predictiile)
            last_vector = self.classifier(penultimate_vector)

        return penultimate_vector, last_vector


# 4 PREGATIREA DATELOR SI A MODELULUI
df = pd.read_csv(CSV_PATH)

train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']

all_data_loader = DataLoader(
    MIMICReportDataset(df, AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True), MAX_LENGTH),
    batch_size=BATCH_SIZE, shuffle=False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

train_loader = DataLoader(MIMICReportDataset(train_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(MIMICReportDataset(val_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)

model = CXRBERTClassifier(MODEL_NAME, NUM_CLASSES).to(DEVICE)
criterion = MaskedBCE()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5 BUCLA ANTRENARE
if __name__ == '__main__':

    print("\n----- INCEPE EXTRAGEREA TRASATURILOR (EMBEDDINGS) ----")
    model.load_state_dict(torch.load("best_cxr_bert_model_3.pth"))
    model.eval()

    nlp_embeddings = {}
    extract_loop = tqdm(all_data_loader, desc="Extragere Vectori")

    for batch in extract_loop:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        study_ids = batch['study_id']

        # Extragem vectorii folosind noua functie
        vec_768, vec_13 = model.extract_features(input_ids, attention_mask)

        # Mutam pe CPU si convertim in numpy
        vec_768_np = vec_768.cpu().numpy()
        vec_13_np = vec_13.cpu().numpy()

        # Salvam in dictionar penultimul si ultimul strat pt fusion offline
        for i, s_id in enumerate(study_ids):
            nlp_embeddings[s_id] = {
                'text_features_768': vec_768_np[i],  # Penultimul strat
                'text_logits_13': vec_13_np[i]  # Ultimul strat
            }

    output_dict_path = "nlp_embeddings_dict.pt"
    torch.save(nlp_embeddings, output_dict_path)
    print(f"\n Vectorii au fost salvati in '{output_dict_path}'")


    best_val_loss = float('inf')

    print("\n---- INCEPE ANTRENAREA -----")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for batch in loop:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)  # Folosim MaskedBCE

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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_cxr_bert_model_3.pth")
            print("-> Model salvat (Val Loss a scazut!)")

    print("\nAntrenare finalizata!")

