import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import re
from tqdm import tqdm

# ==========================================
# 1. CONFIGURĂRI INIȚIALE
# ==========================================
DATASET_PATH = 'cehr_bert_ready_to_train.csv'
MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 6
LEARNING_RATE = 2e-5

# Cele 13 patologii (fără 'No Finding')
TARGET_PATHOLOGIES = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
    'Support Devices'
]
NUM_LABELS = len(TARGET_PATHOLOGIES)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Folosim device-ul: {device}")


# ==========================================
# 2. DEFINIREA CLASEI DATASET
# ==========================================
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

        # Apelăm tokenizer-ul direct
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

# ==========================================
# 3. FUNCȚIA DE ANTRENARE
# ==========================================
def main():
    print("1. Încărcarea datelor...")
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=TARGET_PATHOLOGIES)

    # Împărțim datele: 80% Train, 10% Val, 10% Test
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("2. Pregătirea Tokenizer-ului și Modelului...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Identificăm și adăugăm tokenii noștri clinici (ex: [HR_HIGH])
    all_text = " ".join(train_df['text_sequence'].dropna().tolist())
    custom_tokens = list(set(re.findall(r'\[.*?\]', all_text)))

    existing_special_tokens = tokenizer.all_special_tokens
    new_tokens = [tok for tok in custom_tokens if tok not in existing_special_tokens]

    print(f"Adăugăm {len(new_tokens)} tokeni clinici unici în vocabular...")
    tokenizer.add_tokens(new_tokens)

    # Inițializăm modelul
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )

    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    print("3. Crearea DataLoaderelor...")
    train_dataset = EHRTextDataset(train_df, tokenizer, MAX_LEN)
    val_dataset = EHRTextDataset(val_df, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print("4. Începerea antrenamentului...")
    best_val_auc = 0.0

    for epoch in range(EPOCHS):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                preds = torch.sigmoid(outputs.logits).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # Prelucrarea metricilor
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Convertim probabilitățile în 0 sau 1 (Threshold standard de 0.5)
        preds_binary = (all_preds >= 0.5).astype(int)

        try:
            # Metricile STANDARD MACRO pentru Multi-Label
            val_auc = roc_auc_score(all_labels, all_preds, average='macro')
            val_f1 = f1_score(all_labels, preds_binary, average='macro', zero_division=0)
            val_precision = precision_score(all_labels, preds_binary, average='macro', zero_division=0)
            val_recall = recall_score(all_labels, preds_binary, average='macro', zero_division=0)
        except ValueError:
            val_auc, val_f1, val_precision, val_recall = 0.0, 0.0, 0.0, 0.0

        # Afișare rezultate per epocă
        print(f"\n[Rezultate Epoch {epoch + 1}]")
        print(f"Loss: Train = {avg_train_loss:.4f} | Val = {avg_val_loss:.4f}")
        print(
            f"Metrici Val: Macro AUC = {val_auc:.4f} | Macro F1 = {val_f1:.4f} | Macro Precision = {val_precision:.4f} | Macro Recall = {val_recall:.4f}")

        # Salvăm modelul dacă e cel mai bun de până acum
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_cehr_bert_tabular.pth')
            print(">>> [Salvăm Modelul] Noul Best Macro AUC atins!")

    print("\nAntrenament finalizat. Cel mai bun model a fost salvat ca 'best_cehr_bert_tabular.pth'.")


if __name__ == "__main__":
    main()