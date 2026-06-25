import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from PIL import Image
from tqdm import tqdm
import torchxrayvision as xrv
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import warnings

# Importam metricile
from torchmetrics import AUROC, Accuracy, Precision, Recall


# 1 CONFIGURARE
CSV_PATH = "mimic_complete_master_dataset.csv"
MODEL_NAME_NLP = "microsoft/BiomedVLP-CXR-BERT-specialized"


CNN_WEIGHTS = "mimic_cnn_model_best_1.pth"
NLP_WEIGHTS = "best_cxr_bert_model_3.pth"

BATCH_SIZE = 32
MAX_LENGTH = 256
IMAGE_SIZE = 224
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_COLS = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion',
    'Fracture', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Other'
]
NUM_CLASSES = len(LABEL_COLS)


# 2 DATASET END-TO-END (incarcam imagine si iext simultan)
class MultimodalEndToEndDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, transform=None):
        # Eliminam randurile fara text
        self.df = df.dropna(subset=['report_text']).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

        self.image_paths = self.df['image_path'].values
        self.texts = self.df['report_text'].values
        self.labels = self.df[LABEL_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # PARTEA DE IMAGINE
        path = self.image_paths[idx]
        try:
            image = np.array(Image.open(path).convert('L'))
        except Exception:
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        if self.transform:
            image = self.transform(image=image)['image']
        elif not isinstance(image, torch.Tensor):
            image = torch.tensor(image)

        image = image.float()
        if image.ndim == 2: image = image.unsqueeze(0)
        image = image / 255.0 * 2048 - 1024

        # PARTEA DE TEXT
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


# 3 ARHITECTURA JOINT-FUSION CU BACKBONE-URI INGHETATE
class JointMultimodalModel(nn.Module):
    def __init__(self, cnn_weights_path, nlp_weights_path, num_classes=13):
        super(JointMultimodalModel, self).__init__()

        print("\n-> Incarcare componenta CNN----")
        self.cnn = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
        self.cnn.op_threshs = None

        # 1 Ajustam clasificatorul la 13 clase pt a putea incarca greutatile
        self.cnn.classifier = nn.Linear(1024, num_classes)

        # 2 Incarcam greutatile salvate curatand cheile
        cnn_state_dict = torch.load(cnn_weights_path, map_location=DEVICE,
                                    weights_only=False)
        clean_cnn_dict = {k.replace("densenet.", "").replace("_orig_mod.", ""): v for k, v in cnn_state_dict.items()}
        self.cnn.load_state_dict(clean_cnn_dict, strict=False)

        # 3 ACUM taiem clasificatorul de tot, ca sa ne scoata doar vectorul de 1024 pentru fuziune
        self.cnn.classifier = nn.Identity()

        print("-> Incarcare componenta NLP (BERT)----")

        self.bert = AutoModel.from_pretrained(MODEL_NAME_NLP, trust_remote_code=True)

        nlp_state_dict = torch.load(nlp_weights_path, map_location=DEVICE, weights_only=True)

        # Eliminam exact un singur nivel de "bert." de la inceputul fiecarei chei
        clean_nlp_dict = {}
        for k, v in nlp_state_dict.items():
            if k.startswith("bert."):
                # Taiem primele 5 caractere ("bert.")
                new_key = k[5:]
                clean_nlp_dict[new_key] = v

        # Incarcam dict-ul curatat
        self.bert.load_state_dict(clean_nlp_dict, strict=False)

        # INGHETARE STRATURI
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.bert.parameters():
            param.requires_grad = False



        #  CLASIFICATORUL FINAL DE FUZIUNE - MLP
        input_dim = 1024 + 768  # 1792
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

        # Doar MLP-ul primeste gradienti
        fused_vector = torch.cat((cnn_features, nlp_features), dim=1)
        logits = self.fusion_mlp(fused_vector)
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
    print(f" Folosim {DEVICE}")
    df_master = pd.read_csv(CSV_PATH)

    df_train = df_master[df_master['split'] == 'train']
    df_val = df_master[df_master['split'] == 'val']

    # Augmentari DOAR pentru imagini
    train_aug = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        ToTensorV2(),
    ])
    val_aug = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), ToTensorV2()])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_NLP, trust_remote_code=True)

    print("------Pregatire Dataset-uri End-to-End---")
    train_dataset = MultimodalEndToEndDataset(df_train, tokenizer, MAX_LENGTH, transform=train_aug)
    val_dataset = MultimodalEndToEndDataset(df_val, tokenizer, MAX_LENGTH, transform=val_aug)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = JointMultimodalModel(CNN_WEIGHTS, NLP_WEIGHTS, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = MaskedBCE()

    # Optimizatorul primeste doar parametrii de la MLP , restul inghetati
    optimizer = optim.AdamW(model.fusion_mlp.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    # Initializare metrici
    metrics = {
        'AUC': AUROC(task="multilabel", num_labels=NUM_CLASSES, average='macro').to(DEVICE),
        'Acc': Accuracy(task="binary").to(DEVICE),
        'Prec': Precision(task="binary").to(DEVICE),
        'Rec': Recall(task="binary").to(DEVICE)
    }

    best_auc_overall = 0.0

    print("\n-------- INCEPE ANTRENAREA END-TO-END  ---------")
    for epoch in range(1, EPOCHS + 1):

        # ==================== TRAIN ====================
        model.train()
        train_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoca {epoch}/{EPOCHS} [Train]", ncols=120)
        for batch in train_loop:
            images = batch['image'].to(DEVICE, non_blocking=True)
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            labels = batch['labels'].to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(images, input_ids, attention_mask)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_loop.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # ================= VALIDARE ================
        model.eval()
        val_loss = 0.0
        for m in metrics.values(): m.reset()

        val_loop = tqdm(val_loader, desc=f"Epoca {epoch}/{EPOCHS} [Valid]", ncols=120)
        with torch.no_grad():
            for batch in val_loop:
                images = batch['image'].to(DEVICE, non_blocking=True)
                input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
                attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
                labels = batch['labels'].to(DEVICE, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(images, input_ids, attention_mask)
                    loss = criterion(logits, labels)

                val_loss += loss.item()

                # Calcul metrici
                probs = torch.sigmoid(logits)
                mask = ~torch.isnan(labels)

                metrics['AUC'].update(probs, torch.nan_to_num(labels, nan=0).int())
                if mask.sum() > 0:
                    metrics['Acc'].update(probs[mask], labels[mask].int())
                    metrics['Prec'].update(probs[mask], labels[mask].int())
                    metrics['Rec'].update(probs[mask], labels[mask].int())

                val_loop.set_postfix({'val_loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)

        v_auc = metrics['AUC'].compute().item()
        v_acc = metrics['Acc'].compute().item()
        v_prec = metrics['Prec'].compute().item()
        v_rec = metrics['Rec'].compute().item()

        print(f" -> Rezultate Epoca {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"    Metrici Validare  : AUC: {v_auc:.4f} | Acc: {v_acc:.4f} | Prec: {v_prec:.4f} | Rec: {v_rec:.4f}")

        if v_auc > best_auc_overall:
            best_auc_overall = v_auc
            torch.save(model.state_dict(), "best_end2end_frozen.pth")
            print(f"  >>> Model salvat! (Nou AUC: {best_auc_overall:.4f})")

    print("\n Antrenament End-to-End Finalizat!")