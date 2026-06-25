import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torchxrayvision as xrv
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchmetrics import AUROC, Accuracy, Precision, Recall

# ============================================================
# 1. HARDWARE SETUP
# ============================================================
torch.set_float32_matmul_precision('medium')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("EROARE: Nu am detectat placa video!")
    sys.exit()


# ============================================================
# 2. DATASET & NORMALIZARE (Fixat)
# ============================================================
class MIMICDataset(Dataset):
    def __init__(self, df, label_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.paths = self.df['image_path'].values
        self.labels = self.df[label_cols].values.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            # 1. Citim imaginea (0-255 uint8)
            image = np.array(Image.open(path).convert('L'))
        except Exception:
            image = np.zeros((224, 224), dtype=np.uint8)

        # 2. Augmentări
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # 3. Conversie la Tensor Float
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)

        image = image.float()

        # 4. Asigurăm dimensiunea canalului [1, 224, 224]
        if image.ndim == 2:
            image = image.unsqueeze(0)

        # 5. FIX NORMALIZARE: [0, 255] -> [0, 1] -> [-1024, 1024]
        image = image / 255.0
        image = image * 2048 - 1024

        return image, torch.tensor(self.labels[idx])


# ============================================================
# 3. OVERSAMPLING LOGIC (NOU!)
# ============================================================
def make_weighted_sampler(df, label_cols):
    """
    Calculează greutăți pentru fiecare imagine din setul de antrenare
    astfel încât clasele rare să fie selectate mai des.
    """
    print(" -> Calculare Oversampling Weights...")

    # 1. Calculăm frecvența fiecărei clase (ignoram NaN)
    # Convertim NaN la 0 doar pt numărătoare
    temp_labels = df[label_cols].fillna(0).values
    class_counts = np.sum(temp_labels == 1, axis=0)

    # Evităm împărțirea la 0
    class_counts = np.where(class_counts == 0, 1, class_counts)

    # 2. Greutatea clasei = Total / Frecvență (Clasele rare au greutate mare)
    num_samples = len(df)
    class_weights = num_samples / class_counts

    # 3. Atribuim o greutate fiecărei imagini
    # Pentru multilabel, o imagine primește greutatea MAXIMĂ dintre bolile pe care le are.
    # (Dacă are o boală rară și una comună, vrem să fie tratată ca rară).
    sample_weights = []

    for i in range(num_samples):
        labels = temp_labels[i]
        # Găsim indicii unde avem boală (1.0)
        indices = np.where(labels == 1)[0]

        if len(indices) > 0:
            # Luăm greutatea maximă a bolilor prezente
            weight = np.max(class_weights[indices])
        else:
            # Pentru imagini fără nicio boală (No Finding), dăm o greutate mică/medie
            weight = 0.2

        sample_weights.append(weight)

    sample_weights = torch.DoubleTensor(sample_weights)

    # 4. Creăm Sampler-ul
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True  # Permite să extragem aceeași imagine de mai multe ori pe epocă (Dublare!)
    )

    return sampler


# ============================================================
# 4. MODEL (Fixat)
# ============================================================
def get_model(num_classes):
    print("[MODEL] Loading DenseNet121-MIMIC...")
    model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")

    # FIX: Dezactivăm pragurile vechi (pt eroarea de dimensiune)
    model.op_threshs = None

    model.classifier = nn.Linear(1024, num_classes)
    return model


class MaskedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, targets):
        mask = ~torch.isnan(targets)
        safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
        bce_loss = self.bce(preds, safe_targets)
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return (loss * mask).sum() / (mask.sum() + 1e-8)


class MaskedWeightedBCE(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def forward(self, logits, targets):
        mask = ~torch.isnan(targets)
        safe_targets = torch.where(mask, targets, torch.zeros_like(targets))

        loss = self.bce(logits, safe_targets)     # [B,C]
        loss = loss * mask.float()
        return loss.sum() / (mask.float().sum() + 1e-8)

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
# ============================================================
# 5. ENGINE
# ============================================================
def train_epoch(model, loader, optimizer, criterion, scaler, epoch_idx, total_epochs):
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"Epoca {epoch_idx}/{total_epochs} [Train]", ncols=120)

    printed_nan = False  # <--- adauga asta

    for images, labels in pbar:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        # DEBUG NaN labels (o singura data pe epoca)
        if not printed_nan:
            print(f"[DEBUG] NaN labels in this epoch: {torch.isnan(labels).sum().item()}")
            printed_nan = True

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        pbar.set_postfix({'loss': f"{np.mean(losses[-50:]):.4f}"})

    return np.mean(losses)


@torch.no_grad()
def validate(model, loader, criterion, metrics, epoch_idx, total_epochs):
    model.eval()
    losses = []
    for m in metrics.values(): m.reset()

    pbar = tqdm(loader, desc=f"Epoca {epoch_idx}/{total_epochs} [Valid]", ncols=120)

    for images, labels in pbar:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        losses.append(loss.item())
        probs = torch.sigmoid(outputs)
        mask = ~torch.isnan(labels)

        metrics['AUC'].update(probs, torch.nan_to_num(labels, nan=0).int())
        if mask.sum() > 0:
            metrics['Acc'].update(probs[mask], labels[mask].int())
            metrics['Prec'].update(probs[mask], labels[mask].int())
            metrics['Rec'].update(probs[mask], labels[mask].int())

        pbar.set_postfix({'val_loss': f"{np.mean(losses[-10:]):.4f}"})

    return (np.mean(losses),
            metrics['AUC'].compute().item(),
            metrics['Acc'].compute().item(),
            metrics['Prec'].compute().item(),
            metrics['Rec'].compute().item())


# def prepare_data_split(csv_path):
#     print(f"\n[DATA] Verificare split în {csv_path}...")
#     df = pd.read_csv(csv_path)
#     # Split logic (același ca înainte)
#     unique_patients = df['subject_id'].unique()
#     train_patients, remaining = train_test_split(unique_patients, test_size=0.3, random_state=42)
#     val_patients, test_patients = train_test_split(remaining, test_size=0.5, random_state=42)
#
#     patient_split_map = {}
#     for p in train_patients: patient_split_map[p] = 'train'
#     for p in val_patients: patient_split_map[p] = 'validate'
#     for p in test_patients:  patient_split_map[p] = 'test'
#
#     df['split'] = df['subject_id'].map(patient_split_map)
#     df.to_csv(csv_path, index=False)
#     print(f"  -> Split complet: {df['split'].value_counts().to_dict()}")
#     return df


# ============================================================
# 6. MAIN
# ============================================================
if __name__ == '__main__':
    # Config
    CSV_PATH = "mimic_complete_master_dataset.csv"
    FINAL_MODEL_FILE = "mimic_model_mBCE_2.pth"

    BATCH_SIZE = 128
    IMAGE_SIZE = 224
    NUM_WORKERS = 4

    EPOCHS_PHASE_1 = 30
    EPOCHS_PHASE_2 = 20

    LABEL_COLS = [
        'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion',
        'Fracture', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Other'
    ]

    # 1. Prepare Data
    df_master = pd.read_csv(CSV_PATH)


    df_train = df_master[df_master['split'] == 'train']
    df_val = df_master[df_master['split'] == 'val']

    # temp = df_train[LABEL_COLS]
    # pos = (temp == 1).sum(axis=0).values.astype(np.float32)
    # neg = (temp == 0).sum(axis=0).values.astype(np.float32)
    #
    # pos_weight = torch.tensor(neg / (pos + 1e-6), device=DEVICE)
    #
    # # IMPORTANT: ca sa nu supra-amplificam clasele rare (mai ales daca folosim si oversampling)
    # pos_weight = torch.clamp(pos_weight, max=50.0)
    #
    # print("[INFO] pos_weight:", pos_weight.cpu().numpy())

    # --- AICI APLICĂM OVERSAMPLING ---
    # Creăm sampler-ul bazat pe df_train
    train_sampler = make_weighted_sampler(df_train, LABEL_COLS)

    train_aug = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        ToTensorV2(),
    ])
    val_aug = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), ToTensorV2()])

    # Dataloader cu Sampler
    # IMPORTANT: Când folosim sampler, shuffle TREBUIE să fie False! (Samplerul face shuffle oricum)
    train_loader = DataLoader(
        MIMICDataset(df_train, LABEL_COLS, train_aug),
        batch_size=BATCH_SIZE,
        shuffle=False,  # <--- CRITIC: False când avem sampler
        sampler=train_sampler,  # <--- AICI ESTE MAGIA
        # shuffle=True,
        # sampler=None,

        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3
    )

    val_loader = DataLoader(
        MIMICDataset(df_val, LABEL_COLS, val_aug),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # 2. Init Model & Tools
    model = get_model(len(LABEL_COLS)).to(DEVICE)
    # criterion = MaskedWeightedBCE(pos_weight)
    criterion = MaskedBCE()
    scaler = torch.amp.GradScaler('cuda')

    metrics = {
        'AUC': AUROC(task="multilabel", num_labels=len(LABEL_COLS), average='macro').to(DEVICE),
        'Acc': Accuracy(task="binary").to(DEVICE),
        'Prec': Precision(task="binary").to(DEVICE),
        'Rec': Recall(task="binary").to(DEVICE)
    }

    best_auc_overall = 0.0

    # -----------------------------------------------------------
    # FAZA 1
    # -----------------------------------------------------------
    print(f"\n[FAZA 1] Antrenare Classifier ({EPOCHS_PHASE_1} Epoci)...")
    for param in model.features.parameters(): param.requires_grad = False
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS_PHASE_1 + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, epoch, EPOCHS_PHASE_1)
        v_loss, v_auc, v_acc, v_prec, v_rec = validate(model, val_loader, criterion, metrics, epoch, EPOCHS_PHASE_1)

        print(
            f" -> Epoca {epoch}: Loss: {t_loss:.4f} | Val: {v_loss:.4f} | AUC: {v_auc:.4f} | Acc: {v_acc:.4f} | Prec: {v_prec:.4f} | Rec: {v_rec:.4f}")

        if v_auc > best_auc_overall:
            best_auc_overall = v_auc
            save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(save_model.state_dict(), FINAL_MODEL_FILE)
            print(f"    >>> Model Salvat (AUC: {best_auc_overall:.4f})")

    # -----------------------------------------------------------
    # FAZA 2
    # -----------------------------------------------------------
    print(f"\n[FAZA 2] Fine-Tuning ({EPOCHS_PHASE_2} Epoci)...")
    if os.path.exists(FINAL_MODEL_FILE):
        print(" -> Se reîncarcă cel mai bun model din Faza 1...")
        model.load_state_dict(torch.load(FINAL_MODEL_FILE))

    for param in model.parameters(): param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4,
                                              steps_per_epoch=len(train_loader),
                                              epochs=EPOCHS_PHASE_2, pct_start=0.3)

    for epoch in range(1, EPOCHS_PHASE_2 + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, epoch, EPOCHS_PHASE_2)
        scheduler.step()
        v_loss, v_auc, v_acc, v_prec, v_rec = validate(model, val_loader, criterion, metrics, epoch, EPOCHS_PHASE_2)

        print(
            f" -> Epoca {epoch}: Loss: {t_loss:.4f} | Val: {v_loss:.4f} | AUC: {v_auc:.4f} | Acc: {v_acc:.4f} | Prec: {v_prec:.4f} | Rec: {v_rec:.4f}")

        if v_auc > best_auc_overall:
            best_auc_overall = v_auc
            save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(save_model.state_dict(), FINAL_MODEL_FILE)
            print(f"    >>> Model Actualizat! (New Best AUC: {best_auc_overall:.4f})")

    print("\n[DONE] Antrenament complet.")