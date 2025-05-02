import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import logging
import time
from get_path import get_config


writer = SummaryWriter(log_dir="runs/eyeq")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from preprocessing import augmentation

from transformers import AutoModelForImageClassification, AutoFeatureExtractor

config = get_config()
# относительно корневой директории
DATA_DIR = config['DATA_DIR'] 
SRC_DIR = config["SRC_DIR"]

IMG_DIR = os.path.join(DATA_DIR, "preprocessed") # r"./data/preprocessed/"


class EyeQDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir, transform: transforms.Compose = None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.image_dir = img_dir

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]['image']
        bin_mark = self.df.iloc[idx]['bin_marks']

        img_path = os.path.join(self.image_dir, image_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, bin_mark


transform = transforms.Compose([
    augmentation()[0], 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

df = pd.read_csv(os.path.join(DATA_DIR, 'marking.csv'))
logger.info(df)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["bin_marks"], random_state=42)

train_dataset = EyeQDataset(train_df, IMG_DIR, transform=transform)
test_dataset = EyeQDataset(val_df, IMG_DIR, transform=transform)

train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=True)

def create_model():
    model_name = "google/efficientnet-b0" # https://huggingface.co/google/efficientnet-b0
    model = AutoModelForImageClassification.from_pretrained(model_name)

    # model = models.efficientnet_b0(pretrained = True)
    # model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    model.classifier = nn.Sequential(
        nn.Dropout(0.2, inplace=True),
        nn.Linear(model.classifier.in_features, 1)
    )

    model.config.num_labels = 1
    model.config.id2label = {0:'usable'}
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # "cpu"
logger.info(f"Using device: {device}")

def train(model, device, criterion, optimizer, save_path, num_epoch = 10, patience = 3):
    model.to(device)
    best_f1 = 0.0
    counter = 0 # считает плохие эпохи

    for epoch in range(num_epoch):
        start_time = time.time()
        model.train()

        correct = 0
        total = 0
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            logger.info(f"Train Loss: {loss.item():.4f}")



        model.eval()
        val_loss = 0.0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predictions = torch.sigmoid(outputs) >= 0.5
                correct += (predictions.float() == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = correct / total
        f1 = f1_score(all_labels, all_preds)
        epoch_time = time.time() - start_time
        logger.info(f"Epoch: {epoch}\nValidation Loss: {val_loss/len(test_loader):.4f}, Accuracy: {val_acc:.2%}, F1 Score: {f1:.4f}")

        if f1 > best_f1:
            counter = 0
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best model saved with F1: {best_f1:.4f}")

        else:
            counter += 1
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('F1/val', f1, epoch)
        writer.add_scalar('Time/epoch', epoch_time, epoch)

        if counter == patience:
            logger.info("Model not improving")
            break



def create_crit_optim(model):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return criterion, optimizer

dummy_input = torch.randn(1, 3, 224, 224).to(device)



if __name__ == "__main__":
    model = create_model()
    criterion, optimizer = create_crit_optim(model) 
    train(model, device, criterion, optimizer, "./best_model.pth")
    torch.onnx.export(
        model, 
        dummy_input, 
        os.path.join(DATA_DIR, 'EyeQ.onnx'),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'}, 
            'output': {0: 'batch_size'}
        }
    )