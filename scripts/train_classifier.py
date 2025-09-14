import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import os
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join('..', 'data', 'cleanliness_data')
MODEL_SAVE_PATH = os.path.join('..', 'weights', 'cleanliness_classifier_best.pth')
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.001

def main():
    print(f"Используемое устройство: {DEVICE}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'valid']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2) for x in ['train', 'valid']}
    
    model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1) # Заменяем голову на бинарную классификацию
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss() # Стабильная функция потерь для бинарной классификации
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Цикл обучения ---
    best_f1 = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f'Эпоха {epoch+1}/{NUM_EPOCHS}')
        
        # Фаза обучения
        model.train()
        for inputs, labels in tqdm(dataloaders['train'], desc="Training"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Фаза валидации
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(dataloaders['valid'], desc="Validation"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_f1 = f1_score(all_labels, all_preds)
        print(f'Validation F1-score: {epoch_f1:.4f}')

        # Сохраняем лучшую модель
        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'Новая лучшая модель сохранена в {MODEL_SAVE_PATH} с F1-score: {best_f1:.4f}')

if __name__ == '__main__':
    main()