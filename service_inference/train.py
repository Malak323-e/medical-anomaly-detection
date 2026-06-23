import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

DATA_DIR = "../data/chest_xray/chest_xray"

# Augmentation pour le training (améliore la généralisation)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Pas d'augmentation pour val/test
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(DATA_DIR + "/train", transform=train_transform)
val_data   = datasets.ImageFolder(DATA_DIR + "/val",   transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False, num_workers=0)

print(f"Classes : {train_data.classes}")
print(f"Train : {len(train_data)} images | Val : {len(val_data)} images")

# ResNet18 pré-entraîné + fine-tuning
model = models.resnet18(pretrained=True)

# Gèle les premières couches, entraîne seulement les dernières
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

best_val_acc = 0.0

for epoch in range(8):
    # --- TRAIN ---
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)
        train_loss    += loss.item()

    # --- VAL ---
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    train_acc = 100 * train_correct / train_total
    val_acc   = 100 * val_correct   / val_total
    print(f"Epoch {epoch+1}/8 | Train acc: {train_acc:.1f}% | Val acc: {val_acc:.1f}%")

    # Sauvegarde le meilleur modèle
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "chest_model.pth")
        print(f"  -> Meilleur modèle sauvegardé ({val_acc:.1f}%)")

    scheduler.step()

print(f"\nEntraînement terminé. Meilleure val acc : {best_val_acc:.1f}%")
print("Modèle sauvegardé : chest_model.pth")
