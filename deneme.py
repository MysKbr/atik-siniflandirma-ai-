import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader
import os

# 📁 Veri yolları
train_dir = r"C:\Users\ASUS\OneDrive\Masaüstü\hazir_veriseti\train"
test_dir  = r"C:\Users\ASUS\OneDrive\Masaüstü\hazir_veriseti\test"

# ⚙️ Ayarlar
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📊 ImageNet normalization (KRİTİK)
weights = MobileNet_V2_Weights.DEFAULT
normalize = weights.transforms()

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

# 📂 Dataset
train_data = datasets.ImageFolder(train_dir, transform=transform_train)
test_data = datasets.ImageFolder(test_dir, transform=transform_test)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# 🧠 Model
model = models.mobilenet_v2(weights=weights)

# 4 sınıf (senin datasetine göre)
model.classifier[1] = nn.Linear(model.last_channel, 4)
model = model.to(DEVICE)

# ⚙️ Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 📈 Accuracy fonksiyonu
def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    return acc

# 🏋️ TRAIN
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    acc = evaluate()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Test Acc: {acc:.2f}%")

# 💾 MODEL KAYDET (SAFE PATH)
base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
save_path = os.path.join(base_dir, "atik_mobilenetv2_pytorch.pth")

torch.save(model.state_dict(), save_path)

print(f"✔ Model kaydedildi: {save_path}")