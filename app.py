import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
import os

print("🔥 SİSTEM BAŞLATILIYOR...")

# =========================
# 📁 MODEL AYARLARI
# =========================
classes = ["cam", "kagit", "metal", "plastik"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 KESİN MODEL YOLU (HATASIZ)
MODEL_PATH = r"C:\Users\ASUS\OneDrive\Masaüstü\hazir_veriseti\atik_mobilenetv2_pytorch.pth"

# =========================
# 🧠 MODEL
# =========================
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 4)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("✅ MODEL BAŞARIYLA YÜKLENDİ")
except Exception as e:
    print("❌ MODEL YÜKLENEMEDİ HATA:", e)

model.to(device)
model.eval()

# =========================
# 🖼️ TRANSFORM (TRAIN İLE AYNI OLMALI)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# 🧠 TAHMİN
# =========================
def predict(image):
    if image is None:
        return {}

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]

    return {classes[i]: float(probs[i]) for i in range(len(classes))}

# =========================
# 🎨 GRADIO ARAYÜZ
# =========================
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="♻️ Atık Görseli Yükle"),
    outputs=gr.Label(num_top_classes=4),
    title="Atık Sınıflandırma AI",
    description="Cam - Kağıt - Metal - Plastik sınıflandırma sistemi"
)

# =========================
# 🚀 ÇALIŞTIR
# =========================
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=5000)