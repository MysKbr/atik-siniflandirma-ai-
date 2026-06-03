♻️ Atık Sınıflandırma Yapay Zeka Sistemi (MobileNetV2)
📌 Proje Açıklaması

Bu proje, derin öğrenme (deep learning) kullanarak görüntülerden atık türlerini sınıflandıran bir yapay zeka sistemidir.

Sistem, atıkları 4 sınıfa ayırır:

Cam
Kağıt
Metal
Plastik

Proje uçtan uca bir yapay zeka pipeline’ı içerir:
veri hazırlama → model eğitimi → web arayüzü

🧠 Kullanılan Teknolojiler
Python
PyTorch
Torchvision
MobileNetV2 (Transfer Learning)
Gradio (Web arayüzü)
PIL (Görsel işleme)
pillow-heif (HEIC destek)
📂 Proje Yapısı
ayir.py        → Veri setini hazırlar (train/test ayırır)
deneme.py      → Modeli eğitir (MobileNetV2)
app.py         → Web arayüzü (kullanıcı giriş ekranı)
hazir_veriseti → İşlenmiş veri seti
model.pth      → Eğitilmiş model dosyası
⚙️ 1. Veri Hazırlama (ayir.py)

Bu kod şunları yapar:

Ham görüntüleri okur
HEIC formatını JPG’ye çevirir
Görselleri karıştırır
%80 eğitim, %20 test olarak ayırır
Her sınıf için ayrı klasör oluşturur
Sonuç klasör yapısı:
hazir_veriseti/
   train/
      cam/
      kagit/
      metal/
      plastik/

   test/
      cam/
      kagit/
      metal/
      plastik/
🧠 2. Model Eğitimi (deneme.py)
Model:
MobileNetV2 (önceden eğitilmiş model kullanıldı)
Neden MobileNetV2?
Hafif
Hızlı
Az veriyle iyi sonuç verir
Eğitim Süreci:
Loss fonksiyonu: CrossEntropyLoss
Optimizer: Adam (0.001)
Epoch: 10
Veri artırma (augmentation):
Görseli çevirme
Döndürme
Değerlendirme:

Her epoch sonunda model test verisi ile kontrol edilir ve doğruluk (accuracy) hesaplanır.

Çıktı:

Eğitilen model şu dosyaya kaydedilir:

atik_mobilenetv2_pytorch.pth
🌐 3. Web Uygulaması (app.py)

Bu bölüm kullanıcı arayüzüdür.

Ne yapar?
Kullanıcı bir fotoğraf yükler
Model görüntüyü analiz eder
Hangi sınıfa ait olduğunu söyler
Çalışma mantığı:
Görsel yüklenir
Model işlemlerden geçirir
Softmax ile olasılıklar hesaplanır
En yüksek ihtimal gösterilir
Çalıştırma:
http://127.0.0.1:5000
📊 Model Performansı
MobileNetV2 transfer learning kullanıldı
Yüksek doğruluk oranına ulaşıldı
Gerçek zamanlı tahmin yapılabilir
💡 Projenin Özellikleri

✔ Uçtan uca yapay zeka sistemi
✔ Veri hazırlama otomasyonu
✔ Derin öğrenme modeli
✔ Web tabanlı arayüz
✔ Gerçek zamanlı tahmin

🚀 Nasıl Çalıştırılır?
1. Veri hazırlama:
python ayir.py
2. Model eğitimi:
python deneme.py
3. Uygulama:
python app.py
🎯 Sonuç

Bu proje, görüntü işleme ve yapay zeka kullanarak atıkların otomatik sınıflandırılmasını sağlayan uçtan uca bir sistemdir.

## 👨‍💻 Author
Şevval Gizem Günay - Miyase Kübra Özdemir
