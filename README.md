♻️ Atık Sınıflandırma Yapay Zeka Sistemi (MobileNetV2) 📌 Proje Açıklaması

Bu proje, derin öğrenme (derin öğrenme) kullanarak görüntülerden atık türlerini sınıflandıran bir yapay zeka sistemidir.

Sistem, attıkları 4 sınıfa ayrılır:

Cam Kağıt Metal Plastik

Proje uçtan uca bir yapay zeka boru hattı'ı içerir: veri hazırlama → model eğitimi → web arayüzü

🧠 kullanılan Teknolojiler Python PyTorch Torchvision MobileNetV2 (Transfer Learning) Gradio (Web arayüzü) PIL (Görsel işleme) yastık-heif (HEIC destek) 📂 Proje Yapısı ayir.py → Veri setini hazırlar (train/test ayırma) deneme.py → Model eğitimiir (MobileNetV2) app.py → Web arayüzü (kullanıcı giriş ekranı) hazir_veriseti → İşlenmiş seti model.pth → Eğitilmiş model şeması ⚙️ 1. Veri Hazırlama (ayır.py)

Bu kod yapar:

Ham görüntüleri okur HEIC formatını JPG'ye çeviri görsellerini karıştırır %80 eğitim, %20 test olarak ayırır Her sınıf için ayrı klasörler oluşturur Sonuç klasör yapısı: hazir_veriseti/ train/ cam/ kagit/ metal/ plastik/

test/ cam/ kagit/ metal/ plastik/ 🧠 2. Model Eğitimi (deneme.py) Model: MobileNetV2 (önceden sınıflandırılmış model dökümü) Neden MobileNetV2? Hafif Hızlı Az veriyle iyi sonuç verir Eğitim Süreci: Loss fonksiyonu: CrossEntropyLoss Optimizer: Adam (0.001) Epoch: 10 Veriyi iyileştirme (artırma): görüntüyü çevirme Döndürme Değerlendirme:

Her dönem sonunda model testi verisi ile kontrol edilir ve doğruluk (doğruluk) hesaplanır.

Çıktı:

Eğitilen model şu dosyaya benziyordu:

atik_mobilenetv2_pytorch.pth 🌐 3. Web Uygulaması (app.py)

Bu bölüm kullanıcı arayüzüdür.

Ne yapar? Kullanıcı bir yükler Modelin analizleri hangi sınıfa ait olduğunu söyler Çalışma mantığı: Görsel olarak Model işlemlerden geçer Softmax ile olasılıklar hesaplamalar En yüksek ihtimalle çalıştırılırma: http://127.0.0.1:5000 📊 Model Performansı MobileNetV2 transfer öğrenme Yüksek doğrulukla ulaşılabilir Gerçek zamanlı tahmin yapılabilir

✔ Uçtan uca yapay zeka sistemi ✔ Veri hazırlama otomasyonu ✔ Derin öğrenme modeli ✔ Web tabanlı programlama ✔ Gerçek zamanlı tahmin

🚀 Nasıl Çalıştırılır?

1-Veri hazırlama: python ayir.py
2-Model eğitimi: python deneme.py
3-Uygulama: python app.py 🎯 Sonuç
4-Bu proje, görüntü işleme ve yapay zeka kullanarak atıkların otomatik olarak sınıflandırılmasını sağlayan uçtan uca bir sistemdir.


Yazar
Miyase Kübra Özdemir - Şevval Gizem Günay
