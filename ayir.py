import os
import random
from PIL import Image
import pillow_heif

# HEIC desteği
pillow_heif.register_heif_opener()

# ayir.py'nin bulunduğu klasör
base_dir = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(base_dir, "hazir_veriseti")

categories = ["cam", "kagit", "metal", "plastik"]

train_ratio = 0.8

for category in categories:

    category_path = os.path.join(base_dir, category)

    print("Kontrol edilen klasör:", category_path)

    if not os.path.exists(category_path):
        print(f"HATA: {category_path} bulunamadı!")
        continue

    images = os.listdir(category_path)

    random.shuffle(images)

    train_count = int(len(images) * train_ratio)

    train_images = images[:train_count]
    test_images = images[train_count:]

    for split_name, split_images in [("train", train_images), ("test", test_images)]:

        save_folder = os.path.join(output_dir, split_name, category)

        os.makedirs(save_folder, exist_ok=True)

        for image_name in split_images:

            image_path = os.path.join(category_path, image_name)

            try:
                img = Image.open(image_path)

                new_name = os.path.splitext(image_name)[0] + ".jpg"

                save_path = os.path.join(save_folder, new_name)

                rgb_img = img.convert("RGB")

                rgb_img.save(save_path, "JPEG")

            except Exception as e:
                print(f"Hata oluştu: {image_name} -> {e}")

print("HEIC dönüştürme ve train/test ayırma tamamlandı.")