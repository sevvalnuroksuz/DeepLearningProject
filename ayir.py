import os
import shutil
import random

orijinal_klasor = r"C:\Users\Huaweı\Downloads\verisetivs"  # Tüm verilerin olduğu klasör (her sınıfın alt klasörü var)
hedef_klasor = r"veri_ayrilmis"
oran = 0.2  # %20 validation

# Tüm sınıflar (alt klasör isimleri)
siniflar = os.listdir(orijinal_klasor)

for sinif in siniflar:
    sinif_dosya_yolu = os.path.join(orijinal_klasor, sinif)
    resimler = os.listdir(sinif_dosya_yolu)
    random.shuffle(resimler)  # karıştır

    val_sayisi = int(len(resimler) * oran)
    val_resimler = resimler[:val_sayisi]
    train_resimler = resimler[val_sayisi:]

    # Klasörleri oluştur
    for veri_tipi, resim_listesi in [("train", train_resimler), ("val", val_resimler)]:
        hedef_klasor_yolu = os.path.join(hedef_klasor, veri_tipi, sinif)
        os.makedirs(hedef_klasor_yolu, exist_ok=True)
        for resim in resim_listesi:
            kaynak = os.path.join(sinif_dosya_yolu, resim)
            hedef = os.path.join(hedef_klasor_yolu, resim)
            shutil.copy2(kaynak, hedef)

print("Veri başarıyla ayrıldı: %80 train, %20 val")
