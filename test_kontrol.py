import numpy as np
import os
import librosa

# Test verisini kontrol et
test_dizin = r'C:\Users\Lenovo\Desktop\123\MFCC_Test'
siniflar = ['Kerem_test', 'Emir_test',]

for sinif in siniflar:
    sinif_dizin = os.path.join(test_dizin, sinif)
    if not os.path.exists(sinif_dizin):
        print(f"Hata: {sinif_dizin} klasörü bulunamadı.")
        continue

    print(f"\n{sinif} için dosyalar kontrol ediliyor:")
    for dosya in os.listdir(sinif_dizin):
        dosya_yolu = os.path.join(sinif_dizin, dosya)
        try:
            # WAV dosyasını yükle ve MFCC çıkar
            y, sr = librosa.load(dosya_yolu, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
            print(f"{dosya} - MFCC boyutu: {mfcc.shape}")
        except Exception as e:
            print(f"Hata: {dosya} - {e}")
            # Eğitim ve test veri setindeki sınıf dağılımını kontrol et
from collections import Counter

