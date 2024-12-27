import numpy as np
import os
from collections import Counter

# Eğitim ve test verilerinin bulunduğu dizinleri belirt
egitim_dizin = r'C:\Users\Lenovo\Desktop\123\MFCC'
test_dizin = r'C:\Users\Lenovo\Desktop\123\Bolunmus_wav'

# Eğitim ve test verilerini yüklemek için boş listeler
egitim_mfcc = []
test_mfcc = []

# Eğitim verilerinin yüklenmesi
print("Eğitim verilerini yükleme...")
for klasor_ad in os.listdir(egitim_dizin):
    klasor_yolu = os.path.join(egitim_dizin, klasor_ad)
    if os.path.isdir(klasor_yolu):  # Sadece klasörler için
        for dosya_ad in os.listdir(klasor_yolu):
            if dosya_ad.endswith('.npy'):
                dosya_yolu = os.path.join(klasor_yolu, dosya_ad)
                mfcc = np.load(dosya_yolu)
                egitim_mfcc.append(mfcc)

# Test verilerinin yüklenmesi
print("Test verilerini yükleme...")
for klasor_ad in os.listdir(test_dizin):
    klasor_yolu = os.path.join(test_dizin, klasor_ad)
    if os.path.isdir(klasor_yolu):  # Sadece klasörler için
        for dosya_ad in os.listdir(klasor_yolu):
            if dosya_ad.endswith('.npy'):
                dosya_yolu = os.path.join(klasor_yolu, dosya_ad)
                mfcc = np.load(dosya_yolu)
                test_mfcc.append(mfcc)

# 1. Veri dağılımını kontrol et
print("Eğitim veri dağılımı:")
print(Counter([mfcc.shape for mfcc in egitim_mfcc]))

print("Test veri dağılımı:")
print(Counter([mfcc.shape for mfcc in test_mfcc]))

# 2. MFCC özelliklerinin istatistiksel karşılaştırması
egitim_ort = np.mean([np.mean(mfcc) for mfcc in egitim_mfcc])
test_ort = np.mean([np.mean(mfcc) for mfcc in test_mfcc])

print(f"Eğitim verisi MFCC ortalaması: {egitim_ort}")
print(f"Test verisi MFCC ortalaması: {test_ort}")

egitim_std = np.std([np.mean(mfcc) for mfcc in egitim_mfcc])
test_std = np.std([np.mean(mfcc) for mfcc in test_mfcc])

print(f"Eğitim verisi MFCC standart sapması: {egitim_std}")
print(f"Test verisi MFCC standart sapması: {test_std}")

# 3. Boyut kontrolü
print("Eğitim verisi MFCC boyutları:")
print([mfcc.shape for mfcc in egitim_mfcc])

print("Test verisi MFCC boyutları:")
print([mfcc.shape for mfcc in test_mfcc])