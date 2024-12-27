import os

# Eğitim veri setini yükleme
egitim_y = []
egitim_dizin = r'C:\Users\Lenovo\Desktop\123\MFCC_Egitim'

for klasor_adı in os.listdir(egitim_dizin):
    klasor_yolu = os.path.join(egitim_dizin, klasor_adı)
    if os.path.isdir(klasor_yolu):
        for dosya_adı in os.listdir(klasor_yolu):
            if dosya_adı.endswith('.npy'):
                egitim_y.append(klasor_adı)

# Test veri setini yükleme
test_y = []
test_dizin = r'C:\Users\Lenovo\Desktop\123\MFCC_Test'

for klasor_adı in os.listdir(test_dizin):
    klasor_yolu = os.path.join(test_dizin, klasor_adı)
    if os.path.isdir(klasor_yolu):
        for dosya_adı in os.listdir(klasor_yolu):
            if dosya_adı.endswith('.npy'):
                test_y.append(klasor_adı)

# Eğitim ve test dosyalarını karşılaştırma
egitim_dosyalar = set([str(dosya) for dosya in egitim_y])
test_dosyalar = set([str(dosya) for dosya in test_y])

ortak_dosyalar = egitim_dosyalar.intersection(test_dosyalar)

if ortak_dosyalar:
    print(f"Veri sızıntısı var! Ortak dosyalar: {ortak_dosyalar}")
else:
    print("Eğitim ve Test veri setleri tamamen ayrı!")