from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Kaydedilmiş modeli yükleme
model_yolu = r'C:\Users\Lenovo\Desktop\123\model-kerem-emir.pkl'
model = joblib.load(model_yolu)

# Test veri setini hazırlama
mfcc_dizin = r'C:\Users\Lenovo\Desktop\123\MFCC_Test'
X_test = []
y_test = []

print("Test veri seti hazırlanıyor...")
for klasor_adı in os.listdir(mfcc_dizin):
    klasor_yolu = os.path.join(mfcc_dizin, klasor_adı)
    if os.path.isdir(klasor_yolu):
        for dosya_adı in os.listdir(klasor_yolu):
            if dosya_adı.endswith('.npy'):
                dosya_yolu = os.path.join(klasor_yolu, dosya_adı)
                try:
                    mfcc = np.load(dosya_yolu)
                    if mfcc.size > 0:
                        # MFCC'nin ortalamasını kullanma (gerekirse düzeltin)
                        X_test.append(np.mean(mfcc, axis=1))
                        y_test.append(klasor_adı)
                except Exception as e:
                    print(f"Hata: {dosya_yolu} yüklenemedi. Hata: {e}")

X_test = np.array(X_test)
y_test = np.array(y_test)

if X_test.size == 0 or y_test.size == 0:
    print("Hata: Test veri seti boş.")
    exit()

# Etiketleri sayısal değerlere dönüştürme
le = LabelEncoder()
y_test = le.fit_transform(y_test)

# Test veri seti üzerinde tahmin yapma
tahminler = model.predict(X_test)

# Model başarı oranını hesaplama
dogruluk = accuracy_score(y_test, tahminler)
print(f"Model doğruluğu: {dogruluk * 100:.2f}%")

# Sınıflandırma raporu ve karışıklık matrisi yazdırma
print("Sınıflandırma Raporu:")
print(classification_report(y_test, tahminler))

print("Karışıklık Matrisi:")
print(confusion_matrix(y_test, tahminler))