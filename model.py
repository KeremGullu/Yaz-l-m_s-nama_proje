from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import joblib
import os
import numpy as np
import collections

# MFCC özelliklerinin bulunduğu dizin
mfcc_dizin = r'C:\Users\Lenovo\Desktop\123\MFCC'

X = []
y = []

# MFCC dosyalarını yükleme
print("Eğitim veri seti hazırlanıyor...")
for klasor_adı in os.listdir(mfcc_dizin):
    klasor_yolu = os.path.join(mfcc_dizin, klasor_adı)
    if os.path.isdir(klasor_yolu):  # Sadece klasörler üzerinde çalış
        for dosya_adı in os.listdir(klasor_yolu):
            if dosya_adı.endswith('.npy'):
                dosya_yolu = os.path.join(klasor_yolu, dosya_adı)
                try:
                    mfcc = np.load(dosya_yolu)
                    if mfcc.size > 0:  # Boş dosyaları atla
                        X.append(np.mean(mfcc, axis=1))  # Ortalama MFCC vektörü
                        y.append(klasor_adı)  # Klasör adını etiket olarak kullan
                    else:
                        print(f"UYARI: Boş MFCC dosyası atlandı: {dosya_yolu}")
                except Exception as e:
                    print(f"HATA: Dosya yüklenemedi: {dosya_yolu}. Hata: {e}")

# X ve y dizilerini numpy dizilerine dönüştür
X = np.array(X)
y = np.array(y)

# Veri seti kontrolü
if X.size == 0 or y.size == 0:
    print("Hata: Eğitim veri seti boş. MFCC dosyalarını kontrol edin.")
    exit()

# Sınıf dağılımını kontrol et
print("Eğitim veri seti sınıf dağılımı:", collections.Counter(y))

# Etiketleri sayısal değerlere dönüştür
le = LabelEncoder()
y = le.fit_transform(y)

# Veri setini eğitim ve test kümelerine ayırma
print("Veri seti eğitim ve test kümelerine ayrılıyor...")
X_egitim, X_test, y_egitim, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Eğitim veri setindeki sınıf dengesizliğini giderme (Oversampling)
print("Eğitim veri setindeki sınıf dengesizliği gideriliyor...")
X_egitim_balanced = []
y_egitim_balanced = []

for label in np.unique(y_egitim):
    indices = np.where(y_egitim == label)[0]
    X_class = X_egitim[indices]
    y_class = y_egitim[indices]
    if len(X_class) < 90:  # Az temsil edilen sınıfları 90 örneğe tamamla
        X_resampled, y_resampled = resample(
            X_class, y_class, replace=True, n_samples=90, random_state=42
        )
    else:
        X_resampled, y_resampled = X_class, y_class
    X_egitim_balanced.extend(X_resampled)
    y_egitim_balanced.extend(y_resampled)

X_egitim_balanced = np.array(X_egitim_balanced)
y_egitim_balanced = np.array(y_egitim_balanced)

print("Yeni Eğitim Veri Seti Sınıf Dağılımı:", collections.Counter(y_egitim_balanced))

# Model oluşturma ve eğitme
print("MLP modeli oluşturuluyor ve eğitiliyor...")
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
model.fit(X_egitim_balanced, y_egitim_balanced)

# Modelin doğruluğunu değerlendirme
dogruluk = model.score(X_test, y_test)
print(f"Model doğruluğu: {dogruluk}")

# Eğitimden sonra sınıflandırma raporu ve karışıklık matrisi
print("Sınıflandırma Raporu (Test Kümesi):")
tahminler = model.predict(X_test)
print(classification_report(y_test, tahminler))

print("Karışıklık Matrisi:")
print(confusion_matrix(y_test, tahminler, labels=np.unique(y_test)))

# Modeli diske kaydetme
model_kayit_yolu = r'C:\Users\Lenovo\Desktop\123\model-kerem-emir.pkl'
joblib.dump(model, model_kayit_yolu)
print(f"Model başarıyla kaydedildi: {model_kayit_yolu}")