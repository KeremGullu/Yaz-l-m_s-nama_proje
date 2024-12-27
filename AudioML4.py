import os
from pydub import AudioSegment

# Wav dosyalarının bulunduğu ana dizin
wav_dir = "123/Sesler_wav"

# Çıkış dizini (bölünmüş dosyalar kaydedilecek)
output_dir = "123/Bolunmus_wav"
os.makedirs(output_dir, exist_ok=True)

# Klasörleri tanımla
folders = ["Kerem_ses", "Emir_ses",]

# Her bir klasör için işlemleri uygula
for folder in folders:
    folder_path = os.path.join(wav_dir, folder)
    output_folder = os.path.join(output_dir, folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Klasördeki wav dosyalarını listele
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    for wav_file in wav_files:
        file_path = os.path.join(folder_path, wav_file)
        audio = AudioSegment.from_wav(file_path)
        
       # Ses dosyasını 5 saniyelik parçalara böl
chunk_length_ms = 5000  # 5 saniye
chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

# Parçaları kaydet
for i, chunk in enumerate(chunks):
    chunk.export(os.path.join(output_folder, f"{wav_file[:-4]}_part{i+1}.wav"), format="wav")



import librosa
import numpy as np
import os

# Bölünmüş ses dosyalarının bulunduğu dizin
bolunmus_wav_dir = '123/Bolunmus_wav'

# MFCC özelliklerinin kaydedileceği dizin
mfcc_dir = '123/MFCC'

# MFCC parametreleri
n_mfcc = 128

# Framelere ayırma parametreleri
frame_length = 25  # milisaniye cinsinden
frame_stride = 10   # milisaniye cinsinden

# Dosya klasörlerini işleme
folders = ["Kerem_ses", "Emir_ses",]

# Her klasör için işlemleri uygula
for folder in folders:
    folder_path = os.path.join(bolunmus_wav_dir, folder)
    mfcc_folder = os.path.join(mfcc_dir, folder)
    os.makedirs(mfcc_folder, exist_ok=True)  # MFCC çıktılarının saklanacağı dizini oluştur

    for dosya_adı in os.listdir(folder_path):
        if dosya_adı.endswith('.wav'):
            dosya_yolu = os.path.join(folder_path, dosya_adı)
            
            # Ses dosyasını yükle
            ses, sr = librosa.load(dosya_yolu, sr=None)
            
            # MFCC özelliklerini çıkar
            mfcc = librosa.feature.mfcc(
                y=ses, sr=sr, n_mfcc=n_mfcc,
                hop_length=int(frame_stride * sr / 1000),
                n_fft=int(frame_length * sr / 1000)
            )
            
            # MFCC özelliklerini kaydet
            mfcc_dosya_adı = dosya_adı.split('.')[0] + '.npy'
            mfcc_dosya_yolu = os.path.join(mfcc_folder, mfcc_dosya_adı)
            np.save(mfcc_dosya_yolu, mfcc)

print("MFCC özellikleri başarıyla çıkarıldı.")

# Modeli Eğitmek

# In[28]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# MFCC özelliklerinin bulunduğu dizin
mfcc_dizin = 'Dosyalar/Fatih-Müyesser-Sümeyye-mfcc/'

X = []
y = []

# MFCC dosyalarını yükleme
for dosya_adı in os.listdir(mfcc_dizin):
    if dosya_adı.endswith('.npy'):
        dosya_yolu = os.path.join(mfcc_dizin, dosya_adı)
        mfcc = np.load(dosya_yolu)
        X.append(np.mean(mfcc, axis=1))  # Her dosya için ortalama MFCC vektörü
        y.append(dosya_adı.split(' ')[0])  # Dosya adından etiket çıkarma

X = np.array(X)
y = np.array(y)

# Etiketleri sayısal değerlere dönüştürme
le = LabelEncoder()
y = le.fit_transform(y)

# Veri kümesini eğitim ve test kümelerine ayırma
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLP modeli oluşturma ve eğitme
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
model.fit(X_egitim, y_egitim)

# Modelin doğruluğunu değerlendirme
dogruluk = model.score(X_test, y_test)
print(f"Model doğruluğu: {dogruluk}")

print(X_egitim.shape)


# In[29]:


# Modeli diske kaydetme
model_kayit_yolu = 'model-disk.pkl'
joblib.dump(model, model_kayit_yolu)


# In[30]:


from sklearn.metrics import classification_report, confusion_matrix

# Test veri seti üzerinde tahmin yapma
tahminler = model.predict(X_test)

# Sınıflandırma raporu ve karışıklık matrisini yazdırma
print("Sınıflandırma Raporu:")
print(classification_report(y_test, tahminler))

print("Karışıklık Matrisi:")
print(confusion_matrix(y_test, tahminler))


# Uygulama Geliştirme - Dosya

# In[31]:


# Eğitilmiş modeli yükleme
model_kayit_yolu = 'model-kerem-emir.pkl'
model = joblib.load(model_kayit_yolu)

# Test edilecek ses dosyasının MFCC özelliklerini yükleme
test_ses_dosyasi = 'Dosyalar/Fatih-Müyesser-Sümeyye-Talha-mfcc/Müyesser (1).npy'
mfcc = np.load(test_ses_dosyasi)

# Model üzerinden tahmin yapma
tahmin = model.predict(np.mean(mfcc, axis=1).reshape(1, -1))

# Tahmini sonuç
print("Tahmin edilen kişi: ", tahmin)


# Uygulama Geliştirme - Mikrofon

# In[37]:

import sounddevice as sd
import soundfile as sf

# Eğitilmiş modeli yükleme
model_kayit_yolu = 'model-fatih-müyesser-sümeyye.pkl'
model = joblib.load(model_kayit_yolu)

sinif_isimleri = ['Fatih','Sümeyye','Müyesser']

# Mikrofondan ses almak için gerekli parametreler
saniye_basina_ornek = 44100  # Örnekleme hızı (örneğin, 44100 Hz)
saniye = 5  # 5 saniyelik ses al
kanal_sayisi = 1  # Tek kanallı ses

while True:
    print("Konuşun...")
    ses = sd.rec(int(saniye_basina_ornek * saniye), samplerate=saniye_basina_ornek, channels=kanal_sayisi, dtype='float32')
    sd.wait()  # Ses alımının tamamlanmasını bekleyin
    
    # Ses dosyasını WAV olarak kaydetme
    kayit_yolu = "kayit.wav"
    sf.write(kayit_yolu, np.squeeze(ses), saniye_basina_ornek)
    
    # WAV dosyasını yükleme ve MFCC özelliklerini çıkarma
    y, sr = librosa.load(kayit_yolu, sr=saniye_basina_ornek)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
    mfcc = np.mean(mfcc.T, axis=0)  # Ortalama MFCC vektörü
    
    # Model üzerinden tahmin yapma
    tahmin_indeksi = model.predict(mfcc.reshape(1, -1))[0]
    tahmin_isim = sinif_isimleri[tahmin_indeksi]
    
    # Tahmini sonuç
    print("Tahmin edilen kişi: ", tahmin_isim)
    #os.remove(kayit_yolu)  # Kayıt dosyasını temizleme


# In[17]:


import speech_recognition as sr
from pydub import AudioSegment

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio.export(wav_file_path, format="wav")

def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio, language="tr-TR")
        return transcript
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"

kelimeler = []


if __name__ == "__main__":
    wav_file_path = "kayit.wav"  # Path to the converted WAV file

    # Transcribe the WAV file-
    transcript = transcribe_audio(wav_file_path)
    
    kelimeler.extend(transcript.split())
    
    print("Transcript:")
    print(transcript)
    
    print("Kelime Sayısı:")
    print(len(kelimeler))


# %%
