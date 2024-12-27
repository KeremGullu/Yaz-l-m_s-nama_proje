import numpy as np
import sounddevice as sd
import joblib
import librosa
import soundfile as sf
import os

# Eğitilmiş modeli yükleme
model_kayit_yolu = r'C:\Users\Lenovo\Desktop\123\model-kerem-emir.pkl'
try:
    model = joblib.load(model_kayit_yolu)
    print("Model başarıyla yüklendi!")
except Exception as e:
    print("Model yükleme sırasında hata oluştu:", e)
    exit()

# Sınıf isimleri
sinif_isimleri = ['Kerem', 'Emir', ]

# Mikrofondan ses almak için gerekli parametreler
saniye_basina_ornek = 44100  # Örnekleme hızı (örneğin, 44100 Hz)
saniye = 5  # 5 saniyelik ses al
kanal_sayisi = 1  # Tek kanallı ses

# Ses kaydı alma ve tahmin işlemi
try:
    print("Konuşun...")
    ses = sd.rec(int(saniye_basina_ornek * saniye), samplerate=saniye_basina_ornek, channels=kanal_sayisi, dtype='float32')
    sd.wait()  # Ses alımının tamamlanmasını bekleyin
    print("Ses kaydı tamamlandı!")
    
    # Kaydedilen sesi bir WAV dosyasına yazma
    kayit_yolu = r'C:\Users\Lenovo\Desktop\123\kayit.wav'
    sf.write(kayit_yolu, np.squeeze(ses), saniye_basina_ornek)
    print(f"Ses {kayit_yolu} olarak kaydedildi.")

    # WAV dosyasını yükleme ve MFCC özelliklerini çıkarma
    y, sr = librosa.load(kayit_yolu, sr=saniye_basina_ornek)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
    mfcc = np.mean(mfcc.T, axis=0)  # Ortalama MFCC vektörü

    # Model üzerinden tahmin yapma
    tahmin_indeksi = model.predict(mfcc.reshape(1, -1))[0]
    tahmin_isim = sinif_isimleri[tahmin_indeksi]

    # Tahmini sonuç
    print("Tahmin edilen kişi: ", tahmin_isim)

except Exception as e:
    print("Bir hata oluştu:", e)
    
import speech_recognition as sr
from pydub import AudioSegment


# MP3 dosyasını WAV formatına dönüştürme fonksiyonu
def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio.export(wav_file_path, format="wav")

# Ses dosyasını yazıya dökme fonksiyonu
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
    # Kaydedilen WAV dosyasının yolu
    wav_file_path = r'C:\Users\Lenovo\Desktop\123\kayit.wav'  # Projeye uygun dosya yolu

    # WAV dosyasını yazıya dökme
    transcript = transcribe_audio(wav_file_path)
    
    # Transkriptten alınan kelimeleri listeye ekleme
    kelimeler.extend(transcript.split())
    
    # Yazıya döküm ve kelime sayısını ekrana yazdırma
    print("Transcript:")
    print(transcript)
    
    print("Kelime Sayısı:")
    print(len(kelimeler))