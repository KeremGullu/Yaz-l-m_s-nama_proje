import tkinter as tk
from tkinter import ttk, font
from tkinter import messagebox
import customtkinter as ctk
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr
from pydub import AudioSegment
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import joblib
import librosa
import io
import base64
from googletrans import Translator
from textblob import TextBlob
import logging


# Logger ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    def __init__(self):
        self.translator = Translator()
        # Türkçe duygu sözlüğü
        self.tr_emotion_dict = {
            "mutlu": ["mutlu", "sevinçli", "neşeli", "güzel", "harika", "muhteşem", "süper"],
            "mutsuz": ["üzgün", "mutsuz", "kötü", "berbat", "kederli", "acı"],
            "kızgın": ["kızgın", "sinirli", "öfkeli", "rahatsız", "bıktım"],
            "şaşkın": ["şaşkın", "şaşırdım", "inanamıyorum", "inanılmaz"],
            "nötr": ["normal", "fena değil", "idare eder", "olağan"]
        }

    def analyze_emotion(self, text):
        try:
            if not text or text == "Ses anlaşılamadı":
                return {
                    "duygular": {
                        "mutlu": 0.0,
                        "mutsuz": 0.0,
                        "nötr": 0.0,
                        "üzgün": 0.0,
                        "şaşkın": 0.0
                    },
                    "baskın_duygu": "belirsiz",
                    "güven_skoru": 0.0
                }

            text = text.lower()
            words = text.split()
            
            # Her duygu için eşleşme sayısını tut
            emotion_counts = {emotion: 0 for emotion in self.tr_emotion_dict.keys()}
            
            # Kelimeleri analiz et
            for word in words:
                for emotion, word_list in self.tr_emotion_dict.items():
                    if word in word_list:
                        emotion_counts[emotion] += 1

            # Duygu yüzdelerini hesapla
            emotion_percentages = {}
            total_matches = sum(emotion_counts.values())
            
            if total_matches > 0:
                # Her duygu için yüzdelik hesapla (toplam 100% olacak şekilde)
                for emotion, count in emotion_counts.items():
                    percentage = (count / total_matches) * 100
                    emotion_percentages[emotion] = round(percentage, 1)
            else:
                # Eşleşme yoksa TextBlob ile analiz yap
                try:
                    translated = self.translator.translate(text, dest='en')
                    analysis = TextBlob(translated.text)
                    polarity = analysis.sentiment.polarity
                    subjectivity = analysis.sentiment.subjectivity
                    
                    # TextBlob sonuçlarını normalize edilmiş duygu yüzdelerine dönüştür
                    if polarity > 0:
                        total = abs(polarity) + (1 - abs(polarity)) + subjectivity
                        emotion_percentages = {
                            "mutlu": round((polarity / total) * 100, 1),
                            "mutsuz": 0.0,
                            "nötr": round(((1 - abs(polarity)) / total) * 100, 1),
                            "üzgün": 0.0,
                            "şaşkın": round((subjectivity / total) * 100, 1)
                        }
                    elif polarity < 0:
                        total = abs(polarity) + (1 - abs(polarity)) + subjectivity
                        emotion_percentages = {
                            "mutlu": 0.0,
                            "mutsuz": round((abs(polarity) / 2 / total) * 100, 1),
                            "nötr": round(((1 - abs(polarity)) / total) * 100, 1),
                            "üzgün": round((abs(polarity) / 2 / total) * 100, 1),
                            "şaşkın": round((subjectivity / total) * 100, 1)
                        }
                    else:
                        emotion_percentages = {
                            "mutlu": 0.0,
                            "mutsuz": 0.0,
                            "nötr": round(100 - subjectivity, 1),
                            "üzgün": 0.0,
                            "şaşkın": round(subjectivity, 1)
                        }

                    # Yüzdelerin toplamının 100 olmasını sağla
                    total = sum(emotion_percentages.values())
                    if total > 0:
                        factor = 100 / total
                        emotion_percentages = {
                            emotion: round(score * factor, 1)
                            for emotion, score in emotion_percentages.items()
                        }

                except Exception as e:
                    print(f"TextBlob analiz hatası: {str(e)}")
                    emotion_percentages = {
                        "mutlu": 0.0,
                        "mutsuz": 0.0,
                        "nötr": 100.0,
                        "üzgün": 0.0,
                        "şaşkın": 0.0
                    }

            # En yüksek duyguyu bul
            baskın_duygu = max(emotion_percentages.items(), key=lambda x: x[1])
            
            return {
                "duygular": emotion_percentages,
                "baskın_duygu": baskın_duygu[0],
                "güven_skoru": baskın_duygu[1] / 100
            }
                
        except Exception as e:
            print(f"Duygu analizi hatası: {str(e)}")
            return {
                "duygular": {emotion: 0.0 for emotion in self.tr_emotion_dict.keys()},
                "baskın_duygu": "belirsiz",
                "güven_skoru": 0.0
            }

class AudioRecorderUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Speaker Recognition")
        self.is_recording = False
        self.frames = []
        self.stop_identification = False  # Konuşmacı tanıma işlemini durdurmak için bayrak

        button_frame = tk.Frame(root)
        button_frame.pack(padx=10, pady=10)

        button_style = {
            "font": ("Arial", 12, "normal"),
            "bg": "#4CAF50",
            "fg": "#FFFFFF",
            "activebackground": "#45a049",
            "activeforeground": "#FFFFFF",
            "relief": "raised",
            "bd": 3,
            "width": 15
        }

        self.start_button = tk.Button(button_frame, text="Kayıt Başlat", command=self.start_recording, **button_style)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.stop_button = tk.Button(button_frame, text="Kayıt Durdur", command=self.stop_recording, **button_style)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.process_button = tk.Button(button_frame, text="Kayıdı İşle", command=self.process_recording, **button_style)
        self.process_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.custom_font = font.Font(family="Times New Roman", size=15, weight="normal")

        # Matplotlib stil ayarları
        plt.style.use('dark_background')
        
        # Sinyal grafiği ayarları
        self.signal_plot = plt.figure(figsize=(8, 4), facecolor='black')
        self.ax = self.signal_plot.add_subplot(111)
        self.ax.set_facecolor('black')
        
        # Histogram ayarları
        self.histogram_plot = plt.figure(figsize=(8, 4), facecolor='black')
        self.hist_ax = self.histogram_plot.add_subplot(111)
        self.hist_ax.set_facecolor('black')
        
        self.canvas = FigureCanvasTkAgg(self.signal_plot, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Histogram sınırlarını belirle
        self.hist_ax.set_xlim(-1, 1)  # Ses verisi genellikle -1 ile 1 arasında
        self.hist_ax.set_ylim(0, 1000)  # Y ekseni sınırı
        
        # Grafik başlıkları
        self.hist_ax.set_title('Ses Verisi Dağılımı (Canlı)')
        self.hist_ax.set_xlabel('Amplitüd')
        self.hist_ax.set_ylabel('Frekans')
        
        # Grafiklerin kenar boşluklarını ayarla
        self.signal_plot.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        self.histogram_plot.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        
        plt.ion()  # Interaktif modu aç

        self.info_text = tk.Text(root, height=5, wrap=tk.WORD)
        self.info_text.pack(fill=tk.X)

        self.info_text.tag_configure("header", font=("Arial", 12, "bold"), foreground="#4CAF50")
        self.info_text.tag_configure("content", font=("Arial", 12, "normal"), foreground="#000000")

        # Eğitilmiş modeli yükleme
        model_kayit_yolu = 'model-kerem-emir.pkl'
        self.model = joblib.load(model_kayit_yolu)

        self.sinif_isimleri = ['Kerem','Emir']

        # Mikrofondan ses almak için gerekli parametreler
        self.saniye_basina_ornek = 44100  # Örnekleme hızı (örneğin, 44100 Hz)
        self.saniye = 5  # 5 saniyelik ses al
        self.kanal_sayisi = 1  # Tek kanallı ses

        # Thread güvenli güncelleme için queue ekle
        self.update_queue = []
        self.is_updating = False
        
        # Her 50ms'de bir güncelleme yap
        self.root.after(50, self.process_updates)

        self.emotion_analyzer = EmotionAnalyzer()
        self.topic_analyzer = TopicAnalyzer()

    def start_recording(self):
        if not self.is_recording:
            try:
                print("Kayıt başlatılıyor...")
                self.is_recording = True
                self.stop_identification = False
                self.frames = []
                
                # Ses seviyesi kontrolü için değişken
                self.max_amplitude = 0
                
                devices = sd.query_devices()
                input_device = None
                
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        input_device = i
                        break
                
                if input_device is None:
                    raise Exception("Ses giriş cihazı bulunamadı!")
                
                print(f"Seçilen mikrofon: {devices[input_device]['name']}")
                
                self.stream = sd.InputStream(
                    device=input_device,
                    channels=1,
                    samplerate=44100,
                    blocksize=1024,
                    dtype=np.float32,
                    callback=self.callback
                )
                
                self.stream.start()
                self.root.after(50, self.update_ui)
                
            except Exception as e:
                print(f"Kayıt başlatma hatası: {str(e)}")
                messagebox.showerror("Hata", f"Ses girişi başlatılamadı: {str(e)}")
                self.is_recording = False

    def stop_recording(self):
        try:
            if self.is_recording:
                print("Kayıt durduruluyor...")
                # Önce kayıt durumunu güncelle
                self.is_recording = False
                self.stop_identification = True
                
                # Stream'i güvenli bir şekilde durdur
                if hasattr(self, 'stream'):
                    try:
                        self.stream.stop()
                        self.stream.close()
                        print("Stream kapatıldı")
                    except Exception as e:
                        print(f"Stream kapatma hatası: {str(e)}")

                if len(self.frames) > 0:
                    print(f"Toplam frame sayısı: {len(self.frames)}")
                    self.save_recording()
                    print("Kayıt başarıyla kaydedildi")
                    return True
                else:
                    print("Kayıt için frame bulunamadı!")
                    return False
                    
        except Exception as e:
            print(f"Kayıt durdurma hatası: {str(e)}")
            raise e
        finally:
            # Her durumda stream'i temizlemeye çalış
            if hasattr(self, 'stream'):
                try:
                    self.stream.close()
                except:
                    pass

    def update_final_display(self):
        try:
            # Grafikleri güncelle
            self.plot_signal()
            self.plot_histogram()
            
            # Kaydı kaydet
            self.save_recording()
            
            # GUI'yi güncelle
            self.root.update()
            
        except Exception as e:
            messagebox.showerror("Hata", f"Görüntüleme hatası: {str(e)}")

    def update_info_text(self, tahmin, transcript, kelime_sayisi, emotion_results):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        if transcript == "Could not understand audio":
            transcript = "Ses anlaşılamadı"
        elif "Could not request results" in transcript:
            transcript = "Ses işleme hatası oluştu"
            
        self.info_text.insert(tk.END, "Konuşmacı: ", "header")
        self.info_text.insert(tk.END, f"{tahmin}\n", "content")
        
        self.info_text.insert(tk.END, "Söylenen Kelimeler: ", "header")
        self.info_text.insert(tk.END, f"{transcript}\n", "content")
        
        self.info_text.insert(tk.END, "Kelime Sayısı: ", "header")
        self.info_text.insert(tk.END, f"{kelime_sayisi}\n", "content")
        
        self.info_text.insert(tk.END, "Duygu Analizi: ", "header")
        self.info_text.insert(tk.END, f"{emotion_results}\n", "content")
        
        self.info_text.config(state=tk.DISABLED)

    def callback(self, indata, frames, time, status):
        if status and status.input_overflow:
            return  # Overflow durumunda işlemi atla
            
        try:
            # Maksimum kayıt süresi kontrolü
            MAX_FRAMES = 1300
            
            if len(self.frames) >= MAX_FRAMES:
                print("Maksimum kayıt süresine ulaşıldı...")
                self.root.after(0, self.stop_recording)
                return
            
            # Ses verilerini kaydet
            self.frames.append(indata.copy())
            
            # Her 5 frame'de bir histogramı güncelleme kuyruğuna ekle
            if len(self.frames) % 5 == 0:
                self.update_queue.append(indata)
            
        except Exception as e:
            print(f"Callback hatası: {str(e)}")

    def process_updates(self):
        """Thread güvenli güncelleme işleyicisi"""
        try:
            if not self.is_updating and self.update_queue:
                self.is_updating = True
                new_data = self.update_queue.pop(0)
                self.update_histogram(new_data)
                self.is_updating = False
                
            # Kuyrukta çok fazla veri birikmesin
            if len(self.update_queue) > 10:
                self.update_queue = self.update_queue[-10:]
                
        except Exception as e:
            print(f"Güncelleme işleme hatası: {str(e)}")
        finally:
            # Kayıt devam ediyorsa güncellemeyi sürdür
            if self.is_recording:
                self.root.after(50, self.process_updates)

    def update_histogram(self, new_data):
        try:
            if not self.is_recording:  # Kayıt durmuşsa güncelleme yapma
                return
                
            self.hist_ax.clear()
            self.hist_ax.hist(new_data, bins=50, range=(-1, 1), color='b', alpha=0.7)
            self.hist_ax.set_title('Ses Verisi Dağılımı (Canlı)')
            self.hist_ax.set_xlabel('Amplitüd')
            self.hist_ax.set_ylabel('Frekans')
            
            # Eksenleri sabit tut
            self.hist_ax.set_xlim(-1, 1)
            self.hist_ax.set_ylim(0, 1000)
            
            # Canvası güncelle
            self.hist_canvas.draw()
            
        except Exception as e:
            print(f"Histogram güncelleme hatası: {str(e)}")

    def update_ui(self):
        if self.is_recording:
            self.root.after(50, self.update_ui)  # Güncelleme süresini 50ms'ye düşürdük

    def save_recording(self):
        try:
            if len(self.frames) > 0:
                fs = 44100
                audio_data = np.concatenate(self.frames, axis=0)
                
                print(f"Ses verisi boyutu: {audio_data.shape}")
                
                # Ses dosyasını kaydet
                wav.write("kayit.wav", fs, audio_data)
                print("kayit.wav dosyası oluşturuldu")
                
                # PCM formatına dönüştür
                sound = AudioSegment.from_wav("kayit.wav")
                sound.export("kayit1_pcm.wav", format="wav")
                print("kayit1_pcm.wav dosyası oluşturuldu")
                
        except Exception as e:
            print(f"Kayıt kaydetme hatası: {str(e)}")
            raise e

    def process_recording(self):
        try:
            file = "kayit1_pcm.wav"
            
            if not os.path.exists(file):
                return {
                    "status": "error",
                    "message": f"{file} bulunamadı!"
                }
                
            transcript, kelime_sayisi = self.getWords(file)
            
            # Metni kelimelere ayır
            words = transcript.split()
            
            # Her kelime için konuşmacı tahmini yap
            speaker_counts = {
                "Kerem": 0,
                "Emir": 0,
                "Diğer": 0
            }
            
            # Her kelime için konuşmacı tespiti yap
            for word in words:
                # Kelimeyi ses dosyasına çevir ve tahmin yap
                temp_file = "temp_word.wav"
                try:
                    # Kelimeyi geçici ses dosyasına kaydet
                    self.save_word_as_audio(word, temp_file)
                    
                    # Konuşmacı tahmini yap
                    tahmin = self.speaker_identification(temp_file)
                    speaker_counts[tahmin if tahmin in ["Kerem", "Emir"] else "Diğer"] += 1
                    
                except Exception as e:
                    print(f"Kelime analizi hatası: {str(e)}")
                    continue
                finally:
                    # Geçici dosyayı temizle
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            
            # Toplam kelime sayısını hesapla
            total_words = sum(speaker_counts.values())
            
            # Konuşmacı yüzdelerini hesapla
            speaker_percentages = {}
            if total_words > 0:
                for speaker, count in speaker_counts.items():
                    percentage = (count / total_words) * 100
                    speaker_percentages[speaker] = round(percentage, 1)
            else:
                # Kelime yoksa varsayılan değerler
                speaker_percentages = {
                    "Kerem": 0.0,
                    "Emir": 0.0,
                    "Diğer": 100.0
                }
            
            # En çok konuşan kişiyi tahmin olarak belirle
            tahmin = max(speaker_counts.items(), key=lambda x: x[1])[0]
            
            # Duygu ve konu analizi yap
            emotion_results = self.emotion_analyzer.analyze_emotion(transcript)
            topic_results = self.topic_analyzer.analyze_topics(transcript)
            
            print(f"Transcript sonucu: {transcript}")
            print(f"Kelime sayısı: {kelime_sayisi}")
            print(f"Konuşmacı tahmini: {tahmin}")
            print(f"Konuşmacı dağılımı: {speaker_percentages}")
            print(f"Duygu analizi: {emotion_results}")
            print(f"Konu analizi: {topic_results}")
            
            return {
                "status": "success",
                "speaker": tahmin,
                "wordCount": kelime_sayisi,
                "transcript": transcript,
                "emotions": emotion_results["duygular"],
                "topics": topic_results,
                "speakerPercentages": speaker_percentages
            }
            
        except Exception as e:
            print(f"İşleme hatası: {str(e)}")
            return {
                "status": "error",
                "message": f"İşleme hatası: {str(e)}"
            }

    def getWords(self, file):
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

        # Transcribe the WAV file
        transcript = transcribe_audio(file)
            
        kelimeler.extend(transcript.split())
            
        #print("Transcript:")
        #print(transcript)
            
        #print("Kelime Sayısı:")
        #print(len(kelimeler))
            
        return transcript, len(kelimeler)
        
    def speaker_identification(self, file):
        # Ses dosyasını yükleme ve MFCC özelliklerini çıkarma
        y, sr = librosa.load(file, sr=self.saniye_basina_ornek)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
        mfcc = np.mean(mfcc.T, axis=0)  # Ortalama MFCC vektörü
        
        # Model üzerinden tahmin yapma
        tahmin_indeksi = self.model.predict(mfcc.reshape(1, -1))[0]
        tahmin_isim = self.sinif_isimleri[tahmin_indeksi]
        
        return tahmin_isim
    
    def plot_histogram(self):
        if len(self.frames) > 0:  # Frames listesi boş değilse işlem yap
            audio_data = np.concatenate(self.frames, axis=0)
            self.hist_ax.clear()
            self.hist_ax.hist(audio_data, bins=100, color='b', alpha=0.7)
            self.hist_ax.set_title('Ses Verisi Dağılımı')
            self.hist_ax.set_xlabel('Amplitüd')
            self.hist_ax.set_ylabel('Frekans')
            self.hist_canvas.draw()

    def plot_signal(self):
        if len(self.frames) > 0:  # Frames listesi boş değilse işlem yap
            audio_data = np.concatenate(self.frames, axis=0)
            self.ax.clear()
            self.ax.plot(np.linspace(0, len(audio_data) / 44100, num=len(audio_data)), audio_data)
            self.ax.set_title('Ses Sinyali')
            self.ax.set_xlabel('Zaman (s)')
            self.ax.set_ylabel('Amplitüd')
            self.canvas.draw()

    def get_signal_image(self):
        if len(self.frames) > 0:
            try:
                plt.switch_backend('Agg')  # GUI olmayan backend'e geç
                
                # Yeni bir figure oluştur
                fig = plt.figure(figsize=(8, 4))
                ax = fig.add_subplot(111)
                
                audio_data = np.concatenate(self.frames, axis=0)
                ax.plot(np.linspace(0, len(audio_data) / 44100, num=len(audio_data)), audio_data)
                ax.set_title('Ses Sinyali')
                ax.set_xlabel('Zaman (s)')
                ax.set_ylabel('Amplitüd')
                
                # Figure'ı base64'e çevir
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                plt.close(fig)  # Figure'ı kapat
                buf.seek(0)
                return base64.b64encode(buf.read()).decode('utf-8')
            except Exception as e:
                print(f"Sinyal görüntüsü oluşturma hatası: {str(e)}")
                return None

    def get_histogram_image(self):
        if len(self.frames) > 0:
            try:
                plt.switch_backend('Agg')  # GUI olmayan backend'e geç
                
                # Yeni bir figure oluştur
                fig = plt.figure(figsize=(8, 4))
                ax = fig.add_subplot(111)
                
                audio_data = np.concatenate(self.frames, axis=0)
                ax.hist(audio_data, bins=100, color='b', alpha=0.7)
                ax.set_title('Ses Verisi Dağılımı')
                ax.set_xlabel('Amplitüd')
                ax.set_ylabel('Frekans')
                
                # Figure'ı base64'e çevir
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                plt.close(fig)  # Figure'ı kapat
                buf.seek(0)
                return base64.b64encode(buf.read()).decode('utf-8')
            except Exception as e:
                print(f"Histogram görüntüsü oluşturma hatası: {str(e)}")
                return None

    def save_word_as_audio(self, word, filename):
        """Kelimeyi ses dosyası olarak kaydet"""
        try:
            # Burada text-to-speech kullanarak kelimeyi ses dosyasına çevirebilirsiniz
            # Örnek olarak gTTS kullanabilirsiniz
            from gtts import gTTS
            tts = gTTS(text=word, lang='tr')
            tts.save(filename)
        except Exception as e:
            print(f"Ses dosyası oluşturma hatası: {str(e)}")
            raise e

class TopicAnalyzer:
    def __init__(self):
        # Genişletilmiş Türkçe konu sözlüğü
        self.topic_dict = {
            "teknoloji": [
                "bilgisayar", "internet", "yazılım", "donanım", "teknoloji", "yapay zeka", "robot",
                "uygulama", "program", "sistem", "veri", "kod", "algoritma", "mobil", "web",
                "siber", "güvenlik", "ağ", "bulut", "sunucu", "veritabanı", "programlama"
            ],
            "eğitim": [
                "okul", "öğrenci", "öğretmen", "ders", "sınav", "eğitim", "öğrenim",
                "ödev", "proje", "araştırma", "akademik", "üniversite", "fakülte", "bölüm",
                "kurs", "seminer", "workshop", "laboratuvar", "kütüphane", "bilim"
            ],
            "sağlık": [
                "hastane", "doktor", "sağlık", "hastalık", "tedavi", "ilaç", "muayene",
                "hasta", "hemşire", "ameliyat", "klinik", "tıp", "teşhis", "terapi",
                "psikoloji", "psikiyatri", "diş", "göz", "kalp", "beyin", "spor"
            ],
            "iş_dünyası": [
                "şirket", "i��", "ekonomi", "finans", "para", "yatırım", "borsa",
                "müşteri", "satış", "pazarlama", "reklam", "marka", "proje", "toplantı",
                "yönetim", "strateji", "performans", "hedef", "başarı", "kariyer"
            ],
            "sanat_kültür": [
                "müzik", "resim", "tiyatro", "sinema", "sanat", "konser", "sergi",
                "film", "kitap", "edebiyat", "şiir", "roman", "yazar", "sanatçı",
                "kültür", "festival", "müze", "galeri", "dans", "fotoğraf"
            ],
            "günlük_yaşam": [
                "ev", "aile","ailem", "yemek", "alışveriş", "giyim", "moda", "dekorasyon",
                "tatil", "seyahat", "hobi", "spor", "eğlence", "arkadaş", "sosyal",
                "hava", "trafik", "ulaşım", "zaman", "plan", "organizasyon"
            ],
            "bilim": [
                "fizik", "kimya", "biyoloji", "matematik", "astronomi", "uzay",
                "araştırma", "deney", "teori", "formül", "element", "molekül",
                "enerji", "atom", "genetik", "evrim", "çevre", "ekosistem"
            ],
            "spor": [
                "futbol", "basketbol", "voleybol", "tenis", "yüzme", "koşu",
                "antrenman", "maç", "turnuva", "şampiyona", "takım", "oyuncu",
                "teknik", "taktik", "fitness", "egzersiz", "performans", "yarış"
            ],
            "politika": [
                "siyaset", "hükümet", "meclis", "parti", "seçim", "politika",
                "demokrasi", "hukuk", "adalet", "kanun", "yasa", "devlet",
                "vatandaş", "toplum", "reform", "karar", "lider", "bakan"
            ],
            "çevre": [
                "doğa", "çevre", "iklim", "hava", "su", "toprak", "orman",
                "deniz", "hayvan", "bitki", "ekosistem", "kirlilik", "geri dönüşüm",
                "yenilenebilir", "enerji", "sürdürülebilirlik", "koruma"
            ]
        }

    def analyze_topics(self, text):
        try:
            if not text or text == "Ses anlaşılamadı":
                return []

            text = text.lower()
            words = text.split()
            
            # Her konu için eşleşme sayısını ve detayları tut
            topic_matches = {
                topic: {
                    "count": 0,
                    "matched_words": [],
                    "other_words": []
                } for topic in self.topic_dict.keys()
            }
            
            # Her kelime için tüm konulardaki eşleşmeleri kontrol et
            for word in words:
                found_in_topics = []
                for topic, keywords in self.topic_dict.items():
                    if word in keywords:
                        topic_matches[topic]["count"] += 1
                        topic_matches[topic]["matched_words"].append(word)
                        found_in_topics.append(topic)
                
                # Eğer kelime birden fazla konuda geçiyorsa, her konunun other_words listesine ekle
                if len(found_in_topics) > 1:
                    for topic in found_in_topics:
                        for other_topic in found_in_topics:
                            if other_topic != topic:
                                topic_matches[topic]["other_words"].append(
                                    f"{word} ({getTopicName(other_topic)})"
                                )

            # Konuları skorlarına göre sırala ve detaylı bilgi ekle
            sorted_topics = []
            total_matches = sum(matches["count"] for matches in topic_matches.values())
            
            for topic, matches in topic_matches.items():
                if matches["count"] > 0:
                    # Eğer başka konularla ortak kelime yoksa skor 100
                    if not matches["other_words"]:
                        score = 100.0
                    else:
                        # Ortak kelimeler varsa, kelime sayısına göre orantılı hesapla
                        score = (matches["count"] / total_matches) * 100

                    sorted_topics.append({
                        "konu": topic,
                        "skor": round(score, 1),
                        "eşleşen_kelimeler": list(set(matches["matched_words"])),
                        "ortak_kelimeler": list(set(matches["other_words"])),
                        "kelime_sayısı": matches["count"]
                    })

            # Skorlarına göre sırala
            sorted_topics.sort(key=lambda x: (x["kelime_sayısı"], x["skor"]), reverse=True)
            
            # Toplam kelime sayısına göre yüzdeleri yeniden hesapla
            if len(sorted_topics) > 0:
                total_words = sum(topic["kelime_sayısı"] for topic in sorted_topics)
                for topic in sorted_topics:
                    if total_words > 0:
                        topic["skor"] = round((topic["kelime_sayısı"] / total_words) * 100, 1)

            return sorted_topics

        except Exception as e:
            print(f"Konu analizi hatası: {str(e)}")
            return []

def getTopicName(topic):
    topic_names = {
        'teknoloji': 'Teknoloji',
        'eğitim': 'Eğitim',
        'sağlık': 'Sağlık',
        'iş_dünyası': 'İş Dünyası',
        'sanat_kültür': 'Sanat ve Kültür',
        'günlük_yaşam': 'Günlük Yaşam',
        'bilim': 'Bilim',
        'spor': 'Spor',
        'politika': 'Politika',
        'çevre': 'Çevre'
    }
    return topic_names.get(topic, topic)