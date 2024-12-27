from flask import Flask, render_template, jsonify, render_template_string
import tkinter as tk
from UserInterface import AudioRecorderUI
import threading
import numpy as np
from sonar_dashboard import SonarDashboard

app = Flask(__name__)

# Global değişkenler
root = tk.Tk()
audio_recorder = AudioRecorderUI(root)
root.withdraw()  # Tkinter penceresini gizle

def update_tk():
    while True:
        try:
            root.update()
        except:
            break

# Tkinter güncellemesi için thread başlat
tk_thread = threading.Thread(target=update_tk, daemon=True)
tk_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    try:
        audio_recorder.start_recording()
        return jsonify({"status": "success", "message": "Kayıt başlatıldı"})
    except Exception as e:
        print(f"Kayıt başlatma hatası: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    try:
        success = audio_recorder.stop_recording()
        if success:
            return jsonify({
                "status": "success", 
                "message": "Kayıt durduruldu"
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "Kayıt durdurulamadı - frame bulunamadı"
            })
    except Exception as e:
        print(f"Kayıt durdurma hatası: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"Kayıt durdurma hatası: {str(e)}"
        })

@app.route('/process_recording', methods=['POST'])
def process_recording():
    try:
        results = audio_recorder.process_recording()
        return jsonify(results)
    except Exception as e:
        print(f"İşleme hatası: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_visualization', methods=['GET'])
def get_visualization():
    try:
        # Ses ve histogram verilerini al
        if len(audio_recorder.frames) > 0:
            audio_data = np.concatenate(audio_recorder.frames, axis=0)
            
            # Base64 formatında görüntüleri döndür
            signal_img = audio_recorder.get_signal_image()
            histogram_img = audio_recorder.get_histogram_image()
            
            return jsonify({
                "status": "success",
                "signal": signal_img,
                "histogram": histogram_img
            })
    except Exception as e:
        print(f"Görselleştirme hatası: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/sonar-dashboard')
def show_sonar_dashboard():
    try:
        dashboard = SonarDashboard(
            sonar_url='http://localhost:9000',  # SonarQube sunucu adresi
            project_key='speaker_recognition'    # sonar-project.properties'deki proje anahtarı
        )
        html_report = dashboard.generate_html_report()
        return render_template_string(html_report)
    except Exception as e:
        print(f"SonarQube dashboard hatası: {str(e)}")
        return """
        <div class="dashboard">
            <div class="widget">
                <h3>Hata</h3>
                <div class="metric">
                    SonarQube metriklerine erişilemedi. Lütfen SonarQube sunucusunun çalıştığından emin olun.
                </div>
            </div>
        </div>
        """

if __name__ == '__main__':
    app.run(debug=False, threaded=True) 