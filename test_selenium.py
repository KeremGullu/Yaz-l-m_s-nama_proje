from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import unittest
import time

class TestWebInterface(unittest.TestCase):
    def setUp(self):
        """Her test öncesi çalışacak"""
        self.driver = webdriver.Chrome()  # Chrome driver'ı başlat
        self.driver.get("http://localhost:5000")  # Flask uygulamasının adresi
        self.wait = WebDriverWait(self.driver, 10)

    def tearDown(self):
        """Her test sonrası çalışacak"""
        self.driver.quit()

    def test_case_01_record_button_states(self):
        """
        Test Case ID: TC_WEB_01
        Test Case Name: Kayıt Butonları Durum Kontrolü
        Objective: Kayıt butonlarının doğru durumda olduğunu kontrol etme
        """
        try:
            # Başlangıç durumu kontrolü
            start_button = self.wait.until(
                EC.presence_of_element_located((By.ID, "startButton"))
            )
            stop_button = self.driver.find_element(By.ID, "stopButton")
            process_button = self.driver.find_element(By.ID, "processButton")

            # Başlangıçta stop ve process butonları disabled olmalı
            self.assertFalse(start_button.get_property("disabled"))
            self.assertTrue(stop_button.get_property("disabled"))
            self.assertTrue(process_button.get_property("disabled"))

            # Kayıt başlat
            start_button.click()
            time.sleep(1)  # Butonların güncellenmesi için bekle

            # Kayıt başladıktan sonra start butonu disabled, stop butonu enabled olmalı
            self.assertTrue(start_button.get_property("disabled"))
            self.assertFalse(stop_button.get_property("disabled"))
            self.assertTrue(process_button.get_property("disabled"))

        except Exception as e:
            self.fail(f"Test başarısız: {str(e)}")

    def test_case_02_visualization_update(self):
        """
        Test Case ID: TC_WEB_02
        Test Case Name: Görselleştirme Güncelleme Kontrolü
        Objective: Kayıt sırasında görselleştirmelerin güncellendiğini kontrol etme
        """
        try:
            # Kayıt başlat
            start_button = self.wait.until(
                EC.presence_of_element_located((By.ID, "startButton"))
            )
            start_button.click()
            time.sleep(2)  # Görselleştirmelerin güncellenmesi için bekle

            # Görselleştirme elementlerini kontrol et
            waveform = self.driver.find_element(By.ID, "waveform")
            histogram = self.driver.find_element(By.ID, "histogram")

            # Görselleştirmelerin içinde img elementi olmalı
            self.assertTrue(len(waveform.find_elements(By.TAG_NAME, "img")) > 0)
            self.assertTrue(len(histogram.find_elements(By.TAG_NAME, "img")) > 0)

            # Kayıt durdur
            stop_button = self.driver.find_element(By.ID, "stopButton")
            stop_button.click()

        except Exception as e:
            self.fail(f"Test başarısız: {str(e)}")

    def test_case_03_emotion_analysis(self):
        """
        Test Case ID: TC_WEB_03
        Test Case Name: Duygu Analizi Kontrolü
        Objective: Duygu analizi sonuçlarının doğru gösterildiğini kontrol etme
        """
        try:
            # Kayıt yap ve durdur
            start_button = self.wait.until(
                EC.presence_of_element_located((By.ID, "startButton"))
            )
            start_button.click()
            time.sleep(2)
            
            stop_button = self.driver.find_element(By.ID, "stopButton")
            stop_button.click()
            time.sleep(1)

            # İşleme butonuna tıkla
            process_button = self.driver.find_element(By.ID, "processButton")
            process_button.click()
            time.sleep(2)

            # Duygu skorlarını kontrol et
            emotion_scores = self.driver.find_elements(By.CLASS_NAME, "emotion-score")
            self.assertEqual(len(emotion_scores), 5)  # 5 duygu kategorisi olmalı

            # Progress bar'ların varlığını kontrol et
            progress_bars = self.driver.find_elements(By.CLASS_NAME, "progress-bar")
            self.assertEqual(len(progress_bars), 5)

            # Toplam duygu yüzdesi 100 olmalı
            total_score = 0
            for score in self.driver.find_elements(By.CLASS_NAME, "score-value"):
                value = float(score.text.strip('%'))
                total_score += value
            
            self.assertAlmostEqual(total_score, 100.0, delta=0.1)

        except Exception as e:
            self.fail(f"Test başarısız: {str(e)}")

if __name__ == '__main__':
    unittest.main() 