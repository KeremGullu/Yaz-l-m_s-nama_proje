from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import unittest
import time
import os
from datetime import datetime

class TestBirdRegistration(unittest.TestCase):
    def setUp(self):
        """Test öncesi hazırlık"""
        # Chrome ayarlarını yapılandır
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--ignore-ssl-errors')
        chrome_options.add_argument('--start-maximized')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.get("https://testbirds.com/")  # Doğru URL'yi kullan
        self.wait = WebDriverWait(self.driver, 20)
        
        # Screenshot klasörü oluştur
        self.screenshot_dir = "test_screenshots"
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)

    def test_registration_and_confirmation(self):
        """Test Bird Kayıt Testi"""
        try:
            # Cookie banner'ı kabul et (varsa)
            try:
                cookie_button = self.wait.until(
                    EC.element_to_be_clickable((By.ID, "CookieBoxSaveButton"))
                )
                cookie_button.click()
                time.sleep(1)
            except:
                print("Cookie banner bulunamadı")

            # "Become a Tester" butonunu bul
            try:
                tester_button = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Become a Tester')]"))
                )
                tester_button.click()
            except:
                print("Become a Tester butonu bulunamadı, alternatif arıyorum...")
                try:
                    tester_button = self.wait.until(
                        EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, '/become-a-tester')]"))
                    )
                    tester_button.click()
                except:
                    self.fail("Kayıt sayfasına ulaşılamadı")

            self.take_screenshot("1_tester_page")
            time.sleep(2)

            # Kayıt formunu doldur
            try:
                # İsim
                first_name = self.wait.until(
                    EC.presence_of_element_located((By.NAME, "firstName"))
                )
                first_name.send_keys("Kerem")

                # Soyisim
                last_name = self.driver.find_element(By.NAME, "lastName")
                last_name.send_keys("Güllü")

                # Email
                email = self.driver.find_element(By.NAME, "email")
                email.send_keys("kerengullu@gmail.com")

                # Şifre
                password = self.driver.find_element(By.NAME, "password")
                password.send_keys("kerem22082002")

                self.take_screenshot("2_form_filled")

                # Kullanım şartlarını kabul et
                terms = self.driver.find_element(By.NAME, "terms")
                terms.click()

                # Newsletter checkbox (opsiyonel)
                try:
                    newsletter = self.driver.find_element(By.NAME, "newsletter")
                    newsletter.click()
                except:
                    print("Newsletter checkbox bulunamadı")

                self.take_screenshot("3_before_submit")

                # Formu gönder
                submit_button = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))
                )
                submit_button.click()
                time.sleep(3)

                self.take_screenshot("4_after_submit")

            except Exception as e:
                print(f"Form doldurma hatası: {str(e)}")
                raise

            # Test raporunu oluştur
            with open(f"{self.screenshot_dir}/test_report.txt", "w", encoding='utf-8') as f:
                f.write("Test Bird Kayıt Testi\n")
                f.write(f"Test Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Durum: Başarılı\n")
                f.write("Ekran Görüntüleri:\n")
                for i in range(1, 5):
                    f.write(f"{i}. Adım: {self.screenshot_dir}/step_{i}.png\n")

        except Exception as e:
            self.take_screenshot("error")
            print(f"Test hatası: {str(e)}")
            self.fail(f"Test başarısız oldu. Detaylar için error.png ve konsol çıktısını kontrol edin.")

    def take_screenshot(self, name):
        """Ekran görüntüsü alma"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.screenshot_dir}/{name}_{timestamp}.png"
            self.driver.save_screenshot(filename)
            return filename
        except Exception as e:
            print(f"Screenshot alma hatası: {str(e)}")
            return None

    def tearDown(self):
        """Test sonrası temizlik"""
        if self.driver:
            self.driver.quit()

if __name__ == '__main__':
    unittest.main() 