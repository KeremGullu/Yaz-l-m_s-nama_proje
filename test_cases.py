import unittest
import os
import numpy as np
from UserInterface import EmotionAnalyzer, TopicAnalyzer

class TestSpeakerRecognition(unittest.TestCase):
    def setUp(self):
        """Her test öncesi çalışacak hazırlık fonksiyonu"""
        self.emotion_analyzer = EmotionAnalyzer()
        self.topic_analyzer = TopicAnalyzer()

    def test_case_01_emotion_analysis(self):
        """
        Test Case ID: TC_01
        Test Case Name: Duygu Analizi Testi
        Objective: Duygu analizi fonksiyonunun doğru çalıştığını kontrol etme
        """
        test_text = "Bugün çok mutluyum ve harika bir gün geçirdim"
        result = self.emotion_analyzer.analyze_emotion(test_text)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertAlmostEqual(sum(result["duygular"].values()), 100.0, places=1)
        self.assertIn(result["baskın_duygu"], result["duygular"].keys())
        self.assertTrue(0 <= result["güven_skoru"] <= 1)

    def test_case_02_topic_analysis(self):
        """
        Test Case ID: TC_02
        Test Case Name: Konu Analizi Testi
        Objective: Konu analizi fonksiyonunun doğru çalıştığını kontrol etme
        """
        test_text = "Bilgisayar programlama ve yazılım geliştirme konularında çalışıyorum"
        results = self.topic_analyzer.analyze_topics(test_text)
        
        # Assertions
        self.assertTrue(len(results) > 0)
        for topic in results:
            self.assertTrue(0 <= topic["skor"] <= 100)
            self.assertTrue(len(topic["eşleşen_kelimeler"]) > 0)

    def test_case_03_empty_input(self):
        """
        Test Case ID: TC_03
        Test Case Name: Boş Girdi Testi
        Objective: Boş veya geçersiz girdilerin doğru işlendiğini kontrol etme
        """
        # Boş metin testi
        empty_emotion = self.emotion_analyzer.analyze_emotion("")
        empty_topic = self.topic_analyzer.analyze_topics("")
        
        # None testi
        none_emotion = self.emotion_analyzer.analyze_emotion(None)
        none_topic = self.topic_analyzer.analyze_topics(None)
        
        # Assertions
        self.assertEqual(sum(empty_emotion["duygular"].values()), 100.0)
        self.assertEqual(empty_emotion["baskın_duygu"], "belirsiz")
        self.assertEqual(len(empty_topic), 0)
        
        self.assertEqual(sum(none_emotion["duygular"].values()), 100.0)
        self.assertEqual(none_emotion["baskın_duygu"], "belirsiz")
        self.assertEqual(len(none_topic), 0)

    def test_case_04_mixed_emotions(self):
        """
        Test Case ID: TC_04
        Test Case Name: Karışık Duygular Testi
        Objective: Birden fazla duygu içeren metinlerin analizini kontrol etme
        """
        test_text = "Bugün çok mutluyum ama aynı zamanda biraz endişeliyim ve şaşkınım"
        result = self.emotion_analyzer.analyze_emotion(test_text)
        
        # Assertions
        self.assertGreater(result["duygular"]["mutlu"], 0)
        self.assertGreater(result["duygular"]["şaşkın"], 0)
        self.assertEqual(sum(result["duygular"].values()), 100.0)

    def test_case_05_multiple_topics(self):
        """
        Test Case ID: TC_05
        Test Case Name: Çoklu Konu Testi
        Objective: Birden fazla konu içeren metinlerin analizini kontrol etme
        """
        test_text = "Bilgisayarda futbol oyunu oynuyorum ve spor yapıyorum"
        results = self.topic_analyzer.analyze_topics(test_text)
        
        # En az iki konu bulunmalı (teknoloji ve spor)
        found_topics = [topic["konu"] for topic in results]
        
        # Assertions
        self.assertGreaterEqual(len(results), 2)
        self.assertTrue("teknoloji" in found_topics or "spor" in found_topics)
        self.assertEqual(sum(topic["skor"] for topic in results), 100.0)

if __name__ == '__main__':
    unittest.main() 