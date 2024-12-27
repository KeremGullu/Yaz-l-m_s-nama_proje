import unittest
import tkinter as tk
from UserInterface import AudioRecorderUI

class TestGUI(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.app = AudioRecorderUI(self.root)

    def tearDown(self):
        self.root.destroy()

    def test_initial_state(self):
        """GUI başlangıç durumunu test et"""
        self.assertTrue(self.app.stop_button["state"] == "disabled")
        self.assertTrue(self.app.process_button["state"] == "disabled")
        self.assertFalse(self.app.is_recording)

    def test_button_states(self):
        """Buton durumlarının değişimini test et"""
        # Kayıt başlat
        self.app.start_recording()
        self.assertTrue(self.app.is_recording)
        self.assertTrue(self.app.start_button["state"] == "disabled")
        self.assertTrue(self.app.stop_button["state"] == "normal")

        # Kayıt durdur
        self.app.stop_recording()
        self.assertFalse(self.app.is_recording)
        self.assertTrue(self.app.start_button["state"] == "normal")
        self.assertTrue(self.app.process_button["state"] == "normal")

if __name__ == '__main__':
    unittest.main() 