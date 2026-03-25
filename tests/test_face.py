import unittest
import numpy as np
from src.face_process import detect_and_encode_faces
import cv2

class TestFaceProcess(unittest.TestCase):
    def test_detect_faces(self):
        # 读取测试图片
        img = cv2.imread("src/example_images/test1.jpg")
        face_locations, face_encodings = detect_and_encode_faces(img)
        # 验证检测结果
        self.assertGreater(len(face_locations), 0)
        self.assertEqual(len(face_encodings[0]), 128)

if __name__ == "__main__":
    unittest.main()
