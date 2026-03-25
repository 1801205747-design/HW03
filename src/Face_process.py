import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import cv2

def load_known_faces(known_face_paths: dict) -> tuple:
    """加载已知人脸库，返回特征编码和对应名称"""
    known_face_encodings = []
    known_face_names = []
    for name, path in known_face_paths.items():
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
    return known_face_encodings, known_face_names

def detect_and_encode_faces(image: np.ndarray) -> tuple:
    """检测人脸并生成128维特征编码"""
    # 转换色彩空间（face_recognition用RGB，OpenCV用BGR）
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 检测人脸位置
    face_locations = face_recognition.face_locations(rgb_image)
    # 生成128维特征编码
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return face_locations, face_encodings

def recognize_faces(face_encodings: list, known_encodings: list, known_names: list) -> list:
    """与人脸库比对，返回识别结果"""
    face_names = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"
        # 取最相似的人脸
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
        face_names.append(name)
    return face_names

def draw_face_boxes(image: np.ndarray, face_locations: list, face_names: list) -> Image:
    """在图片上框出人脸并标注名称"""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # 绘制矩形框
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=2)
        # 绘制名称背景
        draw.rectangle(((left, bottom - 35), (right, bottom)), fill=(0, 255, 0), outline=(0, 255, 0))
        draw.text((left + 6, bottom - 30), name, fill=(255, 255, 255), font_size=16)
    return pil_image