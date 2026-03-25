import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from face_process import load_known_faces, detect_and_encode_faces, recognize_faces, draw_face_boxes

# 配置页面
st.set_page_config(page_title="人脸检测与识别", page_icon="👤", layout="wide")
st.title("人脸检测与识别系统")

# 加载已知人脸库（示例：可在example_images/下放自己的人脸图）
known_face_paths = {
    "Test1": "src/example_images/test1.jpg",
    "Test2": "src/example_images/test2.jpg"
}
known_encodings, known_names = load_known_faces(known_face_paths)

# 侧边栏：上传/选择图片
st.sidebar.header("图片来源")
uploaded_file = st.sidebar.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
example_images = [f for f in os.listdir("src/example_images/") if f.endswith((".jpg", ".jpeg", ".png"))]
selected_example = st.sidebar.selectbox("选择示例图片", [""] + example_images)

# 处理图片
if uploaded_file is not None or selected_example:
    # 读取图片
    if uploaded_file:
        image = Image.open(uploaded_file)
    else:
        image = Image.open(os.path.join("src/example_images/", selected_example))
    img_array = np.array(image)
    st.subheader("原始图片")
    st.image(image, use_column_width=True)

    # 人脸检测与编码
    with st.spinner("检测人脸中..."):
        face_locations, face_encodings = detect_and_encode_faces(img_array)
        st.write(f"检测到 **{len(face_locations)}** 个人脸")
        st.write(f"人脸特征编码维度：{len(face_encodings[0]) if face_encodings else 0} 维（128维）")

    # 人脸识别（可选）
    if st.checkbox("开启人脸识别", value=True):
        with st.spinner("识别中..."):
            face_names = recognize_faces(face_encodings, known_encodings, known_names)
            # 绘制标注图
            annotated_image = draw_face_boxes(img_array, face_locations, face_names)
            st.subheader("识别结果")
            st.image(annotated_image, use_column_width=True)
            # 展示识别名称
            st.write("识别出的人脸：", ", ".join(set(face_names)) if face_names else "无")
else:
    st.info("请在侧边栏上传图片或选择示例图片开始检测")