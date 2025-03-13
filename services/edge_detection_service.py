import cv2
import numpy as np
from io import BytesIO

def process_canny_edge(image_bytes: bytes, threshold1: int = 100, threshold2: int = 200) -> bytes:
    # Chuyển bytes thành mảng NumPy
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    # Áp dụng Canny Edge Detection
    edges = cv2.Canny(image, threshold1, threshold2)

    # Chuyển kết quả về dạng bytes
    _, buffer = cv2.imencode(".png", edges)
    return buffer.tobytes()
