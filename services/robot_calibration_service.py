# services/robot_calibration_service.py
import cv2
import numpy as np
from pydantic import BaseModel
from typing import List, Tuple


class Coordinate(BaseModel):
    x: float
    y: float


def preprocess_image(image_data: bytes) -> np.ndarray:
    """Đọc và xử lý ảnh từ bytes"""
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Could not load image")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image)
    blurred = cv2.GaussianBlur(clahe_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    return edges


def find_contour_centers(edges: np.ndarray) -> np.ndarray:
    """Tìm và sắp xếp các tâm của contours"""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000:
            ((x, y), _) = cv2.minEnclosingCircle(contour)
            centers.append([int(x), int(y)])

    if len(centers) != 9:
        raise ValueError(f"Found {len(centers)} contours instead of 9")

    points = np.array(centers, dtype=np.float32)
    points = points[points[:, 1].argsort()]
    sorted_points = []
    for i in range(0, 9, 3):
        row = points[i:i + 3]
        row = row[row[:, 0].argsort()]
        sorted_points.extend(row)

    return np.array(sorted_points, dtype=np.float32)


def calculate_homography(cam_points: np.ndarray, robot_points: List[Coordinate]) -> Tuple[np.ndarray, dict]:
    """
    Tính ma trận homography và sai số giữa tọa độ robot thực tế và dự đoán.

    Args:
        cam_points: Tọa độ camera (numpy array, shape (9, 2)).
        robot_points: Tọa độ robot thực tế (List[Coordinate]).

    Returns:
        Tuple containing:
        - H: Ma trận homography (numpy array, shape (3, 3)).
        - errors: Dictionary chứa sai số (max, mean, và chi tiết từng điểm).
    """
    # Chuyển robot_points thành numpy array
    robot_array = np.array([[p.x, p.y] for p in robot_points], dtype=np.float32)
    if len(robot_array) != 9:
        raise ValueError("Must provide exactly 9 robot coordinates")

    # Tính ma trận homography
    H, _ = cv2.findHomography(cam_points, robot_array)

    # Tính tọa độ robot dự đoán bằng cách áp dụng H lên cam_points
    predicted_points = []
    errors = []
    for i in range(len(cam_points)):
        # Chuyển tọa độ camera thành tọa độ đồng nhất (homogeneous coordinates)
        cam_point = np.array([[cam_points[i][0], cam_points[i][1], 1]], dtype=np.float32).T

        # Ánh xạ sang hệ robot
        robot_point = H @ cam_point
        robot_point = robot_point / robot_point[2]  # Chuẩn hóa tọa độ đồng nhất

        # Lấy tọa độ x, y
        predicted_x, predicted_y = robot_point[0, 0], robot_point[1, 0]
        predicted_points.append([predicted_x, predicted_y])

        # Tính sai số Euclidean giữa tọa độ dự đoán và thực tế
        actual_x, actual_y = robot_array[i][0], robot_array[i][1]
        error = np.sqrt((predicted_x - actual_x) ** 2 + (predicted_y - actual_y) ** 2)
        errors.append(error)

    # Tính sai số trung bình và sai số tối đa
    mean_error = np.mean(errors)
    max_error = np.max(errors)

    # Tạo dictionary chứa thông tin sai số
    error_info = {
        "errors_per_point": errors,  # Sai số từng điểm
        "mean_error": mean_error,  # Sai số trung bình
        "max_error": max_error  # Sai số tối đa
    }

    return H, error_info