from fastapi import APIRouter, UploadFile, File
import time
from services.template_matching_service import process_template_matching
import cv2
import numpy as np
import matplotlib.pyplot as plt

router = APIRouter(prefix="/api", tags=["Template Matching"])  # Thêm prefix và tags để tổ chức API


@router.post("/match-template/")
async def match_template(
        image: UploadFile = File(..., description="The larger image to search in"),
        template: UploadFile = File(..., description="The template image to match"),
        threshold: float = 0.7  # Ngưỡng độ tương đồng, mặc định 0.8 (0-1)
):
    """
    Upload an image and a template image, then perform template matching.
    Returns a JSON string with the number of matched objects and their details (position, angle, scale).
    """
    # Đọc file ảnh lớn và template
    image_bytes = await image.read()
    template_bytes = await template.read()

    start = time.time()
    print(threshold)
    # Xử lý ảnh với Template Matching
    result = process_template_matching(
        image_bytes=image_bytes,
        template_bytes=template_bytes,
        threshold=threshold,
        rot_range=[0, 360],
        rot_interval=2,
        scale_range=[98, 102],
        scale_interval=2,
        min_contour_area=150,
        roi_expand=40,
        max_workers=None  # Có thể điều chỉnh theo CPU (ví dụ: 6 cho i5-13400F)
    )
    print(f"Processing time: {(time.time() - start)} seconds")




    # Chuẩn bị ảnh để vẽ và hiển thị
    image_array = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # Chuyển từ BGR (OpenCV) sang RGB (matplotlib)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    template_array = np.frombuffer(template_bytes, np.uint8)
    template_bgr = cv2.imdecode(template_array, cv2.IMREAD_COLOR)
    height, width = template_bgr.shape[:2]

    # Tạo figure và axes
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Vẽ các hình chữ nhật xoay dựa trên kết quả
    for match in result["matches"]:
        x = match["x"]
        y = match["y"]
        angle = match["angle"]
        scale = match["scale"]

        # Tính kích thước dựa trên scale
        scaled_width = width * scale / 100
        scaled_height = height * scale / 100

        # Tâm xoay
        center_x = x + scaled_width / 2
        center_y = y + scaled_height / 2

        # Ma trận xoay (tính toán thủ công vì matplotlib không có getRotationMatrix2D)
        theta = np.radians(-angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Các đỉnh của hình chữ nhật
        rect_points = np.array([
            [x, y],
            [x + scaled_width, y],
            [x + scaled_width, y + scaled_height],
            [x, y + scaled_height]
        ])

        # Chuyển các điểm về tâm xoay
        rect_points -= [center_x, center_y]

        # Xoay các đỉnh
        rotated_points = np.zeros_like(rect_points)
        rotated_points[:, 0] = rect_points[:, 0] * cos_theta - rect_points[:, 1] * sin_theta
        rotated_points[:, 1] = rect_points[:, 0] * sin_theta + rect_points[:, 1] * cos_theta

        # Chuyển lại vị trí
        rotated_points += [center_x, center_y]

        # Đóng vòng các điểm
        rotated_points = np.append(rotated_points, [rotated_points[0]], axis=0)

        # Vẽ hình chữ nhật xoay
        ax.plot(rotated_points[:, 0], rotated_points[:, 1],
                color='green', linewidth=2)

    # Hiển thị ảnh gốc
    plt.imshow(image_rgb)
    plt.title("Matched Results")
    plt.axis('off')  # Tắt trục tọa độ

    # Hiển thị kết quả
    plt.show()
    print("Done")


    # Trả về kết quả dưới dạng JSON
    return result