from fastapi import APIRouter, UploadFile, File
import time
from services.template_matching_service import process_template_matching
import cv2
import numpy as np

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

    template_array = np.frombuffer(template_bytes, np.uint8)
    template_bgr = cv2.imdecode(template_array, cv2.IMREAD_COLOR)
    height, width = template_bgr.shape[:2]

    # # Vẽ các hình chữ nhật xoay dựa trên kết quả
    # for match in result["matches"]:
    #     x = match["x"]
    #     y = match["y"]
    #     angle = match["angle"]
    #     scale = match["scale"]
    #
    #     # Tính kích thước dựa trên scale
    #     scaled_width = width * scale / 100
    #     scaled_height = height * scale / 100
    #
    #     # Tâm xoay
    #     center_x = x + scaled_width / 2
    #     center_y = y + scaled_height / 2
    #
    #     # Ma trận xoay
    #     rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
    #
    #     # Các đỉnh của hình chữ nhật
    #     rect_points = np.array([
    #         [x, y],
    #         [x + scaled_width, y],
    #         [x + scaled_width, y + scaled_height],
    #         [x, y + scaled_height]
    #     ], dtype=np.float32)
    #
    #     # Xoay các đỉnh
    #     rotated_points = cv2.transform(np.array([rect_points]), rotation_matrix)[0]
    #
    #     # Vẽ hình chữ nhật xoay bằng các đường thẳng
    #     for i in range(4):
    #         cv2.line(image_bgr,
    #                  tuple(rotated_points[i].astype(int)),
    #                  tuple(rotated_points[(i + 1) % 4].astype(int)),
    #                  (0, 255, 0), 2)
    #
    # # Hiển thị ảnh để kiểm tra
    # cv2.imshow("Matched Results", image_bgr)
    # cv2.waitKey(5000)  # Giữ cửa sổ mở trong 5 giây (5000 ms)
    # cv2.destroyAllWindows()  # Đóng cửa sổ sau khi kiểm tra
    print("Done")


    # Trả về kết quả dưới dạng JSON
    return result