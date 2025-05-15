import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from shapely.geometry import Polygon
from shapely.affinity import rotate, scale
from utils.InvariantTM import invariant_match_template, auto_canny
import logging

# Cấu hình mặc định với các giá trị gán cứng từ code test
CONFIG = {
    'clahe_clip_limit': 2.0,
    'clahe_tile_grid_size': (8, 8),
    'canny_sigma': 0.33,
    'morph_kernel_size': (3, 3),
    'min_area_ratio': 0.7,
    'min_contour_size_ratio': 0.4,
    'roi_expand_ratio': 0.03,
    'roi_min_expand': 15,
    'rotation_range': [-180, 180],
    'rotation_interval': 1,
    'scale_range': [100, 101],
    'scale_interval': 1,
}

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_rois_threshold(image: np.ndarray, edges: np.ndarray, template_shape: tuple, config: dict) -> list:
    """
    Tìm các vùng ROI dựa trên thresholding và contours.

    Args:
        image (np.ndarray): Ảnh gốc ở định dạng grayscale.
        edges (np.ndarray): Ảnh biên (không dùng trong phiên bản này, giữ để tương thích).
        template_shape (tuple): Kích thước của template (height, width).
        config (dict): Cấu hình tham số.

    Returns:
        list: Danh sách các ROI (top_left, bottom_right).
    """
    height, width = template_shape
    min_area_threshold = config['min_area_ratio'] * height * width
    min_size = min(image.shape[0], image.shape[1]) * config['min_contour_size_ratio']

    # Tính độ sáng trung bình để xác định nền sáng hay tối
    mean_intensity = np.mean(image)

    # Làm mịn ảnh để giảm nhiễu
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Áp dụng thresholding dựa trên nền sáng hoặc tối
    if mean_intensity > 127:
        # Nền sáng: Lấy đối tượng tối
        _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Nền tối: Lấy đối tượng sáng
        _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Đóng các vùng đứt để cải thiện contours
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Tìm contours trong mask nhị phân
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_size:
            x, y, w, h = cv2.boundingRect(contour)
            expand = max(config['roi_min_expand'], int(max(w, h) * config['roi_expand_ratio']))
            top_left = (max(0, x - expand), max(0, y - expand))
            bottom_right = (min(image.shape[1], x + w + expand), min(image.shape[0], y + h + expand))
            width_roi = bottom_right[0] - top_left[0]
            height_roi = bottom_right[1] - top_left[1]

            # Lọc ROI dựa trên kích thước template
            if height_roi * width_roi >= min_area_threshold:
                # Kiểm tra thêm: ROI phải đủ lớn để chứa template
                if width_roi >= width and height_roi >= height:
                    rois.append((top_left, bottom_right))

    logging.info(f"Tìm thấy {len(rois)} ROIs. Using threshold")
    return rois

def process_roi(roi_info: tuple, img: np.ndarray, template: np.ndarray, threshold: float) -> list:
    """Xử lý template matching cho một ROI."""
    top_left, bottom_right = roi_info
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Đảm bảo tọa độ hợp lệ
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return []

    roi = img[y1:y2, x1:x2]
    height, width = template.shape[:2]

    # Kiểm tra kích thước ROI
    if roi.shape[0] < height or roi.shape[1] < width:
        return []

    points_list = invariant_match_template(
        img_gray=roi,
        template_gray=template,
        method="TM_CCOEFF_NORMED",
        matched_thresh=threshold,
        rot_range=CONFIG['rotation_range'],
        rot_interval=CONFIG['rotation_interval'],
        scale_range=CONFIG['scale_range'],
        scale_interval=CONFIG['scale_interval'],
        rm_redundant=True,
        minmax=True
    )

    # Điều chỉnh tọa độ về ảnh gốc
    adjusted_points = []
    for point_info in points_list:
        point = point_info[0]
        adjusted_point = (point[0] + x1, point[1] + y1)
        adjusted_points.append((adjusted_point, point_info[1], point_info[2], point_info[3]))
    return adjusted_points

def process_template_matching(image_bytes: bytes,
                             template_bytes: bytes,
                             threshold: float = 0.4,
                             edge_base: bool = True,
                             check_overlap: bool = False) -> dict:
    """
    Perform invariant template matching with preprocessing, ROI extraction, and parallel processing.
    Returns matches with an additional 'overlapped' field indicating if the match's bounding box overlaps with others,
    only if check_overlap is True.

    Args:
        image_bytes (bytes): Bytes của ảnh gốc.
        template_bytes (bytes): Bytes của template.
        threshold (float): Ngưỡng khớp mẫu.
        edge_base (bool): Sử dụng ảnh biên nếu True, ảnh grayscale nếu False.
        check_overlap (bool): Kiểm tra chồng lấn giữa các hình chữ nhật chính nếu True.

    Returns:
        dict: Kết quả với số lượng matches và danh sách matches, mỗi match có trường overlapped.
    """
    # Chuyển bytes thành mảng NumPy
    image_array = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Cannot decode image file")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    template_array = np.frombuffer(template_bytes, np.uint8)
    template_bgr = cv2.imdecode(template_array, cv2.IMREAD_COLOR)
    if template_bgr is None:
        raise ValueError("Cannot decode template file")
    template_rgb = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)

    # Tiền xử lý ảnh
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=CONFIG['clahe_clip_limit'],
        tileGridSize=CONFIG['clahe_tile_grid_size']
    )
    clahe_image = clahe.apply(img_gray)
    edges = auto_canny(clahe_image, sigma=CONFIG['canny_sigma'])
    kernel = np.ones(CONFIG['morph_kernel_size'], np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Chuẩn bị ảnh cho matching
    if edge_base:
        v = np.median(template_rgb)
        lower = int(max(0, (1.0 - CONFIG['canny_sigma']) * v))
        upper = int(min(255, (1.0 + CONFIG['canny_sigma']) * v))
        img_edges = cv2.Canny(img_gray, lower, upper)
        temp_edges = cv2.Canny(template_rgb, lower, upper)
        kernel = np.ones(CONFIG['morph_kernel_size'], np.uint8)
        img = cv2.morphologyEx(img_edges, cv2.MORPH_CLOSE, kernel)
        temp = cv2.morphologyEx(temp_edges, cv2.MORPH_CLOSE, kernel)
    else:
        img = img_gray
        temp = template_gray

    # Tìm ROIs
    rois = find_rois_threshold(img, edges, template_gray.shape, CONFIG)

    # Xử lý từng ROI (tạm thời bỏ song song hóa để đơn giản hóa)
    all_points_list = []
    for roi in rois:
        points = process_roi(roi, img, temp, threshold)
        all_points_list.extend(points)

    # Kiểm tra chồng lấn giữa các hình chữ nhật chính
    height, width = template_gray.shape
    rectangles = []
    for point_info in all_points_list:
        point = point_info[0]
        angle = point_info[1]
        scale_factor = point_info[2] / 100  # Chuyển từ phần trăm sang tỷ lệ
        score = point_info[3]

        # Tính toán tọa độ 4 góc của hình chữ nhật
        w_scaled = width * scale_factor
        h_scaled = height * scale_factor
        center_x = point[0] + w_scaled / 2
        center_y = point[1] + h_scaled / 2

        # Tạo Polygon cho hình chữ nhật
        rect_points = [
            (point[0], point[1]),  # Góc trên trái
            (point[0] + w_scaled, point[1]),  # Góc trên phải
            (point[0] + w_scaled, point[1] + h_scaled),  # Góc dưới phải
            (point[0], point[1] + h_scaled)  # Góc dưới trái
        ]
        poly = Polygon(rect_points)
        # poly = scale(poly, xfact=1, yfact=1, origin=(point[0], point[1]))
        poly = rotate(poly, angle, origin=(center_x, center_y), use_radians=False)
        rectangles.append((poly, point, angle, scale_factor, score))

        # Kiểm tra chồng lấn giữa các hình chữ nhật chính (nếu check_overlap=True)
        overlapping = set()
        if check_overlap:
            height, width = template_gray.shape
            rectangles = []
            for point_info in all_points_list:
                point = point_info[0]
                angle = point_info[1]
                scale_factor = point_info[2] / 100  # Chuyển từ phần trăm sang tỷ lệ
                score = point_info[3]

                # Tính toán tọa độ 4 góc của hình chữ nhật
                w_scaled = width * scale_factor
                h_scaled = height * scale_factor
                center_x = point[0] + w_scaled / 2
                center_y = point[1] + h_scaled / 2

                # Tạo Polygon cho hình chữ nhật
                rect_points = [
                    (point[0], point[1]),  # Góc trên trái
                    (point[0] + w_scaled, point[1]),  # Góc trên phải
                    (point[0] + w_scaled, point[1] + h_scaled),  # Góc dưới phải
                    (point[0], point[1] + h_scaled)  # Góc dưới trái
                ]
                poly = Polygon(rect_points)
                # poly = scale(poly, xfact=1, yfact=1, origin=(point[0], point[1]))
                poly = rotate(poly, angle, origin=(center_x, center_y), use_radians=False)
                rectangles.append((poly, point, angle, scale_factor, score))

            # Kiểm tra chồng lấn
            for i, (poly1, _, _, _, _) in enumerate(rectangles):
                for j, (poly2, _, _, _, _) in enumerate(rectangles):
                    if i < j and poly1.intersects(poly2) and not poly1.touches(poly2):
                        overlapping.add(i)
                        overlapping.add(j)

        # Chuẩn bị kết quả trả về
        matches = []
        for i, point_info in enumerate(all_points_list):
            point, angle, scale, score = point_info
            match_info = {
                "x": int(point[0]),
                "y": int(point[1]),
                "angle": float(angle),
                "scale": float(scale),
                "score": round(float(score), 3),
                "overlapped": i in overlapping if check_overlap else False  # Thêm trường overlapped
            }
            matches.append(match_info)

        result = {
            "count": len(matches),
            "matches": matches
        }

        logging.info(f"Số lượng matches: {len(matches)}")
        return result

