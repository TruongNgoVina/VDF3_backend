import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils.InvariantTM import invariant_match_template, auto_canny
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
                             edge_base: bool = True) -> dict:
    """
    Perform invariant template matching with preprocessing, ROI extraction, and parallel processing.
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
    height, width = temp.shape[:2]
    min_area_threshold = CONFIG['min_area_ratio'] * height * width
    min_size = min(img_gray.shape[0], img_gray.shape[1]) * CONFIG['min_contour_size_ratio']

    # Xử lý tương thích với các phiên bản OpenCV
    if cv2.__version__.startswith('3'):
        _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_size:
            x, y, w, h = cv2.boundingRect(contour)
            expand = max(CONFIG['roi_min_expand'], int(max(w, h) * CONFIG['roi_expand_ratio']))
            top_left = (max(0, x - expand), max(0, y - expand))
            bottom_right = (min(img_gray.shape[1], x + w + expand), min(img_gray.shape[0], y + h + expand))
            width_roi = bottom_right[0] - top_left[0]
            height_roi = bottom_right[1] - top_left[1]
            if height_roi * width_roi >= min_area_threshold:
                rois.append((top_left, bottom_right))

    logging.info(f"Tìm thấy {len(rois)} ROIs")

    # Song song hóa xử lý ROIs
    all_points_list = []
    if rois:
        with ProcessPoolExecutor(max_workers=1) as executor:  # Giữ max_workers=1 như code test
            process_func = partial(
                process_roi,
                img=img,
                template=temp,
                threshold=threshold
            )
            results = executor.map(process_func, rois)
            for points in results:
                all_points_list.extend(points)

    # Chuẩn bị kết quả trả về
    matches = []
    for point_info in all_points_list:
        point, angle, scale, score = point_info
        match_info = {
            "x": int(point[0]),
            "y": int(point[1]),
            "angle": float(angle),
            "scale": float(scale),
            "score": round(float(score), 3)
        }
        matches.append(match_info)

    result = {
        "count": len(matches),
        "matches": matches
    }

    logging.info(f"Số lượng matches: {len(matches)}")
    return result