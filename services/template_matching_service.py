import cv2
import numpy as np
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils.InvariantTM import template_crop, invariant_match_template

def process_roi(roi_info, img_rgb, cropped_template_rgb, threshold=0.4, rot_range=[0, 360], rot_interval=2, scale_range=[98, 102], scale_interval=2):
    """Hàm xử lý template matching cho một ROI."""
    top_left, bottom_right = roi_info
    x1, y1 = top_left
    x2, y2 = bottom_right
    roi_rgb = img_rgb[y1:y2, x1:x2]
    height, width = cropped_template_rgb.shape[:2]

    if roi_rgb.shape[0] < height or roi_rgb.shape[1] < width:
        return []

    points_list = invariant_match_template(
        rgbimage=roi_rgb,
        rgbtemplate=cropped_template_rgb,
        method="TM_CCOEFF_NORMED",
        matched_thresh=threshold,
        rot_range=rot_range,
        rot_interval=rot_interval,
        scale_range=scale_range,
        scale_interval=scale_interval,
        rm_redundant=True,
        minmax=True,
        use_edge_matching=True
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
                             rot_range: list = [0, 360],
                             rot_interval: int = 2,
                             scale_range: list = [98, 102],
                             scale_interval: int = 2,
                             min_contour_area: float = 150,
                             roi_expand: int = 40,
                             max_workers: int = None) -> dict:
    """
    Perform invariant template matching with preprocessing, ROI extraction, and parallel processing.
    :param image_bytes: The larger image file in bytes format.
    :param template_bytes: The template image file in bytes format.
    :param threshold: Similarity threshold for matching (default: 0.4, range 0-1).
    :param rot_range: Range of rotation angles in degrees (default: [0, 360]).
    :param rot_interval: Interval between rotation angles (default: 2 degrees).
    :param scale_range: Range of scale percentages (default: [98, 102]).
    :param scale_interval: Interval between scale percentages (default: 2).
    :param min_contour_area: Minimum contour area to consider as ROI (default: 150).
    :param roi_expand: Pixels to expand ROI bounding box (default: 40).
    :param max_workers: Number of parallel workers (default: None, uses CPU count).
    :return: Dictionary containing the number of matches and details of each matched object.
    """
    # Chuyển bytes thành mảng NumPy
    image_array = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    template_array = np.frombuffer(template_bytes, np.uint8)
    template_bgr = cv2.imdecode(template_array, cv2.IMREAD_COLOR)
    template_rgb = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)

    # Cắt template và lấy kích thước
    #cropped_template_rgb = template_crop(template_rgb)
    #cropped_template_rgb = np.array(cropped_template_rgb)
    height, width = template_rgb.shape[:2]

    # Tiền xử lý ảnh chính
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(img_gray)
    blurred = cv2.GaussianBlur(clahe_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    # Tìm contours và trích xuất ROI
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            top_left = (max(0, x - roi_expand), max(0, y - roi_expand))
            bottom_right = (min(img_gray.shape[1], x + w + roi_expand), min(img_gray.shape[0], y + h + roi_expand))
            width_roi = bottom_right[0] - top_left[0]
            height_roi = bottom_right[1] - top_left[1]
            if width_roi >= width and height_roi >= height:
                rois.append((top_left, bottom_right))

    # Song song hóa xử lý các ROI
    all_points_list = []
    if rois:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            process_func = partial(
                process_roi,
                img_rgb=img_rgb,
                cropped_template_rgb=template_rgb,
                threshold=threshold,
                rot_range=rot_range,
                rot_interval=rot_interval,
                scale_range=scale_range,
                scale_interval=scale_interval
            )
            results = executor.map(process_func, rois)
            for points in results:
                all_points_list.extend(points)

    # Chuẩn bị dữ liệu trả về
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

    return result