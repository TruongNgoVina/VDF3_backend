import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Biến toàn cục cho click_and_crop
box_points = []
button_down = False
low_thresh = 0.05


# Hàm Auto Canny
def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)


# Hàm xoay ảnh
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# Hàm co giãn ảnh
def scale_image(image, percent, maxwh):
    max_width = maxwh[1]
    max_height = maxwh[0]
    max_percent_width = max_width / image.shape[1] * 100
    max_percent_height = max_height / image.shape[0] * 100
    max_percent = min(max_percent_width, max_percent_height)
    if percent > max_percent:
        percent = max_percent
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    result = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return result, percent


# Hàm xử lý sự kiện chuột để crop
def click_and_crop(event, x, y, flags, param):
    global box_points, button_down
    if (button_down == False) and (event == cv2.EVENT_LBUTTONDOWN):
        button_down = True
        box_points = [(x, y)]
    elif (button_down == True) and (event == cv2.EVENT_MOUSEMOVE):
        image_copy = param.copy()
        point = (x, y)
        cv2.rectangle(image_copy, box_points[0], point, (0, 255, 0), 2)
        cv2.imshow("Template Cropper - Press C to Crop", image_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        button_down = False
        box_points.append((x, y))
        cv2.rectangle(param, box_points[0], box_points[1], (0, 255, 0), 2)
        cv2.imshow("Template Cropper - Press C to Crop", param)


# Hàm crop mẫu từ ảnh
def template_crop(image):
    clone = image.copy()
    cv2.namedWindow("Template Cropper - Press C to Crop")
    param = image
    cv2.setMouseCallback("Template Cropper - Press C to Crop", click_and_crop, param)
    while True:
        cv2.imshow("Template Cropper - Press C to Crop", image)
        key = cv2.waitKey(1)
        if key == ord("c"):
            cv2.destroyAllWindows()
            break
    if len(box_points) == 2:
        cropped_region = clone[box_points[0][1]:box_points[1][1], box_points[0][0]:box_points[1][0]]
    return cropped_region


# Hàm xây dựng kim tự tháp ảnh
def build_pyramid(image, max_level):
    pyramid = [image]
    current_image = image
    for _ in range(max_level):
        current_image = cv2.pyrDown(current_image)
        if current_image.shape[0] < 1 or current_image.shape[1] < 1:
            break
        pyramid.append(current_image)
    return pyramid


# Hàm tính tỷ lệ co giãn theo cấp kim tự tháp
def get_pyramid_level_scale(level):
    return 1 / (2 ** level)


# Ánh xạ chuỗi method sang hằng số OpenCV
METHOD_MAP = {
    "TM_CCOEFF": cv2.TM_CCOEFF,
    "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    "TM_CCORR": cv2.TM_CCORR,
    "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
    "TM_SQDIFF": cv2.TM_SQDIFF,
    "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED
}


# Hàm xử lý một cặp (góc, tỷ lệ) cụ thể
def match_single_rotation_scale(args):
    """
    Xử lý khớp mẫu cho một góc xoay và tỷ lệ co giãn cụ thể.

    Args:
        args: Tuple chứa (angle, scale, current_img, template_gray, method,
                        matched_thresh, level, image_maxwh)
    Returns:
        List các điểm khớp với thông tin [vị trí, góc, tỷ lệ, điểm số]
    """
    angle, scale, current_img, template_gray, method_str, matched_thresh, level, image_maxwh = args


    # Ánh xạ method từ chuỗi sang hằng số
    method = METHOD_MAP.get(method_str)
    if method is None:
        raise ValueError(f"Invalid method: {method_str}")

    # Co giãn mẫu
    scaled_template_gray, actual_scale = scale_image(template_gray, scale, image_maxwh)

    # Xoay mẫu
    rotated_template = scaled_template_gray if angle == 0 else rotate_image(scaled_template_gray, angle)

    # Áp dụng cv2.matchTemplate
    matched_points = cv2.matchTemplate(current_img, rotated_template, method)

    # Xác định các điểm thỏa mãn ngưỡng
    if method in [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED]:
        satisfied_points = np.where(matched_points >= low_thresh)
    else:  # TM_SQDIFF, TM_SQDIFF_NORMED
        satisfied_points = np.where(matched_points <= low_thresh)

    # Điều chỉnh tọa độ về cấp kim tự tháp gốc
    scale_factor = 2 ** level
    points = []
    for pt in zip(*satisfied_points[::-1]):
        orig_pt = (int(pt[0] * scale_factor), int(pt[1] * scale_factor))
        score = matched_points[pt[1], pt[0]]
        points.append([orig_pt, angle, actual_scale, score])

    return points


# Hàm xử lý min/max cho từng cặp (góc, tỷ lệ)
def match_single_rotation_scale_minmax(args):
    """
    Xử lý khớp mẫu cho một góc xoay và tỷ lệ co giãn, chỉ lấy điểm min/max.

    Args:
        args: Tuple chứa (angle, scale, current_img, template_gray, method,
                        matched_thresh, level, image_maxwh)
    Returns:
        Điểm khớp tốt nhất với thông tin [vị trí, góc, tỷ lệ, điểm số]
    """
    angle, scale, current_img, template_gray, method_str, matched_thresh, level, image_maxwh = args

    # Ánh xạ method từ chuỗi sang hằng số
    method = METHOD_MAP.get(method_str)
    if method is None:
        raise ValueError(f"Invalid method: {method_str}")

    # Co giãn mẫu
    scaled_template_gray, actual_scale = scale_image(template_gray, scale, image_maxwh)

    # Xoay mẫu
    rotated_template = scaled_template_gray if angle == 0 else rotate_image(scaled_template_gray, angle)

    # Áp dụng cv2.matchTemplate
    matched_points = cv2.matchTemplate(current_img, rotated_template, method)

    # Tìm điểm min/max
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
    if method in [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED]:
        if max_val >= low_thresh:
            orig_loc = (int(max_loc[0] * (2 ** level)), int(max_loc[1] * (2 ** level)))
            return [orig_loc, angle, actual_scale, max_val]
    else:  # TM_SQDIFF, TM_SQDIFF_NORMED
        if min_val <= low_thresh:
            orig_loc = (int(min_loc[0] * (2 ** level)), int(min_loc[1] * (2 ** level)))
            return [orig_loc, angle, actual_scale, min_val]
    return None


# Hàm chính với song song hóa
def invariant_match_template(img_gray, template_gray, method, matched_thresh, rot_range, rot_interval,
                             scale_range, scale_interval, rm_redundant, minmax, rgbdiff_thresh=float("inf"),
                             pyramid_levels=2):
    """
    Template matching tool

    Args:
        img_gray: gray image where the search is running.
        template_gray: gray searched template.
        method: Parameter specifying the comparison method (string).
        matched_thresh: Threshold of matched results (0~1).
        rot_range: Array of rotation angle range in degrees.
        rot_interval: Interval of rotation angle in degrees.
        scale_range: Array of scaling range in percentage.
        scale_interval: Interval of scaling in percentage.
        rm_redundant: Remove redundant matches.
        minmax: Find points with min/max value.
        rgbdiff_thresh: Threshold of RGB difference.
        pyramid_levels: Number of pyramid levels.

    Returns:
        List of matched points in format [[point.x, point.y], angle, scale, score].
    """
    # start_time = time.time()
    # Kiểm tra method
    if method not in METHOD_MAP:
        raise ValueError(f"Invalid method: {method}. Must be one of {list(METHOD_MAP.keys())}")

    # Chuyển ảnh và mẫu sang grayscale
    # img_gray = cv2.cvtColor(rgbimage, cv2.COLOR_RGB2GRAY)
    # template_gray = cv2.cvtColor(rgbtemplate, cv2.COLOR_RGB2GRAY)
    image_maxwh = img_gray.shape
    height, width = template_gray.shape

    all_points = []
    img_pyramid = build_pyramid(img_gray, pyramid_levels)

    if not minmax:
        # Trường hợp lấy tất cả điểm vượt ngưỡng
        for level in range(len(img_pyramid)):
            current_img = img_pyramid[level]
            level_scale = get_pyramid_level_scale(level)
            scale_min = max(scale_range[0], int(level_scale * 100 * 0.8))
            scale_max = min(scale_range[1], int(level_scale * 100 * 1.2))

            if scale_min >= scale_max:
                continue

            # Tạo danh sách góc và tỷ lệ
            angles = range(rot_range[0], rot_range[1], rot_interval)
            scales = range(scale_min, scale_max, scale_interval)
            tasks = [(angle, scale, current_img, template_gray, method, matched_thresh, level, image_maxwh)
                     for angle in angles for scale in scales]

            # Song song hóa với ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(match_single_rotation_scale, task) for task in tasks]
                for future in as_completed(futures):
                    points = future.result()
                    all_points.extend(points)

    else:
        # Trường hợp minmax = True với 3 lớp lọc
        for level in range(len(img_pyramid)):
            current_img = img_pyramid[level]
            level_scale = get_pyramid_level_scale(level)
            scale_min = max(scale_range[0], int(level_scale * 100 * 0.8))
            scale_max = min(scale_range[1], int(level_scale * 100 * 1.2))

            if scale_min >= scale_max:
                continue

            scales = range(scale_min, scale_max, scale_interval)

            # Bước 1: Lưới thô 40 độ
            coarse_rot_interval_1 = 35
            coarse_angles_1 = range(rot_range[0], rot_range[1], coarse_rot_interval_1)
            coarse_tasks_1 = [(angle, scale, current_img, template_gray, method, matched_thresh, level, image_maxwh)
                              for angle in coarse_angles_1 for scale in scales]

            coarse_points_1 = []
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(match_single_rotation_scale_minmax, task) for task in coarse_tasks_1]
                for future in as_completed(futures):
                    point = future.result()
                    if point is not None:
                        coarse_points_1.append(point)

            # Tìm điểm cao nhất từ lưới 40 độ
            if coarse_points_1:
                if method in ["TM_CCOEFF", "TM_CCOEFF_NORMED", "TM_CCORR", "TM_CCORR_NORMED"]:
                    coarse_points_1 = sorted(coarse_points_1, key=lambda x: -x[3])
                elif method in ["TM_SQDIFF", "TM_SQDIFF_NORMED"]:
                    coarse_points_1 = sorted(coarse_points_1, key=lambda x: x[3])
                best_coarse_point_1 = coarse_points_1[0]
                best_angle_1 = best_coarse_point_1[1]  # Góc tốt nhất từ lưới 40 độ

                # Bước 2: Lưới thô 10 độ quanh best_angle_1 ±40 độ
                coarse_angle_start_2 = max(rot_range[0], best_angle_1 - 20)
                coarse_angle_end_2 = min(rot_range[1], best_angle_1 + 20)
                coarse_rot_interval_2 = 10
                coarse_angles_2 = range(coarse_angle_start_2, coarse_angle_end_2, coarse_rot_interval_2)
                coarse_tasks_2 = [(angle, scale, current_img, template_gray, method, matched_thresh, level, image_maxwh)
                                  for angle in coarse_angles_2 for scale in scales]

                coarse_points_2 = []
                with ThreadPoolExecutor(max_workers=None) as executor:
                    futures = [executor.submit(match_single_rotation_scale_minmax, task) for task in coarse_tasks_2]
                    for future in as_completed(futures):
                        point = future.result()
                        if point is not None:
                            coarse_points_2.append(point)

                # Tìm điểm cao nhất từ lưới 10 độ
                if coarse_points_2:
                    if method in ["TM_CCOEFF", "TM_CCOEFF_NORMED", "TM_CCORR", "TM_CCORR_NORMED"]:
                        coarse_points_2 = sorted(coarse_points_2, key=lambda x: -x[3])
                    elif method in ["TM_SQDIFF", "TM_SQDIFF_NORMED"]:
                        coarse_points_2 = sorted(coarse_points_2, key=lambda x: x[3])
                    best_coarse_point_2 = coarse_points_2[0]
                    best_angle_2 = best_coarse_point_2[1]  # Góc tốt nhất từ lưới 10 độ

                    # Bước 3: Lưới tinh 1 độ quanh best_angle_2 ±10 độ
                    fine_angle_start = max(rot_range[0], best_angle_2 - 5)
                    fine_angle_end = min(rot_range[1], best_angle_2 + 5)
                    fine_angles = np.arange(fine_angle_start, fine_angle_end, 1)
                    fine_tasks = [(angle, scale, current_img, template_gray, method, matched_thresh, level, image_maxwh)
                                  for angle in fine_angles for scale in scales]

                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(match_single_rotation_scale_minmax, task) for task in fine_tasks]
                        for future in as_completed(futures):
                            point = future.result()
                            if point is not None and point[3] >= matched_thresh:
                                all_points.append(point)

        # Sắp xếp theo điểm số để lấy điểm tốt nhất
        if all_points:
            if method in ["TM_CCOEFF", "TM_CCOEFF_NORMED", "TM_CCORR", "TM_CCORR_NORMED"]:
                all_points = sorted(all_points, key=lambda x: -x[3])
            elif method in ["TM_SQDIFF", "TM_SQDIFF_NORMED"]:
                all_points = sorted(all_points, key=lambda x: x[3])

    # Loại bỏ điểm trùng lặp
    if rm_redundant:
        lone_points_list = []
        visited_points_list = []
        for point_info in all_points:
            point = point_info[0]
            scale = point_info[2]
            score = point_info[3]
            all_visited_points_not_close = True
            if visited_points_list:
                for visited_point in visited_points_list:
                    if (abs(visited_point[0] - point[0]) < (width * scale / 100) and
                            abs(visited_point[1] - point[1]) < (height * scale / 100)):
                        all_visited_points_not_close = False
                if all_visited_points_not_close:
                    lone_points_list.append([point, point_info[1], scale, score])
                    visited_points_list.append(point)
            else:
                lone_points_list.append([point, point_info[1], scale, score])
                visited_points_list.append(point)
        points_list = lone_points_list
    else:
        points_list = all_points
    # print(f"Matching time: {time.time() - start_time}s")

    # Lọc theo khác biệt màu nếu cần
    if rgbdiff_thresh != float("inf"):
        print(">>>RGBDiff Filtering>>>")
        color_filtered_list = []
        template_channels = cv2.mean(rgbtemplate)[:3]
        for point_info in points_list:
            point = point_info[0]
            angle = point_info[1]
            scale = point_info[2]
            score = point_info[3]
            cropped_img = rgbimage[point[1]:point[1] + height, point[0]:point[0] + width]
            if cropped_img.size == 0:
                continue
            cropped_channels = cv2.mean(cropped_img)[:3]
            total_diff = np.sum(np.absolute(np.array(cropped_channels) - np.array(template_channels)))
            print(total_diff)
            if total_diff < rgbdiff_thresh:
                color_filtered_list.append([point, angle, scale, score])
        return color_filtered_list
    # print(f"Time per ROI: {time.time() - start_time}s")
    return points_list



