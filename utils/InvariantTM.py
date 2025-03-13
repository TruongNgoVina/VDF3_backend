import cv2
import numpy as np

box_points = []
button_down = False

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

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

def build_pyramid(image, max_level):
    pyramid = [image]
    current_image = image
    for _ in range(max_level):
        current_image = cv2.pyrDown(current_image)
        if current_image.shape[0] < 1 or current_image.shape[1] < 1:
            break
        pyramid.append(current_image)
    return pyramid

def get_pyramid_level_scale(level):
    return 1 / (2 ** level)

def invariant_match_template(rgbimage, rgbtemplate, method, matched_thresh, rot_range, rot_interval, scale_range, scale_interval, rm_redundant, minmax, rgbdiff_thresh=float("inf"), pyramid_levels=2, use_edge_matching=True):
    """
    Tìm kiếm mẫu trong ảnh với tùy chọn edge matching.
    rgbimage: RGB image where the search is running.
    rgbtemplate: RGB searched template.
    method: Parameter specifying the comparison method.
    matched_thresh: Threshold of matched results (0~1).
    rot_range: Array of rotation angle range in degrees.
    rot_interval: Interval of rotation angle in degrees.
    scale_range: Array of scaling range in percentage.
    scale_interval: Interval of scaling in percentage.
    rm_redundant: Remove redundant matches.
    minmax: Find points with min/max value.
    rgbdiff_thresh: Threshold of RGB difference.
    pyramid_levels: Number of pyramid levels.
    use_edge_matching: Boolean to enable edge-based matching (default: True).

    Returns: List of matched points in format [[point.x, point.y], angle, scale, score].
    """
    # Chuyển ảnh và mẫu sang grayscale
    img_gray = cv2.cvtColor(rgbimage, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(rgbtemplate, cv2.COLOR_RGB2GRAY)
    image_maxwh = img_gray.shape
    height, width = template_gray.shape

    # Trích xuất cạnh nếu use_edge_matching=True
    if use_edge_matching:
        # Tiền xử lý: Làm mờ để giảm nhiễu trước khi phát hiện cạnh
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        template_gray = cv2.GaussianBlur(template_gray, (5, 5), 0)
        # Phát hiện cạnh bằng Canny
        img_gray = cv2.Canny(img_gray, 100, 200)
        template_gray = cv2.Canny(template_gray, 100, 200)

    all_points = []
    img_pyramid = build_pyramid(img_gray, pyramid_levels)

    if not minmax:
        for level in range(len(img_pyramid)):
            current_img = img_pyramid[level]
            level_scale = get_pyramid_level_scale(level)
            scale_min = max(scale_range[0], int(level_scale * 100 * 0.8))
            scale_max = min(scale_range[1], int(level_scale * 100 * 1.2))

            if scale_min >= scale_max:
                continue

            for next_angle in range(rot_range[0], rot_range[1], rot_interval):
                for next_scale in range(scale_min, scale_max, scale_interval):
                    adjusted_scale = next_scale * level_scale
                    scaled_template_gray, actual_scale = scale_image(template_gray, adjusted_scale, image_maxwh)
                    rotated_template = scaled_template_gray if next_angle == 0 else rotate_image(scaled_template_gray, next_angle)

                    if method == "TM_CCOEFF":
                        matched_points = cv2.matchTemplate(current_img, rotated_template, cv2.TM_CCOEFF)
                        satisfied_points = np.where(matched_points >= matched_thresh)
                    elif method == "TM_CCOEFF_NORMED":
                        matched_points = cv2.matchTemplate(current_img, rotated_template, cv2.TM_CCOEFF_NORMED)
                        satisfied_points = np.where(matched_points >= matched_thresh)
                    elif method == "TM_CCORR":
                        matched_points = cv2.matchTemplate(current_img, rotated_template, cv2.TM_CCORR)
                        satisfied_points = np.where(matched_points >= matched_thresh)
                    elif method == "TM_CCORR_NORMED":
                        matched_points = cv2.matchTemplate(current_img, rotated_template, cv2.TM_CCORR_NORMED)
                        satisfied_points = np.where(matched_points >= matched_thresh)
                    elif method == "TM_SQDIFF":
                        matched_points = cv2.matchTemplate(current_img, rotated_template, cv2.TM_SQDIFF)
                        satisfied_points = np.where(matched_points <= matched_thresh)
                    elif method == "TM_SQDIFF_NORMED":
                        matched_points = cv2.matchTemplate(current_img, rotated_template, cv2.TM_SQDIFF_NORMED)
                        satisfied_points = np.where(matched_points <= matched_thresh)
                    else:
                        raise ValueError("Invalid comparison method.")

                    scale_factor = 2 ** level
                    for pt in zip(*satisfied_points[::-1]):
                        orig_pt = (int(pt[0] * scale_factor), int(pt[1] * scale_factor))
                        score = matched_points[pt[1], pt[0]]
                        all_points.append([orig_pt, next_angle, actual_scale, score])

    else:
        for level in range(len(img_pyramid)):
            current_img = img_pyramid[level]
            level_scale = get_pyramid_level_scale(level)
            scale_min = max(scale_range[0], int(level_scale * 100 * 0.8))
            scale_max = min(scale_range[1], int(level_scale * 100 * 1.2))

            if scale_min >= scale_max:
                continue

            for next_angle in range(rot_range[0], rot_range[1], rot_interval):
                for next_scale in range(scale_min, scale_max, scale_interval):
                    adjusted_scale = next_scale * level_scale
                    scaled_template_gray, actual_scale = scale_image(template_gray, adjusted_scale, image_maxwh)
                    rotated_template = scaled_template_gray if next_angle == 0 else rotate_image(scaled_template_gray, next_angle)

                    if method == "TM_CCOEFF":
                        matched_points = cv2.matchTemplate(current_img, rotated_template, cv2.TM_CCOEFF)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                        if max_val >= matched_thresh:
                            orig_loc = (int(max_loc[0] * (2 ** level)), int(max_loc[1] * (2 ** level)))
                            all_points.append([orig_loc, next_angle, actual_scale, max_val])
                    elif method == "TM_CCOEFF_NORMED":
                        matched_points = cv2.matchTemplate(current_img, rotated_template, cv2.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                        if max_val >= matched_thresh:
                            orig_loc = (int(max_loc[0] * (2 ** level)), int(max_loc[1] * (2 ** level)))
                            all_points.append([orig_loc, next_angle, actual_scale, max_val])
                    elif method == "TM_CCORR":
                        matched_points = cv2.matchTemplate(current_img, rotated_template, cv2.TM_CCORR)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                        if max_val >= matched_thresh:
                            orig_loc = (int(max_loc[0] * (2 ** level)), int(max_loc[1] * (2 ** level)))
                            all_points.append([orig_loc, next_angle, actual_scale, max_val])
                    elif method == "TM_CCORR_NORMED":
                        matched_points = cv2.matchTemplate(current_img, rotated_template, cv2.TM_CCORR_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                        if max_val >= matched_thresh:
                            orig_loc = (int(max_loc[0] * (2 ** level)), int(max_loc[1] * (2 ** level)))
                            all_points.append([orig_loc, next_angle, actual_scale, max_val])
                    elif method == "TM_SQDIFF":
                        matched_points = cv2.matchTemplate(current_img, rotated_template, cv2.TM_SQDIFF)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                        if min_val <= matched_thresh:
                            orig_loc = (int(min_loc[0] * (2 ** level)), int(min_loc[1] * (2 ** level)))
                            all_points.append([orig_loc, next_angle, actual_scale, min_val])
                    elif method == "TM_SQDIFF_NORMED":
                        matched_points = cv2.matchTemplate(current_img, rotated_template, cv2.TM_SQDIFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                        if min_val <= matched_thresh:
                            orig_loc = (int(min_loc[0] * (2 ** level)), int(min_loc[1] * (2 ** level)))
                            all_points.append([orig_loc, next_angle, actual_scale, min_val])
                    else:
                        raise ValueError("Invalid comparison method.")

        if method in ["TM_CCOEFF", "TM_CCOEFF_NORMED", "TM_CCORR", "TM_CCORR_NORMED"]:
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method in ["TM_SQDIFF", "TM_SQDIFF_NORMED"]:
            all_points = sorted(all_points, key=lambda x: x[3])

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

    if rgbdiff_thresh != float("inf"):
        print(">>>RGBDiff Filtering>>>")
        color_filtered_list = []
        template_channels = cv2.mean(rgbtemplate)[:3]
        for point_info in points_list:
            point = point_info[0]
            angle = point_info[1]
            scale = point_info[2]
            score = point_info[3]
            cropped_img = rgbimage[point[1]:point[1]+height, point[0]:point[0]+width]
            if cropped_img.size == 0:
                continue
            cropped_channels = cv2.mean(cropped_img)[:3]
            total_diff = np.sum(np.absolute(np.array(cropped_channels) - np.array(template_channels)))
            print(total_diff)
            if total_diff < rgbdiff_thresh:
                color_filtered_list.append([point, angle, scale, score])
        return color_filtered_list

    return points_list

# Ví dụ sử dụng
if __name__ == "__main__":
    rgbimage = cv2.imread("source_image.jpg")
    rgbtemplate = template_crop(rgbimage)

    results = invariant_match_template(
        rgbimage, rgbtemplate,
        method="TM_CCOEFF_NORMED",
        matched_thresh=0.8,
        rot_range=[0, 360],
        rot_interval=10,
        scale_range=[50, 200],
        scale_interval=10,
        rm_redundant=True,
        minmax=False,
        pyramid_levels=2,
        use_edge_matching=True  # Bật edge matching
    )

    print("Kết quả (point, angle, scale, score):")
    for result in results:
        point, angle, scale, score = result
        print(f"Point: {point}, Angle: {angle}, Scale: {scale}, Score: {score:.3f}")