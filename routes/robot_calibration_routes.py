# routes/calibration.py
import json
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from services.robot_calibration_service import preprocess_image, find_contour_centers, calculate_homography

router = APIRouter(prefix="/api", tags=["Robot calibration"])


# Định nghĩa model cho tọa độ
class Coordinate(BaseModel):
    x: float
    y: float


@router.post("/calibrate/")
async def calibrate(
        file: UploadFile = File(..., description="Calibration plate image (jpg, png, etc.)"),
        coordinates_file: UploadFile = File(..., description="JSON file containing robot coordinates")
):
    """
    Calibrate the camera using a calibration plate image and robot coordinates.

    - **file**: Upload the calibration plate image.
    - **coordinates_file**: Upload a JSON file with 9 coordinate points (e.g., [{"x": 129.0, "y": 70.5}, ...]).
    - Returns: Camera points, homography matrix, and reprojection errors.
    """
    try:
        # Bước 1: Đọc và kiểm tra file ảnh
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (jpg, png, etc.)"
            )

        image_content = await file.read()
        edges = preprocess_image(image_content)

        # Bước 2: Tìm tâm contours từ ảnh
        cam_points = find_contour_centers(edges)

        # Bước 3: Đọc và phân tích file JSON
        if not coordinates_file.content_type.startswith("application/json"):
            raise HTTPException(
                status_code=400,
                detail="Coordinates file must be a JSON file"
            )

        coordinates_content = await coordinates_file.read()
        try:
            robot_points = json.loads(coordinates_content.decode('utf-8'))
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON format in coordinates file"
            )

        # Kiểm tra định dạng và số lượng tọa độ
        if not isinstance(robot_points, list) or len(robot_points) != 9:
            raise HTTPException(
                status_code=400,
                detail="JSON must contain exactly 9 coordinate points"
            )

        # Chuyển đổi và xác nhận định dạng của từng điểm
        validated_coordinates = []
        for i, point in enumerate(robot_points):
            if not isinstance(point, dict) or 'x' not in point or 'y' not in point:
                raise HTTPException(
                    status_code=400,
                    detail=f"Coordinate at index {i} must be an object with 'x' and 'y' fields"
                )
            try:
                validated_coordinates.append(Coordinate(x=float(point['x']), y=float(point['y'])))
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=400,
                    detail=f"Coordinates 'x' and 'y' at index {i} must be numeric values"
                )

        # Bước 4: Tính ma trận homography và sai số
        H, error_info = calculate_homography(cam_points, validated_coordinates)

        # Chuyển ma trận H thành list để trả về JSON
        H_list = H.tolist()

        return {
            "message": "Calibration successful",
            "camera_points": cam_points.tolist(),
            "homography_matrix": H_list,
            "reprojection_errors": {
                "errors_per_point": [float(e) for e in error_info["errors_per_point"]],
                # Chuyển thành float để JSON serializable
                "mean_error": float(error_info["mean_error"]),
                "max_error": float(error_info["max_error"])
            }
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")