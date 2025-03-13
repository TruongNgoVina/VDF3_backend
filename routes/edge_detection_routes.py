from fastapi import APIRouter, UploadFile, File
from fastapi.responses import Response
from services.edge_detection_service import process_canny_edge

router = APIRouter()


@router.post("/detect_edges/")
async def detect_edges(
        file: UploadFile = File(...),
        threshold1: int = 100,
        threshold2: int = 200
):
    # Đọc file ảnh
    image_bytes = await file.read()

    # Xử lý ảnh với Canny Edge Detection
    processed_image = process_canny_edge(image_bytes, threshold1, threshold2)

    # Trả ảnh kết quả
    return Response(content=processed_image, media_type="image/png")
