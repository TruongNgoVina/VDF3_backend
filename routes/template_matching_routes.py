from fastapi import APIRouter, UploadFile, File, HTTPException
import time
from services.template_matching_service import process_template_matching
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

router = APIRouter(prefix="/api", tags=["Template Matching"])

@router.post("/match-template/")
async def match_template(
    image: UploadFile = File(..., description="The larger image to search in (JPEG/PNG)"),
    template: UploadFile = File(..., description="The template image to match (JPEG/PNG)"),
    threshold: float = 0.4,
    edge_base: bool = True
):
    """
    Upload an image and a template image, then perform invariant template matching.
    Returns a JSON response with the number of matched objects and their details (position, angle, scale, score).

    Args:
        image: The larger image file to search in.
        template: The template image file to match.
        threshold: Similarity threshold for matching (0.0 to 1.0, default: 0.4).
        edge_base: Whether to use edge-based matching (default: True).

    Returns:
        dict: {"count": int, "matches": list of match details}

    Raises:
        HTTPException: If file format is invalid or processing fails.
    """
    # Kiểm tra định dạng file
    if not image.content_type.startswith("image/") or not template.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files (JPEG/PNG) are supported.")

    # Đọc bytes từ file upload
    try:
        image_bytes = await image.read()
        template_bytes = await template.read()
    except Exception as e:
        logging.error(f"Error reading uploaded files: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to read uploaded files.")

    # Kiểm tra ngưỡng threshold
    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1.")

    # Ghi log thông tin request
    logging.info(f"Received request: threshold={threshold}, edge_base={edge_base}")

    # Đo thời gian xử lý
    start_time = time.time()

    # Gọi service xử lý template matching
    try:
        result = process_template_matching(
            image_bytes=image_bytes,
            template_bytes=template_bytes,
            threshold=threshold,
            edge_base=edge_base
        )
    except Exception as e:
        logging.error(f"Template matching failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Template matching processing failed: {str(e)}")

    # Tính thời gian xử lý
    processing_time = time.time() - start_time
    logging.info(f"Processing completed in {processing_time:.2f} seconds. Matches found: {result['count']}")

    # Trả về kết quả
    return result