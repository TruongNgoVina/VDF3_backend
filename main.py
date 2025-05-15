from fastapi import FastAPI
from routes.edge_detection_routes import router as edge_router
from routes.template_matching_routes import router as matching_router
from routes.robot_calibration_routes import router as robot_calib_router


app = FastAPI()

# Đăng ký route
app.include_router(edge_router)
app.include_router(matching_router)
app.include_router(robot_calib_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
