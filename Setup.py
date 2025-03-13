import os

# Danh sách thư mục cần tạo
folders = [
    "routes",
    "services",
    "models",
    "utils",
    "tests"
]

# Danh sách file cần tạo
files = {
    "main.py": """from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI app"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
""",
    ".env": "SECRET_KEY=your_secret_key",
}

# Tạo thư mục
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    init_file = os.path.join(folder, "__init__.py")
    if not os.path.exists(init_file):
        open(init_file, "w").close()  # Tạo file __init__.py để Python nhận diện package

# Tạo file Python mặc định
for file, content in files.items():
    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8") as f:
            f.write(content)

print("✅ Setup project thành công! Cấu trúc đã được tạo.")
