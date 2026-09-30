from fastapi import FastAPI, UploadFile, File, HTTPException
from core.model import classifier

# ⚠️ BẮT BUỘC KHAI BÁO TÊN BIẾN LÀ 'app'
app = FastAPI(
    title="Flower Classification API",
    description="API nhận diện 5 loài hoa bằng ResNet-18",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok", "model": "ResNet-18"}

@app.post("/api/classify")
async def classify_flower(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File tải lên phải là hình ảnh.")
    
    try:
        contents = await file.read()
        result = classifier.predict(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)