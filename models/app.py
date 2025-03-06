import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

app = FastAPI()

# Kết nối đến Triton Server
TRITON_URL = "localhost:8000"
MODEL_NAME = "yolov9-base_onnx"
client = InferenceServerClient(url=TRITON_URL)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Đọc ảnh gốc
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_resized = img.resize((640, 640))
        img_np = np.array(img_resized)

        # Chuẩn bị input theo định dạng
        img_array = img_np.astype(np.float32) / 255.0  # Normalize về [0,1]
        img_array = np.transpose(img_array, (2, 0, 1))  # CHW format (3, 640, 640)
        img_array = np.expand_dims(img_array, axis=0)  # Batch size = 1

        # Gửi dữ liệu đến Triton
        inputs = [InferInput("images", img_array.shape, "FP32")]
        inputs[0].set_data_from_numpy(img_array)
        outputs = [InferRequestedOutput("output0")]

        # Gọi mô hình YOLOv9
        response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

        # Lấy kết quả từ Triton Server
        output_data = response.as_numpy("output0")
        if output_data is None:
            return {"error": "No output from model"}

        return {"output": output_data.tolist()}

    except Exception as e:
        return {"error": str(e)}
