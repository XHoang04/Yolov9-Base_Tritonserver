import numpy as np
import cv2
from trism.model import TritonModel

# Khởi tạo Triton model
model = TritonModel(
    model="yolov9-base_onnx",
    version=1,
    url="localhost:8001",
    grpc=True
)

# Kiểm tra metadata
for inp in model.inputs:
    print(f"name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}\n")
for out in model.outputs:
    print(f"name: {out.name}, shape: {out.shape}, datatype: {out.dtype}\n")

# Đọc ảnh đầu vào
image_path = "C:/Users/hoang/Downloads/test.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (640, 640))
image = image.astype(np.float32) / 255.0
image = np.expand_dims(image.transpose(2, 0, 1), axis=0)

# Chạy inference
outputs = model.run(data=[image])

# Hiển thị kết quả
print("Output:", outputs)
