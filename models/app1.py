import io
import numpy as np
import cv2
from PIL import Image, ImageDraw
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

app = FastAPI()

# Kết nối đến Triton Server
TRITON_URL = "localhost:8000"
MODEL_NAME = "yolov9-base_onnx"
client = InferenceServerClient(url=TRITON_URL)

# Class labels và màu sắc
labels = ["body", "head", "face"]
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
CONF_THRESHOLD = 0.5
INPUT_SIZE = 640

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size
    # Resize về 640x640
    img_resized = img.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC)

    # Chuyển ảnh thành numpy array [-1, 1]
    img_np = (np.array(img_resized, dtype=np.float32) / 255.0) * 2 - 1
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0).astype(np.float32)

    return img, img_np, (orig_w, orig_h)
def postprocess_output(detections, orig_w, orig_h):
    print("Raw detections shape:", detections.shape)

    if detections.shape[0] == 1:
        detections = np.squeeze(detections, axis=0)

    if detections.ndim == 3 and detections.shape[1] < detections.shape[2]:
        detections = detections.transpose(2, 1, 0)

    print("Processed detections shape:", detections.shape)

    boxes, confidences, class_ids = [], [], []
    scale_x, scale_y = orig_w / INPUT_SIZE, orig_h / INPUT_SIZE

    for detection in detections:
        if len(detection) < 7:
            continue

        cx, cy, bw, bh = detection[:4]
        class_logits = detection[4:7]
        probs = softmax(class_logits)
        final_conf = np.max(probs)
        cls_id = int(np.argmax(probs))

        if final_conf < CONF_THRESHOLD:
            continue

        x = int((cx - bw / 2) * scale_x)
        y = int((cy - bh / 2) * scale_y)
        w = int(bw * scale_x)
        h = int(bh * scale_y)

        boxes.append([x, y, w, h])
        confidences.append(final_conf)
        class_ids.append(cls_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, 0.45)
    if indices is not None and len(indices) > 0:
        indices = indices.flatten()
    else:
        indices = []

    final_detections = [(boxes[i], class_ids[i], confidences[i]) for i in indices]
    return final_detections

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Tiền xử lý ảnh
    orig_image, img_input, (orig_w, orig_h) = preprocess_image(image_bytes)

    inputs = InferInput("images", img_input.shape, "FP32")
    inputs.set_data_from_numpy(img_input, binary_data=True)

    output_names = ["output0"]
    outputs = [InferRequestedOutput(name) for name in output_names]

    results = client.infer(MODEL_NAME, inputs=[inputs], outputs=outputs)
    detections = results.as_numpy("output0").squeeze().T  # (8400, 7)

    final_detections = postprocess_output(detections, orig_w, orig_h)

    # Vẽ bounding box lên ảnh gốc
    draw = ImageDraw.Draw(orig_image)
    for (x, y, w, h), class_id, conf in final_detections:
        x2, y2 = x + w, y + h
        draw.rectangle([x, y, x2, y2], outline=colors[class_id], width=2)

        label = f"{labels[class_id]}: {conf:.2f}"
        draw.text((x, y), label, fill=colors[class_id])

    # Encode ảnh
    img_bytes = io.BytesIO()
    orig_image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/jpeg")
