import cv2
import numpy as np
import tflite_runtime.interpreter as tflite  # lighter than full TF on Pi
import time
from picamera2 import Picamera2

MODEL_PATH      = "./best_float32_yolo8.tflite"
CLASS_NAMES     = ["katun", "rayon", "linen", "mori"]
CONF_THRESHOLD  = 0.5
NMS_THRESHOLD   = 0.45
INPUT_SIZE      = (224, 224)
CAMERA_INDEX    = 0

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Atur input camera agar sesuai dengan yang diperlukan model
input_shape = input_details[0]['shape']
INPUT_SIZE  = (input_shape[2], input_shape[1])
print(f"Model input size: {INPUT_SIZE}")

def preprocess(frame, input_size=INPUT_SIZE):
    img_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size)
    img_norm    = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(img_norm, axis=0)


def postprocess(output, orig_shape):
    predictions  = np.squeeze(output[0]).T
    class_scores = predictions[:, 4:]
    class_ids    = np.argmax(class_scores, axis=1)
    confidences  = np.max(class_scores, axis=1)

    mask         = confidences > CONF_THRESHOLD
    boxes_raw    = predictions[mask, :4]
    class_ids    = class_ids[mask]
    confidences  = confidences[mask]

    if len(boxes_raw) == 0:
        return [], [], []

    oh, ow = orig_shape
    cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    x1 = np.clip((cx - w / 2) * ow, 0, ow).astype(int)
    y1 = np.clip((cy - h / 2) * oh, 0, oh).astype(int)
    x2 = np.clip((cx + w / 2) * ow, 0, ow).astype(int)
    y2 = np.clip((cy + h / 2) * oh, 0, oh).astype(int)
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), confidences.tolist(),
        CONF_THRESHOLD, NMS_THRESHOLD
    )
    indices = indices.flatten() if len(indices) > 0 else []

    return boxes_xyxy[indices], class_ids[indices], confidences[indices]


COLORS = {
    "katun": (0,   255, 0),
    "rayon": (255, 128, 0),
    "linen": (0,   128, 255),
    "mori":  (255, 0,   255),
}

def draw(frame, boxes, class_ids, confidences):
    for box, cls_id, conf in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = box
        name  = CLASS_NAMES[cls_id]
        color = COLORS[name]
        label = f"{name} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    print("Running — press Q to quit")
    prev_time = time.time()

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        orig_shape = frame.shape[:2]

        input_data = preprocess(frame)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        boxes, class_ids, confidences = postprocess(output_data, orig_shape)

        curr_time = time.time()
        fps       = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        frame = draw(frame, boxes, class_ids, confidences)
        cv2.imshow("Fabric Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
