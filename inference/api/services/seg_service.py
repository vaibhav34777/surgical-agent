import onnxruntime as ort
import numpy as np
import cv2

class SegService:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, frame):
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (768, 480))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        logits = self.session.run(None, {self.input_name: img})[0]
        mask = np.argmax(logits[0], axis=0).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask
