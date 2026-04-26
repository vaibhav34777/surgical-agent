import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/rendezvous/pytorch")))
from network import Rendezvous

class VideoService:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Rendezvous('resnet18', hr_output=False, use_ln=True).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_frame(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, _, triplet_logits = self.model(img_tensor)
            probs = torch.sigmoid(triplet_logits).cpu().numpy()[0]
        
        actions = []
        for i, p in enumerate(probs):
            if p > 0.5:
                actions.append(i)
        return actions
