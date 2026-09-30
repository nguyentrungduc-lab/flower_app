import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

FLOWER_CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

class FlowerClassifier:
    def __init__(self, weights_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Khởi tạo ResNet-18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 2. Đổi lớp FC cho 5 loài hoa
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(FLOWER_CLASSES))
        
        if weights_path:
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            except Exception as e:
                print(f"Lưu ý: Không nạp được weights từ {weights_path}: {e}")
                
        self.model = self.model.to(self.device)
        self.model.eval()

        # 3. Transform ảnh đầu vào
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        conf, preds = torch.max(probabilities, 0)
        
        prob_dict = {
            FLOWER_CLASSES[i]: float(probabilities[i].item()) 
            for i in range(len(FLOWER_CLASSES))
        }
        
        return {
            "label": FLOWER_CLASSES[preds.item()],
            "confidence": float(conf.item()),
            "probabilities": prob_dict
        }

# ⚠️ DÒNG BẮT BUỘC ĐỂ API KHÔNG BÁO LỖI IMPORT
# CŨ: classifier = FlowerClassifier()
# MỚI: Chỉ định đường dẫn tới file weights vừa train
import os
weights_file = "artifacts/flower_resnet18.pth" if os.path.exists("artifacts/flower_resnet18.pth") else None
classifier = FlowerClassifier(weights_path=weights_file)