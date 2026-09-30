import os
import urllib.request
import tarfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

def setup_dataset():
    data_dir = "./data"
    flowers_dir = os.path.join(data_dir, "flower_photos")
    
    if not os.path.exists(flowers_dir):
        os.makedirs(data_dir, exist_ok=True)
        url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        tgz_path = os.path.join(data_dir, "flower_photos.tgz")
        
        print("⏳ Đang tải tập dữ liệu TF Flowers (khoảng 218MB)...")
        urllib.request.urlretrieve(url, tgz_path)
        
        print("📦 Đang giải nén dữ liệu...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        os.remove(tgz_path)
        print("✅ Đã chuẩn bị xong dữ liệu!")
    else:
        print("✅ Dữ liệu hoa đã có sẵn.")
        
    return flowers_dir

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Sử dụng thiết bị: {device}")

    flowers_dir = setup_dataset()

    # Preprocessing
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(flowers_dir, transform=data_transforms)
    
    # Chia tập train/val (80% train, 20% val)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Khởi tạo mô hình ResNet-18 pre-trained
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5) # 5 lớp hoa
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 3 # Train nhanh 3 epoch để có kết quả ngay
    print(f"🔥 Bắt đầu huấn luyện trong {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct.double() / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

    # Lưu file trọng số đã train
    os.makedirs("artifacts", exist_ok=True)
    save_path = "artifacts/flower_resnet18.pth"
    torch.save(model.state_dict(), save_path)
    print(f"🎉 Huấn luyện hoàn tất! Trọng số đã được lưu tại: {save_path}")

if __name__ == "__main__":
    train_model()