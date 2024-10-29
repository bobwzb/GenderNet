import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 定义模型结构
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

if __name__ == '__main__':
    # 数据加载和预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 加载数据集
    dataset = datasets.ImageFolder(root='training', transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 使用 DataLoader 加载数据，并启用多线程
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 打印数据集信息
    print(f"Classes: {dataset.classes}")
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # 初始化模型并移动到 GPU
    model = GenderClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


    # 训练模型
    num_epochs = 8
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Batch [{batch_idx + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch + 1}/{num_epochs}] Avg Loss: {running_loss / len(train_loader):.4f}')

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f'GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')

    # 保存模型
    torch.save(model.state_dict(), 'gender_classifier.pth')
    print("Model saved successfully!")

    # 评估模型
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
