import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

# 确保使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 定义与训练时相同的模型结构
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

# 确保代码通过 __main__ 入口运行
if __name__ == '__main__':
    # 初始化模型并加载保存的模型参数
    model = GenderClassifier().to(device)
    model.load_state_dict(torch.load('gender_classifier.pth'))
    model.eval()  
    print("Model loaded successfully!")

    # 定义数据预处理（与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 加载验证集数据
    validation_dataset = datasets.ImageFolder(root='Validation', transform=transform)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 打印验证集信息
    print(f"Validation Classes: {validation_dataset.classes}")
    print(f"Validation Samples: {len(validation_dataset)}")

    # 定义验证函数
    def evaluate(model, loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    # 评估验证集的准确率
    validation_accuracy = evaluate(model, validation_loader)
    print(f'Validation Accuracy: {validation_accuracy:.2f}%')
