import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt


# ================= 1. 基础设置 =================
data_dir = './data'  
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('设备:', device)
if device == 'cuda':
    print('GPU 型号:', torch.cuda.get_device_name(0))


# ================= 2. 数据加载 =================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

full_data = datasets.ImageFolder(data_dir)
class_names = full_data.classes
print('类别:', class_names, '| 图片总数:', len(full_data))

torch.manual_seed(0)
train_size = int(0.8 * len(full_data))
val_size = len(full_data) - train_size
train_subset, val_subset = random_split(full_data, [train_size, val_size])

class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label
    def __len__(self):
        return len(self.subset)

train_data = TransformSubset(train_subset, train_transform)
val_data = TransformSubset(val_subset, val_transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)


# ================= 3. 定义核心训练函数 =================
def train_and_evaluate(model, optimizer, epochs=5, save_name=None):
    criterion = nn.CrossEntropyLoss()
    hist = {'loss': [], 'acc': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                _, pred = torch.max(model(X), 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
                
        l = total_loss / len(train_loader)
        a = 100 * correct / total
        hist['loss'].append(l)
        hist['acc'].append(a)
        print(f'  第{epoch+1}轮 | 损失: {l:.3f} | 准确率: {a:.1f}%')
        
        if save_name and a > 90: 
            torch.save(model.state_dict(), save_name)
            
    return hist


# ================= 4. 开始实验 =================
num_classes = len(class_names)

print("\n===== 开始 A：冻结主干 + SGD =====")
model_a = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model_a.parameters(): p.requires_grad = False
model_a.fc = nn.Linear(model_a.fc.in_features, num_classes)
model_a = model_a.to(device)
optimizer_a = torch.optim.SGD(model_a.fc.parameters(), lr=0.001, momentum=0.9)
hist_a = train_and_evaluate(model_a, optimizer_a, epochs=5)

print("\n===== 开始 B：冻结主干 + Adam  =====")
model_b = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model_b.parameters(): p.requires_grad = False
model_b.fc = nn.Linear(model_b.fc.in_features, num_classes)
model_b = model_b.to(device)
optimizer_b = torch.optim.Adam(model_b.fc.parameters(), lr=0.0001)
hist_b = train_and_evaluate(model_b, optimizer_b, epochs=5)

print("\n===== 开始 C：全量微调 + SGD =====")
model_c = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model_c.parameters(): p.requires_grad = True
model_c.fc = nn.Linear(model_c.fc.in_features, num_classes)
model_c = model_c.to(device)
optimizer_c = torch.optim.SGD(model_c.parameters(), lr=0.0001, momentum=0.9)
hist_c = train_and_evaluate(model_c, optimizer_c, epochs=5)

print("\n===== 开始 D：全量微调 + Adam =====")
model_d = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model_d.parameters(): p.requires_grad = True
model_d.fc = nn.Linear(model_d.fc.in_features, num_classes)
model_d = model_d.to(device)
optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.0001)
hist_d = train_and_evaluate(model_d, optimizer_d, epochs=5)


# ================= 5. 画出曲线  =================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(hist_a['acc'], 'o-', label='A: Frozen + SGD')
plt.plot(hist_b['acc'], 's-', label='B: Frozen + Adam')
plt.plot(hist_c['acc'], '^-', label='C: Full + SGD')
plt.plot(hist_d['acc'], 'd-', label='D: Full + Adam')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(hist_a['loss'], 'o-', label='A: Frozen + SGD')
plt.plot(hist_b['loss'], 's-', label='B: Frozen + Adam')
plt.plot(hist_c['loss'], '^-', label='C: Full + SGD')
plt.plot(hist_d['loss'], 'd-', label='D: Full + Adam')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

save_dir = './Train_test'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'Training_Comparison_Chart.png')
plt.savefig(save_path, dpi=300)

plt.show()


# ================= 6. 模型保存 =================
torch.save(model_a.state_dict(), os.path.join(save_dir, 'model_a.pth'))
torch.save(model_b.state_dict(), os.path.join(save_dir, 'model_b.pth'))
torch.save(model_c.state_dict(), os.path.join(save_dir, 'model_c.pth'))
torch.save(model_d.state_dict(), os.path.join(save_dir, 'model_d.pth'))
print(f"✅ 四个策略的模型已全部保存到 {save_dir} 文件夹！")