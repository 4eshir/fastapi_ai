import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image


class ImageClassifier:
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(ImageClassifier.SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.sigmoid(self.fc2(x))
            return x

    def __init__(self, train_dir, test_dir, model_path='model.pth', num_epochs=10, batch_size=32, learning_rate=0.001):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.model_path = model_path
        self.model = self.SimpleCNN().to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.train_loader = DataLoader(ImageFolder(root=train_dir, transform=self.transform), batch_size=batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(ImageFolder(root=test_dir, transform=self.transform), batch_size=batch_size,
                                      shuffle=False)

        self.num_epochs = num_epochs

        # Проверяем, существует ли сохранённая модель
        if os.path.exists(self.model_path):
            print("Загрузка модели из файла...")
            self.load_model()
        else:
            print("Модель не найдена. Начинаем обучение...")

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()

                self.optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {running_loss / len(self.train_loader):.4f}')

        self.save_model()  # Сохраняем модель после завершения обучения
        print('Обучение завершено!')

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f'Модель сохранена в {self.model_path}')

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()  # Устанавливаем модель в режим оценки
        print(f'Модель загружена из {self.model_path}')

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                outputs = self.model(inputs).squeeze()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Точность на тестовом наборе: {accuracy:.2f}%')

    def predict_image(self, img_path):
        self.model.eval()
        img = Image.open(img_path)
        img = self.transform(img).unsqueeze(0).to(self.device)
        output = self.model(img).squeeze()
        prediction = (output > 0.5).float()
        return "Остальное" if prediction.item() == 1 else "Человек"