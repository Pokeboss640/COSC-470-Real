# requirements
#pip install torch torchvision pandas matplotlib

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

print("PyTorch:", torch.__version__)

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset,batch_size=128,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=128,shuffle=False)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cpu")
device = torch.device("cuda")
print("Device:", device)

class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        return self.layers(x)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1,8,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8*13*13,10)
        )

    def forward(self,x):
        return self.network(x)

class MediumCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*5*5,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )

    def forward(self,x):
        return self.classifier(self.features(x))

class LargeCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        return self.classifier(self.features(x))


class LabCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),
            #nn.Conv2d(64,64,3,padding=1),
            #nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_evaluate(model, epochs=3):

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start = time.time()

    for epoch in range(epochs):
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    train_time = time.time() - start

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    return accuracy, train_time
models = {
    "Dense": DenseNet(),
    "Small CNN": SmallCNN(),
    "Medium CNN": MediumCNN(),
    "Large CNN": LargeCNN(),
    "Lab CNN": LabCNN()
}

results = []

for name, model in models.items():

    print(f"Training {name}")

    accuracy, train_time = train_and_evaluate(model, epochs=3)

    params = count_params(model)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Parameters": params,
        "Model Size (MB)": params * 4 / (1024**2),
        "Train Time (s)": train_time
    })

df = pd.DataFrame(results)
df.sort_values("Accuracy", ascending=False)

plt.figure(figsize=(8,5))
plt.scatter(df["Parameters"], df["Accuracy"], s=150)

for _, row in df.iterrows():
    plt.annotate(row["Model"], (row["Parameters"], row["Accuracy"]))

plt.xlabel("Parameters")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Model Complexity")
plt.grid(True)
plt.show()

fig, ax = plt.subplots(1,2, figsize=(12,5))

ax[0].bar(df["Model"], df["Train Time (s)"])
ax[0].set_title("Training Time")

ax[1].bar(df["Model"], df["Parameters"])
ax[1].set_title("Parameter Count")

plt.tight_layout()
plt.show()

print(accuracy)
print(params)

'''
Putting write up here since I cannot do an extra notebook.
I figured that since the model was already 98.9 percent accurate as given, by just running the last parameter set again I could squeeze out a little bit more accuracy. I was correct.
I gained a whole .001 percent accuracy, but it took barely ay additional time (although probably too much to just keep doing it again and again). So I tested that as well. My test with
one extra 64 pass was .9908 accurate and about 2 seconds slower then the large model. My test with two dropped accuracy back down to .988, meaning that one extra pass is all I can get
away with before the model begins decreasing in accuracy
'''


