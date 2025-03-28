# Denoising Diffusion Probabilistic Model (DDPM)

## 📌 Overview

This repository contains a Jupyter Notebook implementing a **Denoising Diffusion Probabilistic Model (DDPM)**. The notebook walks through the process of training a diffusion model, denoising images, and generating realistic samples.

---

## 📝 Section 1: ## Implementing DDPM in PyTorch

## Implementing DDPM in PyTorch
Let’s code a basic DDPM from scratch in PyTorch!

## 📝 Section 2: ## Import Necessary Libraries

## Import Necessary Libraries

## 💻 Code Block 3

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


```

## 📝 Section 4: ## Define the Diffusion Model

## Define the Diffusion Model

## 💻 Code Block 5

```python
class SimpleDiffusionModel(nn.Module):
    def __init__(self):
        super(SimpleDiffusionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


```

## 📝 Section 7: ## Prepare Dataset (MNIST)

## Prepare Dataset (MNIST)

## 💻 Code Block 8

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


```

## 📝 Section 9: ## Define Noise Schedule & Forward Diffusion Process

## Define Noise Schedule & Forward Diffusion Process

## 💻 Code Block 10

```python
def add_noise(images, noise_level=0.5):
    noise = torch.randn_like(images) * noise_level
    return images + noise


```

## 📝 Section 11: ## Train the Diffusion Model

## Train the Diffusion Model

## 💻 Code Block 13

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleDiffusionModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 5

for epoch in range(num_epochs):
    for images, _ in train_loader:
        images = images.to(device)
        noisy_images = add_noise(images)
        
        optimizer.zero_grad()
        output = model(noisy_images)
        
        loss = criterion(output, images)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save trained model
torch.save(model.state_dict(), "diffusion_model.pth")


```

## 📝 Section 14: ## Generate New Images

## Generate New Images

## 💻 Code Block 15

```python
def generate_images(model, noise_level=0.5):
    model.eval()
    noise = torch.randn((1, 1, 28, 28)).to(device)
    with torch.no_grad():
        denoised_image = model(noise)
    return denoised_image

# Load Model
model.load_state_dict(torch.load("diffusion_model.pth"))
model.to(device)

# Generate Image
generated_img = generate_images(model).cpu().squeeze().numpy()

plt.imshow(generated_img, cmap="gray")
plt.show()


```

## 📝 Section 16: ## 🎯 Final Summary

## 🎯 Final Summary
#### Defined a simple U-Net model for denoising.

#### Added noise to images and trained the model.

#### Saved the trained model for future use.

#### Generated new images by reversing the noise process.




