import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import watson_vgg


# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 클래스 정의
class PairedDataset(Dataset):
    def __init__(self, gt_dir, noisy_dir, transform=None):
        """
        gt_dir: Ground truth 이미지가 저장된 폴더 경로
        noisy_dir: 노이즈가 추가된 이미지가 저장된 폴더 경로
        transform: 이미지 전처리를 위한 변환
        """
        self.gt_dir = gt_dir
        self.noisy_dir = noisy_dir
        self.transform = transform
        self.image_list = sorted(os.listdir(noisy_dir))
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        gt_image_path = os.path.join(self.gt_dir, self.image_list[idx])
        noisy_image_path = os.path.join(self.noisy_dir, self.image_list[idx])
        
        # GT 이미지와 노이즈 이미지 로드
        gt_image = Image.open(gt_image_path).convert('RGB')
        noisy_image = Image.open(noisy_image_path).convert('RGB')
        
        # 변환 적용
        if self.transform:
            gt_image = self.transform(gt_image)
            noisy_image = self.transform(noisy_image)
            
        return noisy_image, gt_image

# 데이터 변환
img_size = 512
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# GT 데이터 및 노이즈 데이터 경로 설정
gt_dir = "./data/gt/"
train_noisy_dir = "./data/train/"
val_noisy_dir = "./data/val/"
test_noisy_dir = "./data/test/"

# 데이터 로더 설정
train_dataset = PairedDataset(gt_dir=gt_dir, noisy_dir=train_noisy_dir, transform=transform)
val_dataset = PairedDataset(gt_dir=gt_dir, noisy_dir=val_noisy_dir, transform=transform)
test_dataset = PairedDataset(gt_dir=gt_dir, noisy_dir=test_noisy_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ResNet 모델 정의
class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, img_size * img_size * 3)  # 이미지 크기와 채널에 맞게 조정
        self.upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=True)
        
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 3, img_size, img_size)  # RGB 채널 이미지 형태로 변환
        return self.upsample(x)  # 입력 크기와 맞추기

# UNet 모델 정의
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 인코더 구성
        self.enc1 = self.conv_block(3, 64)   # 64 채널
        self.pool1 = nn.MaxPool2d(2, 2)      # MaxPooling
        
        self.enc2 = self.conv_block(64, 128)  # 128 채널
        self.pool2 = nn.MaxPool2d(2, 2)      # MaxPooling
        
        self.enc3 = self.conv_block(128, 256) # 256 채널
        self.pool3 = nn.MaxPool2d(2, 2)      # MaxPooling
        
        self.enc4 = self.conv_block(256, 512) # 512 채널
        self.pool4 = nn.MaxPool2d(2, 2)      # MaxPooling

        self.bottleneck = self.conv_block(512, 1024)  # Bottleneck
        
        # 디코더 구성
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # Upsampling
        self.dec4 = self.conv_block(1024, 512)  # 스킵 연결을 위한 1024 채널
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)   # Upsampling
        self.dec3 = self.conv_block(512, 256)  # 스킵 연결을 위한 512 채널
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   # Upsampling
        self.dec2 = self.conv_block(256, 128)  # 스킵 연결을 위한 256 채널
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)    # Upsampling
        self.dec1 = self.conv_block(128, 64)   # 스킵 연결을 위한 128 채널
        
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)  # 최종 출력

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 인코더 부분
        enc1 = self.enc1(x)  # (N, 64, H, W)
        enc2 = self.enc2(self.pool1(enc1))  # (N, 128, H/2, W/2)
        enc3 = self.enc3(self.pool2(enc2))  # (N, 256, H/4, W/4)
        enc4 = self.enc4(self.pool3(enc3))  # (N, 512, H/8, W/8)
        
        bottleneck = self.bottleneck(self.pool4(enc4))  # (N, 1024, H/16, W/16)

        
        # 디코더 부분

        dec4 = self.upconv4(bottleneck)  # (N, 512, H/8, W/8)
        dec4 = torch.cat((dec4, enc4), dim=1)  # 스킵 연결
        dec4 = self.dec4(dec4)  # (N, 512, H/8, W/8)

        dec3 = self.upconv3(dec4)  # (N, 256, H/4, W/4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # 스킵 연결
        dec3 = self.dec3(dec3)  # (N, 256, H/4, W/4)

        dec2 = self.upconv2(dec3)  # (N, 128, H/2, W/2)
        dec2 = torch.cat((dec2, enc2), dim=1)  # 스킵 연결
        dec2 = self.dec2(dec2)  # (N, 128, H/2, W/2)

        dec1 = self.upconv1(dec2)  # (N, 64, H, W)
        dec1 = torch.cat((dec1, enc1), dim=1)  # 스킵 연결
        dec1 = self.dec1(dec1)  # (N, 64, H, W)

        return self.final_conv(dec1)  # (N, 3, H, W)

# 모델 초기화 예시
# model = UNet()
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    
#model = ResNetModel().to(device)
model = UNet().to(device)

# 손실 함수와 최적화기 설정

criterion = watson_vgg.WatsonDistanceVgg()
criterion_2 = nn.MSELoss()
loss_percep = criterion.to(device)
loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]


optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 학습 및 검증 과정 정의
num_epochs = 50
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, gt_images in train_loader:
        images, gt_images = images.to(device), gt_images.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_i(outputs, gt_images)+criterion_2(outputs,gt_images )
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # 검증 과정
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, gt_images in val_loader:
            images, gt_images = images.to(device), gt_images.to(device)
            outputs = model(images)
            loss = loss_i(outputs, gt_images)+criterion_2(outputs,gt_images )
            val_loss += loss.item() * images.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # 모델 주기적 저장 (예: 10 에포크마다)
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"./result2/resnet_model_epoch_{epoch+1}.pth")
    
    # 테스트 데이터에 대한 예측 저장
    if (epoch + 1) % 5 == 0:  # 10 에포크마다 테스트 이미지 저장
        model.eval()
        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):
                images = images.to(device)
                outputs = model(images)
                output_img = outputs.cpu().squeeze().permute(1, 2, 0).numpy()
                output_img = (output_img * 255).astype(np.uint8)
                output_pil = Image.fromarray(output_img)
                output_pil.save(f"./result2/test_output_epoch_{epoch+1}_sample_{i+1}.png")
                if i >= 5:  # 테스트 샘플 이미지 5개까지만 저장
                    break

# 최종 모델 저장
torch.save(model.state_dict(), "./result2/resnet_model_final.pth")

# 학습 및 검증 손실 그래프
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train and Validation Loss")
plt.legend()
plt.savefig("train_val_loss_curve.png")
plt.show()
