import os
import numpy as np
from PIL import Image
import random

# 노이즈 추가 함수 정의
def add_gaussian_noise(image, mean=0, stddev=25):
    """
    가우시안 노이즈를 이미지에 추가합니다.
    """
    np_image = np.array(image, dtype=np.float32)
    noise = np.random.normal(mean, stddev, np_image.shape)
    noisy_image = np_image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

# 데이터 불러오기 및 저장할 디렉토리 설정
gt_dir = "./data/gt/"
train_output_dir ='./data/train/'
val_output_dir = "./data/val/"
test_output_dir = "./data/test/"

# 디렉토리 생성
os.makedirs(val_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# 이미지 리스트
image_list = sorted(os.listdir(gt_dir))

# 이미지 개수
num_train_images = int(0.7 * len(image_list))  # 전체 이미지의 70%를 학습용으로 사용
num_val_images = int(0.2 * len(image_list))  # 전체 이미지의 20%를 검증용으로 사용
num_test_images = int(0.1 * len(image_list))  # 전체 이미지의 10%를 테스트용으로 사용

'''
# 셔플 및 분할
random.seed(42)
random.shuffle(image_list)
'''
train_images = image_list[:num_train_images]
val_images = image_list[num_train_images: num_train_images +num_val_images]
test_images = image_list[len(image_list) - num_test_images:]

# 학습용 데이터 생성 및 저장
for image_name in train_images:
    image_path = os.path.join(gt_dir, image_name)
    image = Image.open(image_path).convert('RGB')
    
    noisy_image = add_gaussian_noise(image)
    
    # 노이즈가 추가된 학습용 이미지 저장
    noisy_image.save(os.path.join(train_output_dir, f"{image_name}"))

print(f"Generated {len(train_images)} noisy training images in '{train_output_dir}'.")
    
# 검증용 데이터 생성 및 저장
for image_name in val_images:
    image_path = os.path.join(gt_dir, image_name)
    image = Image.open(image_path).convert('RGB')
    
    noisy_image = add_gaussian_noise(image)
    
    # 노이즈가 추가된 검증용 이미지 저장
    noisy_image.save(os.path.join(val_output_dir, f"{image_name}"))

print(f"Generated {len(val_images)} noisy validation images in '{val_output_dir}'.")

# 테스트용 데이터 생성 및 저장
for image_name in test_images:
    image_path = os.path.join(gt_dir, image_name)
    image = Image.open(image_path).convert('RGB')
    
    noisy_image = add_gaussian_noise(image)
    
    # 노이즈가 추가된 테스트용 이미지 저장
    noisy_image.save(os.path.join(test_output_dir, f"{image_name}"))

print(f"Generated {len(test_images)} noisy test images in '{test_output_dir}'.")
