"""
데이터 증강
전통적 증강 기법 구현
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class TraditionalAugmentation:
    """전통적 데이터 증강"""
    
    def __init__(self):
        """CIFAR-10에 효과적인 증강 기법들"""
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
        ])
    
    def __call__(self, img):
        """증강 적용"""
        return self.transform(img)


class AugmentedDataset(Dataset):
    """증강된 데이터셋"""
    
    def __init__(self, original_dataset, augmentation=None, num_augment=1):
        """
        Args:
            original_dataset: 원본 데이터셋
            augmentation: 증강 변환
            num_augment: 원본 1장당 증강 몇 장 (기본 1)
        """
        self.original_dataset = original_dataset
        self.augmentation = augmentation
        self.num_augment = num_augment
        
        # 정규화 (학습용)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])
    
    def __len__(self):
        # 원본 + 증강본
        return len(self.original_dataset) * (1 + self.num_augment)
    
    def __getitem__(self, idx):
    # 원본 인덱스 계산
        original_idx = idx // (1 + self.num_augment)
        augment_idx = idx % (1 + self.num_augment)
    
    # 원본 이미지, 레이블 가져오기
        img, label = self.original_dataset[original_idx]
    
    # PIL Image로 변환
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        elif isinstance(img, np.ndarray):  # numpy 배열일 수도
            img = Image.fromarray(img)
    
    # ✨ 크기 확인 및 리사이즈 (다운샘플링)
        if img.size != (32, 32):
            img = img.resize((32, 32), Image.LANCZOS)
    
    # 첫 번째는 원본, 나머지는 증강
        if augment_idx > 0 and self.augmentation:
            img = self.augmentation(img)
    
    # 정규화
        img = self.normalize(img)
    
        return img, label


def create_augmented_dataset(dataset, num_augment=1):
    """
    증강 데이터셋 생성
    
    Args:
        dataset: 원본 데이터셋
        num_augment: 원본 1장당 증강 개수
    
    Returns:
        augmented_dataset: 증강된 데이터셋
    """
    augmentation = TraditionalAugmentation()
    augmented_dataset = AugmentedDataset(
        dataset, 
        augmentation=augmentation,
        num_augment=num_augment
    )
    
    print(f"✅ 증강 데이터셋 생성 완료!")
    print(f"   원본: {len(dataset):,}장")
    print(f"   증강 후: {len(augmented_dataset):,}장 (원본 + {num_augment}배 증강)")
    
    return augmented_dataset


if __name__ == "__main__":
    # 테스트
    from data_preparation import create_few_shot_cifar10
    
    train_data, _ = create_few_shot_cifar10(samples_per_class=10)
    aug_data = create_augmented_dataset(train_data, num_augment=1)
    
    print(f"\n샘플 확인:")
    img, label = aug_data[0]
    print(f"   이미지 shape: {img.shape}")
    print(f"   레이블: {label}")