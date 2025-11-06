"""
CIFAR-10 데이터 준비
클래스당 N장만 샘플링하여 Few-shot 환경 구성 + 테스트셋 정규화 포함
"""

import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, Dataset


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def create_few_shot_cifar10(samples_per_class=100, data_dir='./data/cifar10', seed=42,
                             train_transform=None):
    """
    CIFAR-10에서 클래스당 지정된 개수만큼 샘플링 (train),
    test는 정규화(transform) 적용해서 반환 (A안)
    """
    set_seed(seed)

    # 테스트셋 정규화 파이프라인 (A안 핵심)
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # train: transform=None (원본 유지 후 Subset으로 few-shot) --> train_transform 으로 변경완
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    # test: 정규화 적용
    testset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=normalize
    )

    # 클래스당 샘플링
    targets = np.array(trainset.targets)
    indices = []
    for class_id in range(10):
        class_indices = np.where(targets == class_id)[0]
        selected = np.random.choice(class_indices, samples_per_class, replace=False)
        indices.extend(selected)

    train_subset = Subset(trainset, indices)

    # 로그
    print(f"\n✅ 데이터 준비 완료!")
    print(f"   원본 훈련 데이터: {len(trainset):,}장")
    print(f"   Few-shot 훈련 데이터: {len(train_subset):,}장 ({samples_per_class} × 10)")
    print(f"   테스트 데이터(정규화 적용): {len(testset):,}장")

    return train_subset, testset


def get_class_names():
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

# --- Subset(PIL)을 Tensor+Normalize로 바꿔주는 래퍼 ---
class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]  # 보통 PIL Image
        # 혹시 텐서/넘파이인 경우도 안전하게 처리
        import torch
        from PIL import Image
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = self.transform(img)      # ToTensor + Normalize
        return img, label



# ------------------------------------------------------------
# Stable Diffusion으로 생성된 이미지(sd32)를 CIFAR-10과 병합하는 함수
# ------------------------------------------------------------
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset
from pathlib import Path

def build_cifar10_with_sd(split="train",
                          cifar_root="./data/cifar10",
                          sd_root="./data/aug/sd32",
                          include_sd=True,
                          samples_per_class=100,
                          seed=42):
    """
    CIFAR-10 few-shot subset + Stable Diffusion으로 생성된 sd32 이미지 병합

    Args:
        split: "train" | "test"
        cifar_root: CIFAR-10 경로
        sd_root: Stable Diffusion 32x32 이미지 폴더
        include_sd: 생성 이미지 포함 여부
        samples_per_class: few-shot에서 클래스당 사용할 샘플 수
        seed: 랜덤 시드 (재현성)

    Returns:
        Dataset (ConcatDataset or CIFAR-10 testset)
    """
    set_seed(seed)
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    if split == "train":
        # 원본 few-shot 훈련 세트 생성 (transform=None)
        train_subset, _ = create_few_shot_cifar10(
            samples_per_class=samples_per_class,
            data_dir=cifar_root,
            seed=seed,
            train_transform=normalize,
        )
        

        # 생성 이미지(sd32) 병합
        if include_sd and Path(sd_root).exists():
            sd_dataset = ImageFolder(root=sd_root, transform=normalize)
            print(f"✅ Stable Diffusion 생성 이미지 불러옴: {len(sd_dataset):,}장")
            return ConcatDataset([train_subset, sd_dataset])
        else:
            print("⚠️ Stable Diffusion 이미지 없음 또는 비활성화 상태.")
            return train_subset

    else:
        # 테스트셋은 항상 정규화만 적용
        testset = datasets.CIFAR10(
            root=cifar_root,
            train=False,
            download=True,
            transform=normalize
        )
        return testset



if __name__ == "__main__":
    train_data, test_data = create_few_shot_cifar10(samples_per_class=100)
    print(f"\n클래스 목록: {get_class_names()}")
