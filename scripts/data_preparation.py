"""
CIFAR-10 데이터 준비
클래스당 N장만 샘플링하여 Few-shot 환경 구성 + 테스트셋 정규화 포함
"""

import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


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


def create_few_shot_cifar10(samples_per_class=100, data_dir='./data/cifar10', seed=42):
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

    # train: transform=None (원본 유지 후 Subset으로 few-shot)
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None
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


if __name__ == "__main__":
    train_data, test_data = create_few_shot_cifar10(samples_per_class=100)
    print(f"\n클래스 목록: {get_class_names()}")
