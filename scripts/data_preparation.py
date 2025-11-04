"""
CIFAR-10 데이터 준비
클래스당 100장만 샘플링하여 Few-shot 환경 구성
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import os


def create_few_shot_cifar10(samples_per_class=100, data_dir='./data/cifar10', seed=42):
    """
    CIFAR-10에서 클래스당 지정된 개수만큼 샘플링
    
    Args:
        samples_per_class: 클래스당 샘플 개수 (기본 100)
        data_dir: 데이터 저장 경로
        seed: 랜덤 시드 (재현성)
    
    Returns:
        train_subset: Few-shot 훈련 데이터
        test_dataset: 전체 테스트 데이터
    """
    
    # 데이터 다운로드 (처음만)
    print("CIFAR-10 데이터 다운로드 중...")
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None  # 일단 transform 없이
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=None
    )
    
    # 클래스당 샘플링
    print(f"\n클래스당 {samples_per_class}장 샘플링 중...")
    targets = np.array(trainset.targets)
    indices = []
    
    np.random.seed(seed)  # 재현성 보장
    
    for class_id in range(10):
        # 해당 클래스의 모든 인덱스
        class_indices = np.where(targets == class_id)[0]
        
        # 랜덤 샘플링
        selected = np.random.choice(
            class_indices,
            samples_per_class,
            replace=False
        )
        indices.extend(selected)
    
    # Subset 생성
    train_subset = Subset(trainset, indices)
    
    # 정보 출력
    print(f"\n✅ 데이터 준비 완료!")
    print(f"   원본 훈련 데이터: {len(trainset):,}장")
    print(f"   Few-shot 훈련 데이터: {len(train_subset):,}장 ({samples_per_class} × 10)")
    print(f"   테스트 데이터: {len(testset):,}장")
    
    return train_subset, testset


def get_class_names():
    """CIFAR-10 클래스 이름"""
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]


if __name__ == "__main__":
    # 테스트 실행
    train_data, test_data = create_few_shot_cifar10(samples_per_class=100)
    
    print(f"\n클래스 목록: {get_class_names()}")