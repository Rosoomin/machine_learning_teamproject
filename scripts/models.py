"""
분류 모델 정의
CIFAR-10에 최적화된 ResNet-18
"""

import torch.nn as nn
import torchvision.models as models


def get_resnet18_cifar10(num_classes=10, pretrained=False):
    """
    CIFAR-10용 ResNet-18
    
    Args:
        num_classes: 클래스 개수 (기본 10)
        pretrained: ImageNet 사전학습 가중치 사용 여부
    
    Returns:
        model: CIFAR-10에 최적화된 ResNet-18
    """
    
    # ResNet-18 불러오기
    model = models.resnet18(pretrained=pretrained)
    
    # CIFAR-10 최적화 (32×32 작은 이미지)
    # 1. 첫 Conv layer 수정 (7×7 → 3×3)
    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    
    # 2. MaxPool 제거 (이미지가 작아서)
    model.maxpool = nn.Identity()
    
    # 3. 마지막 FC layer를 10개 클래스로
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


if __name__ == "__main__":
    # 테스트
    import torch
    
    model = get_resnet18_cifar10()
    print("✅ ResNet-18 모델 생성 완료!")
    print(f"   파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 더미 입력으로 테스트
    x = torch.randn(1, 3, 32, 32)  # CIFAR-10 크기
    output = model(x)
    print(f"   입력 크기: {x.shape}")
    print(f"   출력 크기: {output.shape}")