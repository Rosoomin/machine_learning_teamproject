"""
실험 실행 스크립트
다양한 설정으로 실험 가능
"""

import torch
import os
from pathlib import Path

from data_preparation import create_few_shot_cifar10  # ✅ A안: 테스트셋 정규화 여기서 처리
from augmentation import create_augmented_dataset, AugmentedDataset
from models import get_resnet18_cifar10
from train import train_classifier


def run_experiment(
    experiment_name,
    samples_per_class=100,
    num_augment=0,
    epochs=100,
    batch_size=128,
    lr=0.1,
    use_generated=False,
    generated_path=None
):
    """
    실험 실행
    
    Args:
        experiment_name: 실험 이름 (예: "baseline", "traditional", "sd")
        samples_per_class: 클래스당 원본 이미지 개수
        num_augment: 전통적 증강 배수 (0=안 함, 1=1배, 2=2배...)
        epochs: 학습 에포크
        batch_size: 배치 크기
        lr: 학습률
        use_generated: 생성 이미지 사용 여부
        generated_path: 생성 이미지 경로
    """
    
    print(f"\n{'='*70}")
    print(f"실험: {experiment_name}")
    print(f"{'='*70}")
    print(f"설정:")
    print(f"  - 원본 이미지: {samples_per_class}장/클래스 (총 {samples_per_class*10}장)")
    print(f"  - 전통적 증강: {num_augment}배")
    print(f"  - 생성 이미지: {'사용' if use_generated else '미사용'}")
    print(f"  - Epochs: {epochs}")
    print(f"{'='*70}\n")
    
    # 1. 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. 데이터 준비
    print("📂 데이터 준비 중...")
    train_original, test_data = create_few_shot_cifar10(
        samples_per_class=samples_per_class
    )
    
    # 3. 증강 적용
    if num_augment > 0:
        print(f"🔄 전통적 증강 적용 ({num_augment}배)...")
        train_data = create_augmented_dataset(train_original, num_augment=num_augment)
    else:
        print("📌 원본 데이터만 사용")
        train_data = AugmentedDataset(train_original, augmentation=None, num_augment=0)
    
    # 4. 생성 이미지 추가 (TODO: 나중에 구현)
    if use_generated and generated_path:
        print(f"🎨 생성 이미지 추가: {generated_path}")
        # TODO: 생성 이미지 로드 및 병합
        pass
    
    print(f"\n최종 훈련 데이터: {len(train_data):,}장")
    print(f"테스트 데이터: {len(test_data):,}장\n")
    
    # 5. 모델 생성
    print("🔧 모델 생성...")
    model = get_resnet18_cifar10()
    
    # 6. 학습
    save_dir = Path('./models')
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f'{experiment_name}_best.pth'
    
    trained_model, history = train_classifier(
        model=model,
        train_dataset=train_data,
        test_dataset=test_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        save_path=str(save_path)
    )
    
    # 7. 결과 저장
    import json
    result = {
        'experiment_name': experiment_name,
        'samples_per_class': samples_per_class,
        'num_augment': num_augment,
        'use_generated': use_generated,
        'total_train_images': len(train_data),
        'epochs': epochs,
        'best_test_accuracy': history['best_acc'],
        'final_train_accuracy': history['train_acc'][-1],
        'final_test_accuracy': history['test_acc'][-1]
    }
    
    result_dir = Path('./results')
    result_dir.mkdir(exist_ok=True)
    
    with open(result_dir / f'{experiment_name}_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ 결과 저장: {result_dir / f'{experiment_name}_result.json'}")
    
    return result


# ================================================================
# 미리 정의된 실험들
# ================================================================

def experiment_baseline(samples=100, epochs=100):
    """실험 1: 원본만"""
    return run_experiment(
        experiment_name='baseline',
        samples_per_class=samples,
        num_augment=0,  # 증강 안 함
        epochs=epochs
    )


def experiment_traditional(samples=100, augment_ratio=1, epochs=100):
    """실험 2: 원본 + 전통적 증강"""
    return run_experiment(
        experiment_name='traditional',
        samples_per_class=samples,
        num_augment=augment_ratio,  # 1배 증강
        epochs=epochs
    )


def experiment_generated(samples=100, generated_path=None, epochs=100):
    """실험 3: 원본 + 생성 이미지"""
    return run_experiment(
        experiment_name='generated',
        samples_per_class=samples,
        num_augment=0,
        use_generated=True,
        generated_path=generated_path,
        epochs=epochs
    )


# ================================================================
# 메인 실행
# ================================================================

if __name__ == "__main__":
    print("🚀 CIFAR-10 데이터 증강 실험")
    print("="*70)
    
    # 실험 설정 (여기만 수정하면 됨!)
    SAMPLES_PER_CLASS = 10  # 클래스당 원본 이미지 수
    EPOCHS = 5             # 학습 에포크
    
    # 실험 1: Baseline (원본 1,000장)
    print("\n" + "="*70)
    print("실험 1: Baseline")
    print("="*70)
    result1 = experiment_baseline(samples=SAMPLES_PER_CLASS, epochs=EPOCHS)
    
    # 실험 2: Traditional (원본 1,000 + 증강 1,000 = 2,000장)
    print("\n" + "="*70)
    print("실험 2: Traditional Augmentation")
    print("="*70)
    result2 = experiment_traditional(
        samples=SAMPLES_PER_CLASS,
        augment_ratio=1,  # 1배 증강 (1,000장 추가)
        epochs=EPOCHS
    )
    
    # 실험 3: Generated (원본 1,000 + 생성 1,000 = 2,000장)
    # TODO: 생성 이미지 준비 후 활성화
    # print("\n" + "="*70)
    # print("실험 3: Generated Images")
    # print("="*70)
    # result3 = experiment_generated(
    #     samples=SAMPLES_PER_CLASS,
    #     generated_path='./data/sd_generated_32',
    #     epochs=EPOCHS
    # )
    
    # 결과 비교
    print("\n" + "="*70)
    print("📊 실험 결과 요약")
    print("="*70)
    print(f"실험 1 (Baseline):     {result1['best_test_accuracy']:.2f}%")
    print(f"실험 2 (Traditional):  {result2['best_test_accuracy']:.2f}%")
    # print(f"실험 3 (Generated):    {result3['best_test_accuracy']:.2f}%")
    print("="*70)
