"""
실험 실행 스크립트
여러 설정(원본/전통/생성형)을 같은 파이프라인으로 공정 비교
"""

from pathlib import Path
import json
import torch

from data_preparation import create_few_shot_cifar10, build_cifar10_with_sd
from augmentation import create_augmented_dataset, AugmentedDataset
from models_custom import get_resnet18_cifar10
from train import train_classifier


def run_experiment(
    experiment_name: str,
    samples_per_class: int = 100,
    num_augment: int = 0,          # 전통적 증강 배수(0이면 미사용)
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.05,
    use_generated: bool = False,   # 생성형(sd32) 포함 여부
) -> dict:
    """
    실험 실행

    Args:
        experiment_name: 실험 이름 (예: "baseline", "traditional", "generated")
        samples_per_class: 클래스당 원본 이미지 개수
        num_augment: 전통적 증강 배수 (0=안 함, 1=1배, 2=2배...)
        epochs: 학습 에포크
        batch_size: 배치 크기
        lr: 학습률
        use_generated: Stable Diffusion(sd32) 생성 이미지 포함 여부

    Returns:
        result(dict): 핵심 지표 및 메타 정보
    """
    print(f"\n{'='*70}")
    print(f"실험: {experiment_name}")
    print(f"{'='*70}")
    print("설정:")
    print(f"  - 원본 이미지: {samples_per_class}장/클래스 (총 {samples_per_class*10}장)")
    print(f"  - 전통적 증강: {num_augment}배")
    print(f"  - 생성 이미지(sd32): {'포함' if use_generated else '미포함'}")
    print(f"  - Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"{'='*70}\n")

    # 1) 디바이스
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2) 데이터 준비 (생성형 포함/미포함에 따라 공정한 경로 선택)
    if use_generated:
        # ✅ 표준 파이프라인: 원본 few-shot + sd32 병합, 테스트셋도 동일 기준
        print("📂 데이터 준비 중... (원본 + 생성형 sd32 병합)")
        train_data = build_cifar10_with_sd(
            split="train",
            include_sd=True,
            samples_per_class=samples_per_class,
        )
        test_data = build_cifar10_with_sd(split="test")
        print("🎨 생성형 이미지 포함하여 학습합니다.")
    else:
        print("📂 데이터 준비 중... (원본/전통)")
        train_original, test_data = create_few_shot_cifar10(samples_per_class=samples_per_class)
        if num_augment > 0:
            print(f"🔄 전통적 증강 적용: {num_augment}배")
            train_data = create_augmented_dataset(train_original, num_augment=num_augment)
        else:
            print("📌 원본 데이터만 사용")
            train_data = AugmentedDataset(train_original, augmentation=None, num_augment=0)

    print(f"\n최종 훈련 데이터: {len(train_data):,}장")
    print(f"테스트 데이터: {len(test_data):,}장\n")

    # 3) 모델
    print("🔧 모델 생성...")
    model = get_resnet18_cifar10()

    # 4) 학습
    save_dir = Path("./models")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"{experiment_name}_best.pth"

    trained_model, history = train_classifier(
        model=model,
        train_dataset=train_data,
        test_dataset=test_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        save_path=str(save_path),
    )

    # 5) 결과 저장
    result = {
        "experiment_name": experiment_name,
        "samples_per_class": samples_per_class,
        "num_augment": num_augment,
        "use_generated": use_generated,
        "total_train_images": len(train_data),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "best_test_accuracy": history["best_acc"],
        "final_train_accuracy": history["train_acc"][-1],
        "final_test_accuracy": history["test_acc"][-1],
    }

    result_dir = Path("./results")
    result_dir.mkdir(exist_ok=True)
    with open(result_dir / f"{experiment_name}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ 결과 저장: {result_dir / f'{experiment_name}_result.json'}")
    print(f"💾 가중치 저장: {save_path}")
    print(f"🎯 Best test accuracy: {history['best_acc']:.4f}")

    return result


# ==============================
# 미리 정의한 실험 단축 함수들
# ==============================

def experiment_baseline(samples: int = 100, epochs: int = 100,
                        batch_size: int = 128, lr: float = 0.05) -> dict:
    """실험 1: 원본만"""
    return run_experiment(
        experiment_name="baseline",
        samples_per_class=samples,
        num_augment=0,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_generated=False,
    )


def experiment_traditional(samples: int = 100, augment_ratio: int = 1, epochs: int = 100,
                           batch_size: int = 128, lr: float = 0.05) -> dict:
    """실험 2: 원본 + 전통적 증강"""
    return run_experiment(
        experiment_name="traditional",
        samples_per_class=samples,
        num_augment=augment_ratio,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_generated=False,
    )


def experiment_generated(samples: int = 100, epochs: int = 100,
                         batch_size: int = 128, lr: float = 0.05) -> dict:
    """실험 3: 원본 + 생성형(sd32)"""
    return run_experiment(
        experiment_name="generated",
        samples_per_class=samples,
        num_augment=0,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_generated=True,
    )


# ==============================
# 메인 실행(예시)
# ==============================

if __name__ == "__main__":
    print("🚀 CIFAR-10 데이터 증강 실험")
    print("="*70)

    # 공정 비교 기본값(필요시 여기만 바꾸면 전체에 반영)
    SAMPLES_PER_CLASS = 100
    EPOCHS = 100
    BATCH = 128
    LR = 0.05

    # 실험 1: Baseline (원본 1,000장)
    print("\n" + "="*70)
    print("실험 1: Baseline")
    print("="*70)
    result1 = experiment_baseline(samples=SAMPLES_PER_CLASS, epochs=EPOCHS,
                                  batch_size=BATCH, lr=LR)

    # 실험 2: Traditional (원본 1,000 + 증강 1,000 = 2,000장)
    print("\n" + "="*70)
    print("실험 2: Traditional Augmentation")
    print("="*70)
    result2 = experiment_traditional(samples=SAMPLES_PER_CLASS, augment_ratio=1,
                                     epochs=EPOCHS, batch_size=BATCH, lr=LR)

    # 실험 3: Generated (원본 1,000 + 생성 1,000 = 2,000장)
    # 필요 시 활성화
    # print("\n" + "="*70)
    # print("실험 3: Generated Images")
    # print("="*70)
    # result3 = experiment_generated(samples=SAMPLES_PER_CLASS, epochs=EPOCHS,
    #                                batch_size=BATCH, lr=LR)

    # 결과 요약
    print("\n" + "="*70)
    print("📊 실험 결과 요약")
    print("="*70)
    print(f"실험 1 (Baseline):     {result1['best_test_accuracy']:.4f}")
    print(f"실험 2 (Traditional):  {result2['best_test_accuracy']:.4f}")
    # print(f"실험 3 (Generated):    {result3['best_test_accuracy']:.4f}")
    print("="*70)
