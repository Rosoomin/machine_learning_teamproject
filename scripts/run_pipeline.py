"""
전체 파이프라인 실행 스크립트

순서:
1. CIFAR-10 few-shot 데이터 준비
2. 전통적 증강 생성 (200장)
3. LoRA 학습 (원본 100 + 증강 200)
4. SD 증강 생성 (200장)
5. 4가지 실험 실행
"""

import argparse
from pathlib import Path


def run_pipeline(
    samples_per_class=100,
    traditional_augments=2,
    lora_rank=8,
    lora_steps=1000,
    sd_images_per_class=200,
    sd_strength=0.5,
    classifier_epochs=100,
    skip_traditional=False,
    skip_lora=False,
    skip_generation=False,
    skip_experiments=False
):
    """전체 파이프라인 실행"""
    
    print("="*70)
    print("CIFAR-10 Data Augmentation Pipeline")
    print("="*70)
    print(f"Config:")
    print(f"  Samples per class: {samples_per_class}")
    print(f"  Traditional augments per image: {traditional_augments}")
    print(f"  LoRA rank: {lora_rank}, steps: {lora_steps}")
    print(f"  SD images per class: {sd_images_per_class}")
    print(f"  Classifier epochs: {classifier_epochs}")
    print("="*70)
    
    # Step 1: 데이터 준비
    print("\n[Step 1] Loading CIFAR-10 few-shot dataset...")
    from data import get_few_shot_cifar10
    
    train_subset, test_subset = get_few_shot_cifar10(
        samples_per_class=samples_per_class
    )
    
    # Step 2: 전통적 증강
    traditional_aug_dir = Path('./data/aug_traditional')
    
    if not skip_traditional:
        print("\n[Step 2] Generating traditional augmentations...")
        from augment_traditional import generate_traditional_augmentations
        
        generate_traditional_augmentations(
            dataset=train_subset,
            augments_per_image=traditional_augments,
            output_dir=traditional_aug_dir
        )
    else:
        print("\n[Step 2] Skipping traditional augmentation (already exists)")
    
    # Step 3: LoRA 학습
    lora_path = Path('./models/lora/lora_weights.safetensors')
    
    if not skip_lora:
        print("\n[Step 3] Training LoRA...")
        from train_lora import train_lora
        
        train_lora(
            original_dataset=train_subset,
            aug_dir=traditional_aug_dir,
            rank=lora_rank,
            steps=lora_steps
        )
    else:
        print(f"\n[Step 3] Skipping LoRA training")
        if not lora_path.exists():
            print(f"Warning: LoRA weights not found at {lora_path}")
    
    # Step 4: SD 증강 생성
    sd_aug_dir = Path('./data/aug_sd_32')
    
    if not skip_generation:
        print("\n[Step 4] Generating SD augmentations...")
        from generate_sd import generate_sd_augmentations, downsample_to_32x32
        
        generate_sd_augmentations(
            original_dataset=train_subset,
            lora_path=lora_path,
            images_per_class=sd_images_per_class,
            strength=sd_strength
        )
        
        downsample_to_32x32()
    else:
        print("\n[Step 4] Skipping SD generation (already exists)")
    
    # Step 5: 실험 실행
    if not skip_experiments:
        print("\n[Step 5] Running experiments...")
        from train_classifier import run_experiments
        from resnet import get_resnet18_cifar10
        
        results = run_experiments(
            original_dataset=train_subset,
            test_dataset=test_subset,
            model_fn=get_resnet18_cifar10,
            traditional_aug_dir=traditional_aug_dir,
            sd_aug_dir=sd_aug_dir,
            epochs=classifier_epochs
        )
        
        print("\n" + "="*70)
        print("Pipeline completed!")
        print("="*70)
        
        return results
    else:
        print("\n[Step 5] Skipping experiments")
        print("\n" + "="*70)
        print("Pipeline completed (experiments skipped)")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Augmentation Pipeline')
    
    # 데이터 설정
    parser.add_argument('--samples_per_class', type=int, default=100,
                       help='클래스당 원본 이미지 개수')
    parser.add_argument('--traditional_augments', type=int, default=2,
                       help='각 이미지당 전통적 증강 개수')
    
    # LoRA 설정
    parser.add_argument('--lora_rank', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_steps', type=int, default=1000,
                       help='LoRA 학습 스텝')
    
    # SD 생성 설정
    parser.add_argument('--sd_images_per_class', type=int, default=200,
                       help='클래스당 SD 생성 이미지 개수')
    parser.add_argument('--sd_strength', type=float, default=0.5,
                       help='SD img2img strength')
    
    # 분류기 설정
    parser.add_argument('--classifier_epochs', type=int, default=100,
                       help='분류기 학습 에포크')
    
    # 스킵 옵션
    parser.add_argument('--skip_traditional', action='store_true',
                       help='전통적 증강 건너뛰기')
    parser.add_argument('--skip_lora', action='store_true',
                       help='LoRA 학습 건너뛰기')
    parser.add_argument('--skip_generation', action='store_true',
                       help='SD 생성 건너뛰기')
    parser.add_argument('--skip_experiments', action='store_true',
                       help='실험 건너뛰기')
    
    args = parser.parse_args()
    
    run_pipeline(
        samples_per_class=args.samples_per_class,
        traditional_augments=args.traditional_augments,
        lora_rank=args.lora_rank,
        lora_steps=args.lora_steps,
        sd_images_per_class=args.sd_images_per_class,
        sd_strength=args.sd_strength,
        classifier_epochs=args.classifier_epochs,
        skip_traditional=args.skip_traditional,
        skip_lora=args.skip_lora,
        skip_generation=args.skip_generation,
        skip_experiments=args.skip_experiments
    )


if __name__ == '__main__':
    main()
