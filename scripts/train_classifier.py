import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from pathlib import Path
import json


def train_and_evaluate(
    train_dataset,
    test_dataset,
    model_fn,
    epochs=100,
    batch_size=128,
    lr=0.05,
    device='cuda',
    save_path=None
):
    """
    분류기 학습 및 평가
    
    Args:
        train_dataset: 학습 데이터셋
        test_dataset: 테스트 데이터셋
        model_fn: 모델 생성 함수
        epochs: 에포크 수
        batch_size: 배치 크기
        lr: 학습률
        device: 디바이스
        save_path: 체크포인트 저장 경로
    """
    model = model_fn().to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))
    
    best_acc = 0.0
    history = {'train_acc': [], 'test_acc': [], 'best_acc': 0.0}
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        metric = MulticlassAccuracy(num_classes=10).to(device)
        train_loss = 0.0
        train_samples = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device, dtype=torch.float16, 
                               enabled=(device=='cuda')):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * x.size(0)
            train_samples += x.size(0)
            metric.update(logits, y)
        
        train_acc = metric.compute().item()
        train_loss = train_loss / train_samples
        
        # Test
        model.eval()
        metric = MulticlassAccuracy(num_classes=10).to(device)
        test_loss = 0.0
        test_samples = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                
                test_loss += loss.item() * x.size(0)
                test_samples += x.size(0)
                metric.update(logits, y)
        
        test_acc = metric.compute().item()
        test_loss = test_loss / test_samples
        
        scheduler.step()
        
        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            if save_path:
                torch.save(model.state_dict(), save_path)
        
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['best_acc'] = best_acc
        
        if epoch % 10 == 0 or epoch == epochs:
            print(f"[{epoch:03d}] train {train_loss:.4f}/{train_acc:.4f}  "
                  f"test {test_loss:.4f}/{test_acc:.4f}  best {best_acc:.4f}")
    
    return history


def run_experiments(
    original_dataset,
    test_dataset,
    model_fn,
    traditional_aug_dir='./data/aug_traditional',
    sd_aug_dir='./data/aug_sd_32',
    output_dir='./results',
    epochs=100,
    batch_size=128,
    lr=0.05
):
    """
    4가지 실험 실행
    
    1. 원본 100장만
    2. 원본 100 + 전통적 증강 200
    3. 원본 100 + SD 증강 200
    4. 원본 100 + 전통적 100 + SD 100
    """
    from augment_traditional import AugmentedDataset
    from data import CLASS_NAMES
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 실험 1: 원본만
    print("\n" + "="*70)
    print("Experiment 1: Original only (100 images)")
    print("="*70)
    
    dataset1 = AugmentedDataset(original_dataset, aug_dir=None)
    history1 = train_and_evaluate(
        dataset1, test_dataset, model_fn,
        epochs=epochs, batch_size=batch_size, lr=lr,
        save_path=output_path / 'exp1_best.pth'
    )
    results['exp1_original'] = {
        'train_size': len(dataset1),
        'best_acc': history1['best_acc'],
        'final_acc': history1['test_acc'][-1]
    }
    
    # 실험 2: 원본 + 전통적 증강
    print("\n" + "="*70)
    print("Experiment 2: Original + Traditional augmentation")
    print("="*70)
    
    dataset2 = AugmentedDataset(original_dataset, aug_dir=traditional_aug_dir)
    history2 = train_and_evaluate(
        dataset2, test_dataset, model_fn,
        epochs=epochs, batch_size=batch_size, lr=lr,
        save_path=output_path / 'exp2_best.pth'
    )
    results['exp2_traditional'] = {
        'train_size': len(dataset2),
        'best_acc': history2['best_acc'],
        'final_acc': history2['test_acc'][-1]
    }
    
    # 실험 3: 원본 + SD 증강
    print("\n" + "="*70)
    print("Experiment 3: Original + SD augmentation")
    print("="*70)
    
    dataset3 = AugmentedDataset(original_dataset, aug_dir=sd_aug_dir)
    history3 = train_and_evaluate(
        dataset3, test_dataset, model_fn,
        epochs=epochs, batch_size=batch_size, lr=lr,
        save_path=output_path / 'exp3_best.pth'
    )
    results['exp3_sd'] = {
        'train_size': len(dataset3),
        'best_acc': history3['best_acc'],
        'final_acc': history3['test_acc'][-1]
    }
    
    # 실험 4: 원본 + 전통적 100 + SD 100 (하이브리드)
    print("\n" + "="*70)
    print("Experiment 4: Original + Traditional(100) + SD(100)")
    print("="*70)
    
    # 각 클래스에서 100개씩만 사용
    from torch.utils.data import Subset
    
    # 전통적 증강에서 클래스당 100개 (각 클래스 200개 중 절반)
    trad_indices = []
    for class_id in range(10):
        class_count = 0
        for idx in range(len(original_dataset), len(dataset2)):
            _, label = dataset2[idx]
            if label == class_id:
                trad_indices.append(idx)
                class_count += 1
                if class_count >= 100:
                    break
    
    # SD 증강에서 클래스당 100개
    sd_indices = []
    for class_id in range(10):
        class_count = 0
        for idx in range(len(original_dataset), len(dataset3)):
            _, label = dataset3[idx]
            if label == class_id:
                sd_indices.append(idx)
                class_count += 1
                if class_count >= 100:
                    break
    
    # 원본 인덱스
    orig_indices = list(range(len(original_dataset)))
    
    # 합치기
    all_indices = orig_indices + trad_indices + sd_indices
    dataset4 = Subset(dataset2 if len(trad_indices) > 0 else dataset3, all_indices)
    
    history4 = train_and_evaluate(
        dataset4, test_dataset, model_fn,
        epochs=epochs, batch_size=batch_size, lr=lr,
        save_path=output_path / 'exp4_best.pth'
    )
    results['exp4_hybrid'] = {
        'train_size': len(dataset4),
        'best_acc': history4['best_acc'],
        'final_acc': history4['test_acc'][-1]
    }
    
    # 결과 저장
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 요약 출력
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    for name, result in results.items():
        print(f"{name:20s}: size={result['train_size']:4d}  "
              f"best_acc={result['best_acc']:.4f}")
    print("="*70)
    
    return results


if __name__ == '__main__':
    from data import get_few_shot_cifar10
    from resnet import get_resnet18_cifar10
    
    train_subset, test_subset = get_few_shot_cifar10(samples_per_class=100)
    
    run_experiments(
        original_dataset=train_subset,
        test_dataset=test_subset,
        model_fn=get_resnet18_cifar10,
        epochs=100
    )
