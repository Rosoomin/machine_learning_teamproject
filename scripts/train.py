# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy

import argparse
import os
from data_preparation import create_few_shot_cifar10, build_cifar10_with_sd
from augmentation import create_augmented_dataset
from models_custom import get_resnet18_cifar10



def _make_loader(dataset, batch_size, train, num_workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    metric = MulticlassAccuracy(num_classes=10).to(device)
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        metric.update(logits, y)
    return loss_sum / n, metric.compute().item()


def train_classifier(model, train_dataset, test_dataset,
                     epochs=100, batch_size=128, lr=0.05,
                     device='cuda', save_path='./models/best.pth'):
    device = device if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    train_loader = _make_loader(train_dataset, batch_size, train=True)
    test_loader  = _make_loader(test_dataset,  batch_size, train=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))

    best_acc = 0.0
    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': [], 'best_acc': 0.0}

    for ep in range(1, epochs+1):
        model.train()
        metric = MulticlassAccuracy(num_classes=10).to(device)
        loss_sum, n = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device=='cuda')):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
            metric.update(logits, y)

        tr_loss = loss_sum / n
        tr_acc = metric.compute().item()
        te_loss, te_acc = _evaluate(model, test_loader, device)
        scheduler.step()

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), save_path)

        history['train_loss'].append(tr_loss)
        history['test_loss'].append(te_loss)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)
        history['best_acc'] = best_acc

        print(f"[{ep:03d}] train {tr_loss:.4f}/{tr_acc:.4f}  |  test {te_loss:.4f}/{te_acc:.4f}  |  lr {scheduler.get_last_lr()[0]:.5f}")

    # 최고 성능 가중치 로드
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, history

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples_per_class", type=int, default=100,
                   help="클래스당 사용할 CIFAR-10 원본 샘플 수")
    p.add_argument("--use_trad_aug", action="store_true",
                   help="전통적 증강 적용 (원본 + 증강)")
    p.add_argument("--num_augment", type=int, default=1,
                   help="원본 1장당 생성할 증강 수 (1이면 2배)")
    p.add_argument("--use_sd", action="store_true",
                   help="Stable Diffusion 생성 이미지 포함 여부")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--save_path", type=str, default="./models/best.pth")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # 1️⃣ 기본 데이터셋: few-shot CIFAR-10
    train_subset, testset = create_few_shot_cifar10(samples_per_class=args.samples_per_class)

    # 2️⃣ 증강 옵션 처리
    if args.use_trad_aug:
        # --- 전통적 증강 적용 ---
        train_ds = create_augmented_dataset(train_subset, num_augment=args.num_augment)
        print(f"✅ 전통적 증강 적용 완료! (원본+증강 총 {len(train_ds)}장)")
    elif args.use_sd:
        # --- Stable Diffusion 생성 이미지 포함 ---
        train_ds = build_cifar10_with_sd(
            split="train",
            include_sd=True,
            samples_per_class=args.samples_per_class
        )
        print(f"✅ 생성형 이미지 포함 완료! (총 {len(train_ds)}장)")
    else:
        # --- 원본만 ---
        train_ds = train_subset
        print(f"✅ 원본 데이터만 사용 ({len(train_ds)}장)")

    test_ds = testset

    # 3️⃣ 모델 준비
    model = get_resnet18_cifar10(num_classes=10)

    # 4️⃣ 학습 실행
    model, hist = train_classifier(
        model=model,
        train_dataset=train_ds,
        test_dataset=test_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device="cuda",
        save_path=args.save_path
    )

    print(f"\n🎯 Best test accuracy: {hist['best_acc']:.4f}")
    print(f"📁 Saved model: {args.save_path}")


if __name__ == "__main__":
    main()