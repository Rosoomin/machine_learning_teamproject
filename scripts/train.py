# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy


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
                     epochs=50, batch_size=128, lr=0.1,
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
