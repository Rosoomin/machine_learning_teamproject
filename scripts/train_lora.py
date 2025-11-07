import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
import safetensors.torch as safetorch


class LoRATrainDataset(Dataset):
    """LoRA 학습용 데이터셋 (원본 + 전통적 증강)"""
    
    def __init__(self, original_dataset, aug_dir='./data/aug_traditional', target_size=512):
        from data import CLASS_NAMES
        
        self.data = []
        self.target_size = target_size
        
        # 원본 100장
        for idx in range(len(original_dataset)):
            img, label = original_dataset[idx]
            if isinstance(img, torch.Tensor):
                from torchvision import transforms as T
                img = T.ToPILImage()(img)
            self.data.append((img, CLASS_NAMES[label]))
        
        # 증강 200장
        aug_path = Path(aug_dir)
        if aug_path.exists():
            for class_id, class_name in enumerate(CLASS_NAMES):
                class_dir = aug_path / class_name
                if class_dir.exists():
                    for img_path in sorted(class_dir.glob('*.png')):
                        img = Image.open(img_path).convert('RGB')
                        self.data.append((img, class_name))
        
        print(f"LoRA training dataset: {len(self.data)} images")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, class_name = self.data[idx]
        
        # 32x32 -> 512x512 업스케일
        img = img.resize((self.target_size, self.target_size), Image.LANCZOS)
        
        # PIL -> Tensor, normalize to [-1, 1]
        img = torch.from_numpy(np.array(img)).float() / 255.0
        img = img.permute(2, 0, 1)
        img = (img - 0.5) * 2.0
        
        prompt = f"a photo of {class_name}"
        
        return {'pixel_values': img, 'prompt': prompt}


def train_lora(
    original_dataset,
    aug_dir='./data/aug_traditional',
    output_dir='./models/lora',
    rank=8,
    steps=1000,
    batch_size=1,
    lr=1e-4,
    device='cuda'
):
    """
    SD img2img LoRA 파인튜닝
    
    Args:
        original_dataset: 원본 few-shot dataset
        aug_dir: 전통적 증강 이미지 경로
        output_dir: LoRA 가중치 저장 경로
        rank: LoRA rank
        steps: 학습 스텝
        batch_size: 배치 크기
        lr: 학습률
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Training LoRA (rank={rank}, steps={steps})")
    
    # 데이터셋
    dataset = LoRATrainDataset(original_dataset, aug_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # 모델 로드
    print("Loading Stable Diffusion components...")
    
    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)
    text_encoder.requires_grad_(False)
    
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    vae.requires_grad_(False)
    
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16
    ).to(device)
    
    # LoRA 적용
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    noise_scheduler = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-2)
    
    # 학습 루프
    print("Training...")
    unet.train()
    
    global_step = 0
    pbar = tqdm(total=steps)
    
    while global_step < steps:
        for batch in loader:
            pixel_values = batch['pixel_values'].to(device, dtype=torch.float16)
            prompts = batch['prompt']
            
            # Text embedding
            text_inputs = tokenizer(
                prompts, padding="max_length", max_length=77,
                truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
            
            # VAE encode
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            
            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                     (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            model_pred = unet(noisy_latents, timesteps, text_embeddings).sample
            
            # Loss
            loss = F.mse_loss(model_pred.float(), noise.float())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            global_step += 1
            if global_step >= steps:
                break
    
    pbar.close()
    
    # LoRA 가중치 저장
    print("Saving LoRA weights...")
    lora_state = {k: v.cpu() for k, v in unet.state_dict().items() if 'lora' in k}
    save_path = output_path / 'lora_weights.safetensors'
    safetorch.save_file(lora_state, save_path)
    
    print(f"Saved to {save_path}")
    
    return save_path


if __name__ == '__main__':
    from data import get_few_shot_cifar10
    
    train_subset, _ = get_few_shot_cifar10(samples_per_class=100)
    
    # 원본 100 + 전통적 증강 200으로 학습
    train_lora(
        original_dataset=train_subset,
        aug_dir='./data/aug_traditional',
        rank=8,
        steps=1000
    )


