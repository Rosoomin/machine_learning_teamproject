# scripts/gen_sd.py
"""
Stable Diffusion v1.5로 CIFAR-10 클래스별 512x512 이미지를 생성하는 스크립트.

- 출력 경로: data/aug/sd512/<class>/img_00000.png ...
- 재실행(resume) 지원: 이미 저장된 개수만큼 건너뛰고 이어서 생성
- 기본 설정: 클래스당 100장, 배치=6, steps=30, guidance=7.5, seed=42

사용 예:
    python scripts/gen_sd.py                          # 기본값
    python scripts/gen_sd.py --per_class 200          # 클래스당 200장
    python scripts/gen_sd.py --batch 8 --steps 20     # 빠른 생성
"""

from __future__ import annotations
from pathlib import Path
import argparse
import math
import re
import torch
from diffusers import StableDiffusionPipeline


# ===== 1) 클래스/프롬프트 정의 =====
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# CIFAR-10 각 클래스에 맞춘 기본 프롬프트 (너무 길지 않게, 현실 사진 톤 위주)
PROMPTS = {
    "airplane":   "a photo of an airplane in the sky, realistic, high detail, natural lighting",
    "automobile": "a photo of a car on a street, realistic, high detail, natural lighting",
    "bird":       "a photo of a small bird on a branch, shallow depth of field, realistic",
    "cat":        "a photo of a domestic cat looking at camera, realistic fur, natural light",
    "deer":       "a photo of a deer in a forest, natural lighting, realistic",
    "dog":        "a photo of a dog outdoors, realistic fur, natural light",
    "frog":       "a macro photo of a frog on a leaf, detailed texture",
    "horse":      "a photo of a horse running in a field, natural lighting",
    "ship":       "a photo of a ship on the sea, waves, high detail",
    "truck":      "a photo of a truck on a road, realistic, high detail"
}

# 품질 저하 요소를 억제하기 위한 네거티브 프롬프트
NEG = "low quality, blurry, deformed, extra limbs, text, watermark, oversaturated, nsfw"

# 출력 루트(512px 저장)
OUT = Path("data/aug/sd512")


# ===== 2) 유틸: 현재 폴더에 저장된 이미지 개수 파악 (resume 용) =====
def count_existing_images(cls_dir: Path) -> int:
    """
    cls_dir 내의 img_00000.png 패턴을 찾아 현재까지 저장된 개수를 반환.
    파일이 중간에 비어 있어도 안전하게 '최대 인덱스 + 1'을 계산한다.
    """
    if not cls_dir.exists():
        return 0
    pattern = re.compile(r"img_(\d{5})\.png$")
    max_idx = -1
    for p in cls_dir.glob("*.png"):
        m = pattern.match(p.name)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1  # 다음에 저장할 인덱스


# ===== 3) Stable Diffusion 파이프라인 로드 =====
def load_pipe(device: str = "cuda") -> StableDiffusionPipeline:
    """
    Stable Diffusion v1.5 파이프라인을 로드하고, 메모리 절약 옵션을 설정한다.
    - fp16 사용(속도/메모리 절약)
    - attention slicing 활성화(메모리 절약)
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,          # 교육용/클린 데이터셋 용도라 비활성. 필요시 켜도 OK.
        revision=None
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    # VRAM이 빡빡할 때는 아래 옵션이 도움이 될 수 있음 (CUDA에서만 의미 있음)
    if device == "cuda" and torch.cuda.is_available():
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
    return pipe


# ===== 4) 한 클래스의 이미지를 배치로 생성해서 저장 =====
def generate_class(
    pipe: StableDiffusionPipeline,
    class_name: str,
    prompt: str,
    total_needed: int,
    batch: int = 6,
    steps: int = 30,
    guidance: float = 7.5,
    seed: int = 42
) -> int:
    """
    지정한 클래스 폴더에 total_needed 장까지 생성/저장한다.
    이미 일부가 저장돼 있으면 그 이후부터 이어서 생성(resume).
    반환값: 이번 실행에서 실제로 새로 저장한 개수.
    """
    cls_dir = OUT / class_name
    cls_dir.mkdir(parents=True, exist_ok=True)

    # 이미 저장된 개수(=다음 저장 인덱스) 확인하여 이어서 생성
    start_idx = count_existing_images(cls_dir)
    if start_idx >= total_needed:
        print(f"[{class_name}] already has {start_idx}/{total_needed}. skip.")
        return 0

    remaining = total_needed - start_idx
    num_batches = math.ceil(remaining / batch)

    # 생성 재현성을 위해 generator에 seed 고정
    g = torch.Generator(device=pipe.device.type).manual_seed(seed)

    saved_now = 0
    for b in range(num_batches):
        bs = min(batch, remaining - saved_now)
        with torch.autocast(device_type=pipe.device.type):
            out = pipe(
                [prompt] * bs,
                negative_prompt=[NEG] * bs,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=g
            )

        for im in out.images:
            idx = start_idx + saved_now
            im.convert("RGB").save(cls_dir / f"img_{idx:05d}.png")
            saved_now += 1

        print(f"[{class_name}] {start_idx + saved_now}/{total_needed}")

    print(f"[DONE] {class_name}: +{saved_now} (total {start_idx + saved_now}/{total_needed})")
    return saved_now


# ===== 5) 전체 클래스 루프 =====
def run_generation(per_class: int, batch: int, steps: int, guidance: float, seed: int) -> None:
    """
    10개 CIFAR-10 클래스에 대해 순차적으로 이미지 생성.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}, fp16=True, per_class={per_class}, batch={batch}, steps={steps}, guidance={guidance}")

    pipe = load_pipe(device=device)

    total_new = 0
    for cname in CLASS_NAMES:
        total_new += generate_class(
            pipe=pipe,
            class_name=cname,
            prompt=PROMPTS[cname],
            total_needed=per_class,
            batch=batch,
            steps=steps,
            guidance=guidance,
            seed=seed
        )
    print(f"[summary] newly saved images: {total_new}")


# ===== 6) CLI 진입점 =====
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CIFAR-10-like images with Stable Diffusion v1.5")
    p.add_argument("--per_class", type=int, default=100, help="클래스당 생성할 이미지 수 (기본 100)")
    p.add_argument("--batch", type=int, default=6, help="생성 배치 크기 (VRAM에 맞춰 조정)")
    p.add_argument("--steps", type=int, default=30, help="num_inference_steps (20~30 권장)")
    p.add_argument("--guidance", type=float, default=7.5, help="classifier-free guidance scale")
    p.add_argument("--seed", type=int, default=42, help="난수 시드 (재현성)")
    return p.parse_args()


def main():
    args = parse_args()
    run_generation(
        per_class=args.per_class,
        batch=args.batch,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
