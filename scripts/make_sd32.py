# scripts/make_sd32.py
"""
Stable Diffusion으로 생성된 512x512 이미지를 CIFAR-10 학습용 32x32로 다운샘플하여 저장.

- 입력:  data/aug/sd512/<class>/img_00000.png
- 출력:  data/aug/sd32/<class>/img_00000.png
- augmentation.py의 downsample_to_32()를 재사용하여 전처리 일관성 유지
- 이미 변환된 파일이 있으면 건너뛰는 resume 동작 지원
"""

from pathlib import Path
from PIL import Image

# 1) 경로 설정
SRC = Path("data/aug/sd512")   # SD 원본(512) 이미지 폴더
DST = Path("data/aug/sd32")    # 32x32로 저장할 폴더
DST.mkdir(parents=True, exist_ok=True)

# 2) 프로젝트의 다운샘플 함수 재사용 (없으면 LANCZOS fallback)
try:
    # 프로젝트 구조에 따라 import 경로가 다를 수 있음:
    #  - 같은 폴더에서 실행하면: from augmentation import downsample_to_32
    #  - 패키지로 실행하면:      from scripts.augmentation import downsample_to_32
    try:
        from scripts.augmentation import downsample_to_32
    except Exception:
        from augmentation import downsample_to_32  # 로컬 실행 대비
    print("[info] using downsample_to_32() from augmentation.py")
except Exception as e:
    print(f"[warn] cannot import downsample_to_32 from augmentation.py: {e}")
    print("[info] fallback: using internal LANCZOS downsampler")
    def downsample_to_32(img, resample="lanczos"):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.resize((32, 32), Image.LANCZOS)

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"{SRC} not found. 먼저 gen_sd.py를 실행해 주세요.")

    total_in, total_out, skipped = 0, 0, 0

    # 클래스 폴더 순회 (예: airplane, automobile, ... )
    for cls_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        out_dir = DST / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(cls_dir.glob("*.png"))
        total_in += len(img_paths)

        for img_path in img_paths:
            out_path = out_dir / img_path.name
            # resume: 이미 변환되어 있으면 건너뜀
            if out_path.exists():
                skipped += 1
                continue

            img = Image.open(img_path).convert("RGB")
            img32 = downsample_to_32(img)  # augmentation.py와 동일한 보간법 사용
            img32.save(out_path)
            total_out += 1

        print(f"[{cls_dir.name}] in={len(img_paths)}  saved={len(img_paths) - (skipped if skipped and out_dir.exists() else 0)}  (skip 누적={skipped})")

    print(f"\n[summary] scanned={total_in}, saved_new={total_out}, skipped_existing={skipped}")
    print(f"output → {DST.resolve()}")

if __name__ == "__main__":
    main()
