#!/usr/bin/env python3
"""Generate manifests for ALL retinal vessel datasets.

Produces JSONL manifests in the standard format:
  {"id": "...", "image_path": "...", "mask_path": "...", "fov_mask_path": "...", "meta": {"dataset": "..."}}

All paths are relative to the dataset_root.
"""

import json
import os
from pathlib import Path

MANIFEST_DIR = Path(__file__).parent.parent / "manifests"
MANIFEST_DIR.mkdir(exist_ok=True)

BASE = "/home/lucian/medical-imaging-ai/Retina Datasets/Full retina dataset"
OCTA_BASE = "/home/lucian/medical-imaging-ai/OCTA 500"


def write_manifest(entries: list[dict], name: str) -> None:
    path = MANIFEST_DIR / f"{name}.jsonl"
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"  ✓ {name}.jsonl — {len(entries)} entries")


# ── FIVES ────────────────────────────────────────────────────────────────
def make_fives():
    root = f"{BASE}/FIVES"
    entries = []
    for img_name in sorted(os.listdir(f"{root}/images")):
        if not img_name.endswith(".png"):
            continue
        sample_id = img_name.replace(".png", "")
        mask_name = img_name  # Same filename in masks/
        mask_path = f"masks/{mask_name}"
        if not os.path.exists(f"{root}/{mask_path}"):
            print(f"    WARN: no mask for {img_name}")
            continue
        entries.append({
            "id": sample_id,
            "image_path": f"images/{img_name}",
            "mask_path": mask_path,
            "meta": {"dataset": "FIVES"},
        })
    write_manifest(entries, "fives")
    return entries


# ── DRIVE ────────────────────────────────────────────────────────────────
def make_drive():
    root = f"{BASE}/DRIVE"
    entries = []

    # Training set
    train_imgs = f"{root}/training/images"
    for img_name in sorted(os.listdir(train_imgs)):
        if not img_name.endswith(".tif"):
            continue
        num = img_name.split("_")[0]  # e.g. "21"
        sample_id = f"train_{num}"
        mask_name = f"{num}_manual1.gif"
        mask_path = f"training/1st_manual/{mask_name}"
        # FOV mask
        fov_name = f"{num}_training_mask.gif"
        fov_path = f"training/mask/{fov_name}"
        entry = {
            "id": sample_id,
            "image_path": f"training/images/{img_name}",
            "mask_path": mask_path,
            "meta": {"dataset": "DRIVE", "split": "train"},
        }
        if os.path.exists(f"{root}/{fov_path}"):
            entry["fov_mask_path"] = fov_path
        entries.append(entry)

    # Test set
    test_imgs = f"{root}/test/images"
    for img_name in sorted(os.listdir(test_imgs)):
        if not img_name.endswith(".tif"):
            continue
        num = img_name.split("_")[0]
        sample_id = f"test_{num}"
        # Check for mask in test set
        mask_candidates = [
            f"test/mask/{num}_manual1.gif",
            f"test/1st_manual/{num}_manual1.gif",
        ]
        mask_path = None
        for mc in mask_candidates:
            if os.path.exists(f"{root}/{mc}"):
                mask_path = mc
                break
        if mask_path is None:
            # Try matching pattern
            test_mask_dir = f"{root}/test/mask"
            if os.path.isdir(test_mask_dir):
                for mf in os.listdir(test_mask_dir):
                    if mf.startswith(num):
                        mask_path = f"test/mask/{mf}"
                        break
        if mask_path is None:
            print(f"    WARN: no mask for DRIVE test {img_name}")
            continue

        entry = {
            "id": sample_id,
            "image_path": f"test/images/{img_name}",
            "mask_path": mask_path,
            "meta": {"dataset": "DRIVE", "split": "test"},
        }
        # FOV mask for test
        fov_candidates = [f"test/mask/{num}_test_mask.gif"]
        for fc in fov_candidates:
            if os.path.exists(f"{root}/{fc}"):
                entry["fov_mask_path"] = fc
                break
        entries.append(entry)

    write_manifest(entries, "drive")
    return entries


# ── CHASE_DB1 ────────────────────────────────────────────────────────────
def make_chase():
    root = f"{BASE}/CHASE"
    entries = []
    for img_name in sorted(os.listdir(f"{root}/images")):
        if not (img_name.endswith(".tif") or img_name.endswith(".jpg") or img_name.endswith(".png")):
            continue
        # Image: test_01_test.tif or training_09_test.tif
        stem = img_name.rsplit(".", 1)[0]  # e.g. "test_01_test" or "training_09_test"
        # Mask: same prefix but "_manual1" instead of "_test"
        # test_01_test -> test_01_manual1
        # training_09_test -> training_09_manual1
        mask_stem = stem.rsplit("_", 1)[0] + "_manual1"  # replace last part
        mask_name = f"{mask_stem}.tif"
        mask_path = f"masks/{mask_name}"
        if not os.path.exists(f"{root}/{mask_path}"):
            print(f"    WARN: no mask for CHASE {img_name} (tried {mask_path})")
            continue
        # Extract a clean id
        parts = stem.split("_")
        sample_id = f"chase_{parts[0]}_{parts[1]}"  # e.g. chase_test_01
        entries.append({
            "id": sample_id,
            "image_path": f"images/{img_name}",
            "mask_path": mask_path,
            "meta": {"dataset": "CHASE_DB1"},
        })
    write_manifest(entries, "chase")
    return entries


# ── STARE ────────────────────────────────────────────────────────────────
def make_stare():
    root = f"{BASE}/STARE"
    entries = []
    for img_name in sorted(os.listdir(f"{root}/images")):
        if not img_name.endswith(".ppm"):
            continue
        sample_id = img_name.replace(".ppm", "")
        # Mask: im0001.ah.ppm or im0001.ah.ppm.png
        mask_candidates = [
            f"masks/{sample_id}.ah.ppm",
            f"masks/{sample_id}.ah.ppm.png",
            f"masks/{sample_id}.vk.ppm",
        ]
        mask_path = None
        for mc in mask_candidates:
            if os.path.exists(f"{root}/{mc}"):
                mask_path = mc
                break
        if mask_path is None:
            print(f"    WARN: no mask for STARE {img_name}")
            continue
        entries.append({
            "id": sample_id,
            "image_path": f"images/{img_name}",
            "mask_path": mask_path,
            "meta": {"dataset": "STARE"},
        })
    write_manifest(entries, "stare")
    return entries


# ── OCTA-500 (LargeVessel segmentation) ──────────────────────────────────
def make_octa500():
    """OCTA-500 with GT_LargeVessel labels.
    
    500 subjects total, with en-face OCTA(ILM_OPL) projection maps in:
      - 3mm FOV (IDs 10001-10300): OCTA_6mm_part8/OCTA_6mm/Projection Maps/OCTA(ILM_OPL)/
      - 6mm FOV (IDs 10301-10500): OCTA_3mm_part3/OCTA_3mm/Projection Maps/OCTA(ILM_OPL)/
    
    Labels: Label/GT_LargeVessel/<id>.bmp
    
    Official baseline code (Code/2D Baselines/options/base_options.py) uses
    OCTA(ILM_OPL) as input modality at 400×400 resolution.
    """
    # All known projection map locations
    proj_dirs = [
        "OCTA_6mm_part8/OCTA_6mm/Projection Maps/OCTA(ILM_OPL)",  # 3mm IDs
        "OCTA_3mm_part3/OCTA_3mm/Projection Maps/OCTA(ILM_OPL)",  # 6mm IDs
    ]
    
    entries = []
    label_dir = f"{OCTA_BASE}/Label/GT_LargeVessel"
    
    for label_name in sorted(os.listdir(label_dir)):
        if not label_name.endswith(".bmp"):
            continue
        sample_id = label_name.replace(".bmp", "")
        sid = int(sample_id)
        fov = "3mm" if sid <= 10300 else "6mm"
        
        # Search all projection map locations
        proj_path = None
        for pdir in proj_dirs:
            candidate = f"{pdir}/{sample_id}.bmp"
            if os.path.exists(f"{OCTA_BASE}/{candidate}"):
                proj_path = candidate
                break
        
        if proj_path is None:
            continue
            
        entries.append({
            "id": f"octa_{sample_id}",
            "image_path": proj_path,
            "mask_path": f"Label/GT_LargeVessel/{label_name}",
            "meta": {"dataset": "OCTA-500", "subset": fov},
        })
    
    if entries:
        write_manifest(entries, "octa500")
    return entries


if __name__ == "__main__":
    print("Generating manifests for all retinal vessel datasets\n")
    
    print("FIVES (800 images):")
    fives = make_fives()
    
    print("\nDRIVE (40 images):")
    drive = make_drive()
    
    print("\nCHASE_DB1 (28 images):")
    chase = make_chase()
    
    print("\nSTARE (20 images):")
    stare = make_stare()
    
    print("\nOCTA-500:")
    octa = make_octa500()
    
    print(f"\n{'='*50}")
    print(f"Total manifests generated:")
    print(f"  FIVES:     {len(fives)} images")
    print(f"  DRIVE:     {len(drive)} images")
    print(f"  CHASE:     {len(chase)} images")
    print(f"  STARE:     {len(stare)} images")
    print(f"  OCTA-500:  {len(octa)} images")
