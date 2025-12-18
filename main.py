from __future__ import annotations

import json
import re
import shutil
import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import os
from PIL import Image
import math
from rapidfuzz import fuzz

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

FIXED_THRESHOLD = 2.0

MAX_CAPTIONS = 5

QUERY = "super mario movie trailer"


OUTPUT_ROOT = Path(r"C:\Users\ariel\Desktop\ASDUA\ass2ex2 scene images")


CAPTIONS_JSON = OUTPUT_ROOT / "scene_captions.json"

FFMPEG_EXE = r"C:\Users\ariel\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

COLLAGE_PATH = OUTPUT_ROOT / "collage.png"
COLLAGE_THUMB = (256, 256)  
FUZZ_THRESHOLD = 75        

MIN_SCENE_LEN_FRAMES = 12

YTDLP_FORMAT = "bestvideo[height<=1080]+bestaudio/best[height<=1080]"

MOONDREAM_MODEL_ID = "vikhyatk/moondream2"
MOONDREAM_REVISION = None


CAPTION_LENGTH = "short"


NEAR_DUP_THRESHOLD = 6



def run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def ensure_dirs() -> Tuple[Path, Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    downloads_dir = OUTPUT_ROOT / "downloads"
    scenes_dir = OUTPUT_ROOT / "scenes"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    return downloads_dir, scenes_dir


def download_youtube_query(query: str, downloads_dir: Path) -> Path:
    """
    Downloads the first YouTube search result for `query` using yt-dlp.
    Returns the downloaded MP4 file path.
    """
    outtmpl = str(downloads_dir / "%(title).200s [%(id)s].%(ext)s")
    



    cmd = [
        sys.executable, "-m", "yt_dlp",
        f"ytsearch1:{query}",
        "-o", outtmpl,
        "-f", YTDLP_FORMAT,
        "--merge-output-format", "mp4",
        "--no-playlist",
    ]
    run(cmd)

    mp4s = sorted(downloads_dir.glob("*.mp4"),
                  key=lambda p: p.stat().st_mtime,
                  reverse=True)
    if not mp4s:
        raise FileNotFoundError("No MP4 found after yt-dlp finished.")
    return mp4s[0]


def scene_output_dir(scenes_root: Path, video_path: Path,
                     threshold: float, min_len: int) -> Path:
    safe_stem = "".join(c for c in video_path.stem if c not in r'<>:"/\|?*')
    out = scenes_root / f"{safe_stem}_adaptive_thr{threshold}_min{min_len}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_scenedetect_save_images(video_path: Path,
                                out_dir: Path,
                                threshold: float,
                                min_scene_len: int) -> None:
    
    for p in list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png")):
        p.unlink(missing_ok=True)

    cmd = [
        sys.executable, "-m", "scenedetect",
        "-i", str(video_path),
        "detect-adaptive",
        "--threshold", str(threshold),
        "--min-scene-len", str(min_scene_len),
        "save-images",
        "-o", str(out_dir),
        "--num-images", "1",  
    ]
    run(cmd)



def count_scene_images(out_dir: Path) -> int:
    return len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))


def run_scenedetect_fixed(video_path: Path, scenes_root: Path) -> Path:
    """
    Run PySceneDetect once with a fixed threshold.
    Saves exactly one image per scene.
    """
    out_dir = scene_output_dir(
        scenes_root,
        video_path,
        FIXED_THRESHOLD,
        MIN_SCENE_LEN_FRAMES
    )

    run_scenedetect_save_images(
        video_path,
        out_dir,
        FIXED_THRESHOLD,
        MIN_SCENE_LEN_FRAMES
    )

    n = count_scene_images(out_dir)
    print(f"\nFixed threshold={FIXED_THRESHOLD} -> {n} scene images")

    return out_dir





def ahash(path: Path, size: int = 8) -> int:
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size))
        
        pixels = list(im.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, px in enumerate(pixels):
        if px >= avg:
            bits |= (1 << i)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def analyze_scene_diversity(images_dir: Path, near_dup_threshold: int = 6) -> None:
    imgs = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not imgs:
        return

    hashes = [(p, ahash(p)) for p in imgs]
    near_dup_pairs = []

    
    for i in range(1, len(hashes)):
        p_prev, h_prev = hashes[i - 1]
        p_cur, h_cur = hashes[i]
        dist = hamming_distance(h_prev, h_cur)
        if dist <= near_dup_threshold:
            near_dup_pairs.append((p_prev.name, p_cur.name, dist))

    print("\n--- Scene diversity report ---")
    print("Total images:", len(imgs))
    print("Near-duplicate consecutive pairs:", len(near_dup_pairs))
    for a, b, d in near_dup_pairs[:10]:
        print(f"  similar: {a} <-> {b} (distance={d})")

    if not near_dup_pairs:
        print("✅ Scene images look meaningfully different.")
    else:
        print("⚠️ Some consecutive scenes are very similar.")
        print("   Tip: raise threshold slightly or raise MIN_SCENE_LEN_FRAMES.")



def load_moondream():
    """
    Loads Moondream locally (downloads weights once, then cached by HF).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading Moondream on device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        MOONDREAM_MODEL_ID,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MOONDREAM_MODEL_ID)

    
    model.to(device)
    
    model.eval()

    return model, tokenizer, device


def list_scene_images(images_dir: Path) -> List[Path]:
    imgs = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not imgs:
        raise FileNotFoundError(f"No scene images found in {images_dir}")
    return imgs


def extract_scene_number(p: Path) -> int:
    """
    Try to extract a scene number from filename.
    If not possible, fallback to first number found; else -1.
    """
    name = p.stem

    
    m = re.search(r"(?:scene|Scene)[^\d]*0*(\d+)", name)
    if m:
        return int(m.group(1))

    m = re.search(r"0*(\d+)", name)
    if m:
        return int(m.group(1))

    return -1


def caption_scenes(images_dir: Path) -> Dict[str, str]:
    """
    Returns dict mapping scene_number (string) -> caption
    """
    model, _tokenizer, device = load_moondream()

    imgs = list_scene_images(images_dir)

    
    imgs_sorted = sorted(imgs, key=lambda p: (extract_scene_number(p), p.name))
    
    
    if MAX_CAPTIONS is not None and len(imgs_sorted) > MAX_CAPTIONS:
        step = max(1, len(imgs_sorted) // MAX_CAPTIONS)
        imgs_sorted = imgs_sorted[::step][:MAX_CAPTIONS]


    captions: Dict[str, str] = {}

    
    fallback_counter = 1

    for img_path in imgs_sorted:
        
        scene_num = extract_scene_number(img_path)
        if scene_num == -1:
            scene_num = fallback_counter
            fallback_counter += 1

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im.thumbnail((256, 256))
            im = im.copy()

        
        
        with torch.inference_mode():
            result = model.caption(im, length=CAPTION_LENGTH)
        caption_text = result["caption"] if isinstance(result, dict) and "caption" in result else str(result)

        captions[str(scene_num)] = caption_text
        print(f"Scene {scene_num}: {caption_text}")

    return captions


def save_captions_json(captions: Dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)



def load_captions_json(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def find_scenes_exact(captions: Dict[str, str], word: str) -> List[str]:
    w = word.strip().lower()
    hits = []
    for scene_id, cap in captions.items():
        if w in str(cap).lower():
            hits.append(str(scene_id))
    return hits

def find_scenes_fuzzy(captions: Dict[str, str], word: str, threshold: int = FUZZ_THRESHOLD) -> List[str]:
    
    w = word.strip().lower()
    hits = []
    for scene_id, cap in captions.items():
        text = str(cap).lower()
        score = fuzz.partial_ratio(w, text)
        if score >= threshold:
            hits.append(str(scene_id))
    return hits

def find_image_for_scene(images_dir: Path, scene_id: str) -> Optional[Path]:
    
    sid = int(scene_id)
    patterns = [
        f"*Scene-{sid:03d}-*.jpg",
        f"*Scene-{sid:03d}-*.png",
        f"*scene-{sid:03d}-*.jpg",
        f"*scene-{sid:03d}-*.png",
    ]
    for pat in patterns:
        matches = sorted(images_dir.glob(pat))
        if matches:
            return matches[0]
    return None

def make_collage(image_paths: List[Path], out_path: Path, thumb_size=COLLAGE_THUMB) -> None:
    if not image_paths:
        raise ValueError("No images to build collage from.")

    thumbs = []
    for p in image_paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail(thumb_size)
            thumbs.append(im.copy())

    n = len(thumbs)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    cell_w, cell_h = thumb_size
    collage = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))

    for idx, im in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        x = c * cell_w
        y = r * cell_h

        
        ox = x + (cell_w - im.width) // 2
        oy = y + (cell_h - im.height) // 2
        collage.paste(im, (ox, oy))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    collage.save(out_path)
    



def search_loop(images_dir: Path, captions_path: Path) -> None:
    captions = load_captions_json(captions_path)

    while True:
        word = input("\nSearch the video using a word: ").strip()
        if not word:
            print("Empty input. Exiting search.")
            return

        exact = find_scenes_exact(captions, word)

        fuzzy = find_scenes_fuzzy(captions, word, threshold=FUZZ_THRESHOLD)

        hits = fuzzy if fuzzy else exact

        if not hits:
            print("No matching scenes found.")
            continue

        print(f"Found {len(hits)} scenes: {', '.join(hits)}")

        image_paths = []
        missing = []
        for sid in hits:
            p = find_image_for_scene(images_dir, sid)
            if p:
                image_paths.append(p)
            else:
                missing.append(sid)

        if missing:
            print("Warning: missing images for scenes:", ", ".join(missing))

        if not image_paths:
            print("No images were found for the matched scenes.")
            continue

        make_collage(image_paths, COLLAGE_PATH)
        print(f"✅ Saved collage to: {COLLAGE_PATH}")
        os.startfile(str(COLLAGE_PATH))




def main() -> None:
    downloads_dir, scenes_root = ensure_dirs()

   
    if CAPTIONS_JSON.exists():
        
        scene_dirs = [p for p in scenes_root.iterdir() if p.is_dir()]
        if not scene_dirs:
            print(f"Found {CAPTIONS_JSON} but no scenes folders in {scenes_root}.")
            return
        chosen_images_dir = max(scene_dirs, key=lambda p: p.stat().st_mtime)
        print(f"✅ Found existing {CAPTIONS_JSON}. Skipping pipeline.")
        print("Using scenes folder:", chosen_images_dir)
        search_loop(chosen_images_dir, CAPTIONS_JSON)
        return


    print("Output root:", OUTPUT_ROOT)
    print("Downloading query:", QUERY)

    video_path = download_youtube_query(QUERY, downloads_dir)
    print("Downloaded video:", video_path)

    print("\nDetecting scenes and saving images...")
    chosen_images_dir = run_scenedetect_fixed(video_path, scenes_root)

    analyze_scene_diversity(chosen_images_dir, near_dup_threshold=NEAR_DUP_THRESHOLD)

    print("\nCaptioning each scene image with Moondream...")
    captions = caption_scenes(chosen_images_dir)

    print(f"\nSaving captions JSON to: {CAPTIONS_JSON}")
    save_captions_json(captions, CAPTIONS_JSON)

    print("\n✅ DONE generating captions.")
    search_loop(chosen_images_dir, CAPTIONS_JSON)


if __name__ == "__main__":
    main()
