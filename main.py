from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import shutil

from PIL import Image



QUERY = "super mario movie trailer"

OUTPUT_ROOT = Path(r"C:\Users\ariel\Desktop\ASDUA\ass2ex2 scene images")

TARGET_MIN_SCENES = 50
TARGET_MAX_SCENES = 80

MIN_SCENE_LEN_FRAMES = 12

THRESHOLDS_TO_TRY = [
    10.0, 8.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.7, 2.5,
    2.3, 2.1, 2.0, 1.8, 1.6, 1.4, 1.2
]




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
    for p in out_dir.glob("*.jpg"):
        p.unlink(missing_ok=True)
    for p in out_dir.glob("*.png"):
        p.unlink(missing_ok=True)

    cmd = [
        sys.executable, "-m", "scenedetect",
        "-i", str(video_path),
        "detect-adaptive",
        "--threshold", str(threshold),
        "--min-scene-len", str(min_scene_len),
        "save-images",
        "-o", str(out_dir),
    ]
    run(cmd)


def count_scene_images(out_dir: Path) -> int:
    return len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))


def auto_tune_and_save(video_path: Path, scenes_root: Path) -> Path:
    created_dirs: list[Path] = []

    best_dir = None
    best_dist = None
    best_count = -1

    for thr in THRESHOLDS_TO_TRY:
        out_dir = scene_output_dir(scenes_root, video_path, thr, MIN_SCENE_LEN_FRAMES)
        created_dirs.append(out_dir)

        run_scenedetect_save_images(video_path, out_dir, thr, MIN_SCENE_LEN_FRAMES)
        n = count_scene_images(out_dir)

        print(f"Threshold={thr} -> {n} scene images")

        if TARGET_MIN_SCENES <= n <= TARGET_MAX_SCENES:
            best_dir = out_dir
            break

        dist = (TARGET_MIN_SCENES - n) if n < TARGET_MIN_SCENES else (n - TARGET_MAX_SCENES)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_dir = out_dir
            best_count = n

    assert best_dir is not None

    
    for d in created_dirs:
        if d != best_dir and d.exists():
            shutil.rmtree(d, ignore_errors=True)

    if not (TARGET_MIN_SCENES <= count_scene_images(best_dir) <= TARGET_MAX_SCENES):
        print(f"\n⚠️ Didn't hit exact target range. Closest was {best_count} images in: {best_dir}")

    return best_dir





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


def analyze_scene_diversity(images_dir: Path,
                            near_dup_threshold: int = 6) -> None:
    imgs = sorted(
        list(images_dir.glob("*.jpg")) +
        list(images_dir.glob("*.png"))
    )
    if not imgs:
        print("No images to analyze.")
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
    print("Near-duplicate pairs:", len(near_dup_pairs))

    for a, b, d in near_dup_pairs[:10]:
        print(f"  similar: {a} <-> {b} (distance={d})")

    if not near_dup_pairs:
        print("✅ Scene images look meaningfully different.")
    else:
        print("⚠️ Some consecutive scenes are very similar.")
        print("   Consider raising threshold or min_scene_len.")




def main() -> None:
    downloads_dir, scenes_root = ensure_dirs()

    print("Output root:", OUTPUT_ROOT)
    print("Downloading query:", QUERY)

    video_path = download_youtube_query(QUERY, downloads_dir)
    print("Downloaded video:", video_path)

    print("\nDetecting scenes and saving images...")
    chosen_dir = auto_tune_and_save(video_path, scenes_root)

    analyze_scene_diversity(chosen_dir)

    print("\n✅ DONE")
    print("Scene images folder:", chosen_dir)


if __name__ == "__main__":
    main()
