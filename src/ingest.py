import os
import zipfile
import shutil
import hashlib
from pathlib import Path

import requests

# CONFIG
DOWNLOAD_URL = "https://mulligan-server.onrender.com/download-all"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"
ZIP_PATH = DOWNLOADS_DIR / "mulligan_uploads.zip"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TEMP_DIR = PROJECT_ROOT / "data" / "_temp_extract"

DATA_EXTS = {".xlsx", ".xls", ".csv"}

# HELPER: HASH FILE CONTENT
def file_hash(filepath: Path, chunk_size: int = 8192) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

# STEP 1: DOWNLOAD ZIP
def download_zip(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

# STEP 2: EXTRACT ZIP
def extract_zip(zip_path: Path, extract_to: Path) -> None:
    if extract_to.exists():
        shutil.rmtree(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

# STEP 3: BUILD HASH SET OF EXISTING FILES
def get_existing_hashes(raw_dir: Path) -> set[str]:
    hashes = set()

    if not raw_dir.exists():
        return hashes

    for file in raw_dir.iterdir():
        if file.is_file() and file.suffix.lower() in DATA_EXTS:
            hashes.add(file_hash(file))

    return hashes

# STEP 4: COPY NEW FILES ONLY
def make_unique_destination(dest: Path) -> Path:
    if not dest.exists():
        return dest

    counter = 1
    while True:
        candidate = dest.parent / f"{dest.stem}_{counter}{dest.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1

def ingest_new_files(temp_dir: Path, raw_dir: Path) -> tuple[int, int]:
    existing_hashes = get_existing_hashes(raw_dir)

    new_count = 0
    skipped_count = 0

    for root, _, files in os.walk(temp_dir):
        for fname in files:
            fpath = Path(root) / fname

            if fpath.suffix.lower() not in DATA_EXTS:
                continue

            h = file_hash(fpath)

            if h in existing_hashes:
                skipped_count += 1
                continue

            dest = make_unique_destination(raw_dir / fname)
            shutil.copy2(fpath, dest)

            existing_hashes.add(h)
            new_count += 1

    return new_count, skipped_count

# CLEANUP
def cleanup(paths: list[Path]) -> None:
    for path in paths:
        if path.is_dir() and path.exists():
            shutil.rmtree(path)
        elif path.is_file() and path.exists():
            path.unlink()

# MAIN
def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print("Downloading zip from server...")
        download_zip(DOWNLOAD_URL, ZIP_PATH)

        print("Extracting zip...")
        extract_zip(ZIP_PATH, TEMP_DIR)

        print("Ingesting new Excel files...")
        new_count, skipped_count = ingest_new_files(TEMP_DIR, RAW_DIR)

        print(f"\nAdded {new_count} new files")
        print(f"Skipped {skipped_count} duplicates")

    except requests.RequestException as e:
        print(f"Download failed: {e}")
    except zipfile.BadZipFile:
        print("Downloaded file is not a valid zip archive.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        cleanup([TEMP_DIR, ZIP_PATH])

if __name__ == "__main__":
    main()