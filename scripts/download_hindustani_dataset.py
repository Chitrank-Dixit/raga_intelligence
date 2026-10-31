"""
Downloads the real Hindustani classical audio dataset for the 10 canonical ragas
from Hugging Face (neerajaabhyankar/hindustani-raag-small).
"""

import sys
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_DIRS = [
    "Bhairav",
    "Bhairavi",
    "Yaman",
    "Bhoopali",
    "Deshkar",
    "Bageshree",
    "Malkauns",
    "Todi",
    "DarbariKanada",
    "Kafi",
]

def download_dataset(output_dir: str = "./data/hindustani_dataset"):
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    
    allow_patterns = [f"{r}/*.mp3" for r in CANONICAL_DIRS] + ["metadata.csv"]
    logger.info(f"Downloading 10 canonical ragas subset into {p}...")
    
    local_dir = snapshot_download(
        repo_id="neerajaabhyankar/hindustani-raag-small",
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=output_dir,
        max_workers=8,
    )
    
    logger.info(f"Download complete! Saved to {local_dir}")
    mp3_files = list(Path(local_dir).rglob("*.mp3"))
    logger.info(f"Verified {len(mp3_files)} audio clips downloaded on disk.")

if __name__ == "__main__":
    download_dataset()

