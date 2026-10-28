"""
External Web Agent Dataset Downloader & Ingestion Utility.
Downloads and integrates reliable web agent datasets:
1. WebLINX (McGill-NLP/WebLINX): Real-world multi-turn conversational web browsing traces across 150 websites.
2. WebArena (web-arena-x/webarena): Benchmark tasks and action references across 5 domains.
"""

import json
import logging
import os
from pathlib import Path
import sys
import urllib.request
import urllib.error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("external_dataset_downloader")

DATA_DIR = Path("data")
WEBARENA_DIR = DATA_DIR / "raw_webarena"
WEBLINX_DIR = DATA_DIR / "raw_weblinx"


def download_webarena_tasks() -> bool:
    """Downloads official WebArena task configuration and reference action definitions."""
    WEBARENA_DIR.mkdir(parents=True, exist_ok=True)
    target_file = WEBARENA_DIR / "test.raw.json"
    
    url = "https://raw.githubusercontent.com/web-arena-x/webarena/main/config_files/test.raw.json"
    logger.info("Downloading WebArena task definitions from %s...", url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "JED-M2-LAM/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            with open(target_file, "wb") as f:
                f.write(content)
        data = json.loads(content.decode("utf-8"))
        logger.info("Successfully downloaded WebArena tasks: %d tasks saved to %s", len(data), target_file)
        return True
    except Exception as e:
        logger.error("Failed to download WebArena tasks: %s", e)
        return False


def download_weblinx_metadata() -> bool:
    """Downloads WebLINX demonstration index and dialogue metadata from Hugging Face."""
    WEBLINX_DIR.mkdir(parents=True, exist_ok=True)
    splits_url = "https://huggingface.co/datasets/McGill-NLP/WebLINX/resolve/main/splits.json"
    target_file = WEBLINX_DIR / "splits.json"
    
    logger.info("Downloading WebLINX split definitions from %s...", splits_url)
    try:
        req = urllib.request.Request(splits_url, headers={"User-Agent": "JED-M2-LAM/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            with open(target_file, "wb") as f:
                f.write(content)
        data = json.loads(content.decode("utf-8"))
        logger.info("Successfully downloaded WebLINX splits: %s saved to %s", list(data.keys()), target_file)
        return True
    except Exception as e:
        logger.error("Failed to download WebLINX splits: %s", e)
        return False


if __name__ == "__main__":
    logger.info("Starting external datasets download...")
    wa_ok = download_webarena_tasks()
    wl_ok = download_weblinx_metadata()
    logger.info("Download summary: WebArena=%s, WebLINX=%s", wa_ok, wl_ok)
