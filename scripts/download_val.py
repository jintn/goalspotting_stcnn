"""
Download the SoccerNet validation split (224p videos + Labels-v2.json).

Usage:
    python download_val.py --password YOUR_NV_PASSWORD

Get your password by registering at https://www.soccer-net.org/data
"""

import argparse
from pathlib import Path

DATA_DIR = "/home/jinny/aspotting/valset"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--password", required=True, help="NV_PASSWORD from soccer-net.org")
    args = parser.parse_args()

    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        raise SystemExit("SoccerNet package not found. Run: pip install SoccerNet")

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=DATA_DIR)
    mySoccerNetDownloader.password = args.password

    print("Downloading Labels-v2.json for validation split...")
    mySoccerNetDownloader.downloadGames(
        files=["Labels-v2.json"],
        split=["valid"],
    )

    print("Downloading 224p videos for validation split...")
    mySoccerNetDownloader.downloadGames(
        files=["1_224p.mkv", "2_224p.mkv"],
        split=["valid"],
    )

    print("Done.")

if __name__ == "__main__":
    main()
