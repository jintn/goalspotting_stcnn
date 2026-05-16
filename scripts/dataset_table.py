import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "dataset"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "soccernet_class_distribution.csv"
DEFAULT_SPLIT_MANIFEST_PATH = Path(__file__).resolve().parent / "manifests" / "manifest_goal_vs_bg.csv"
INCLUDED_LABELS = ("goal", "shots off target", "shots on target")

def extract_event_label(label):
    """
    Extract clean class name from SoccerNet label string.
    Example: 'Goal' or 'Shot on target'
    """
    return label.strip().lower()


def relative_game_path(dataset_path, label_path):
    return str(label_path.parent.relative_to(dataset_path))


def load_split_map(manifest_path, dataset_path):
    split_map = {}
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        return split_map

    with manifest_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            split = (row.get("split") or "").strip().lower()
            rel_game = (row.get("rel_game") or "").strip()

            if not rel_game:
                match_dir = (row.get("match_dir") or "").strip()
                if match_dir:
                    marker = f"{dataset_path.name}/"
                    if marker in match_dir:
                        rel_game = match_dir.split(marker, 1)[1]

            if split and rel_game:
                split_map[rel_game] = split

    return split_map


def process_dataset(dataset_path, split_map=None):
    class_counter = Counter()
    split_class_counters = defaultdict(Counter)
    split_game_counters = Counter()
    game_counter = 0

    dataset_path = Path(dataset_path)
    split_map = split_map or {}
    for filepath in dataset_path.rglob("Labels-v2.json"):
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        split = split_map.get(relative_game_path(dataset_path, filepath), "unassigned")
        game_counter += 1
        split_game_counters[split] += 1

        for ann in annotations:
            label = extract_event_label(ann.get("label", "unknown"))
            class_counter[label] += 1
            split_class_counters[split][label] += 1

    return class_counter, game_counter, split_class_counters, split_game_counters


def create_table(class_counter):
    filtered_counter = Counter(
        {
            label: count
            for label, count in class_counter.items()
            if label in INCLUDED_LABELS
        }
    )
    total = sum(filtered_counter.values())
    rows = []
    for label, count in filtered_counter.most_common():
        percentage = (count / total) * 100 if total else 0.0
        rows.append({"Class": label, "Count": count, "Percentage": percentage})
    return rows


def print_table(rows):
    class_width = max(len("Class"), *(len(row["Class"]) for row in rows))
    count_width = max(len("Count"), *(len(str(row["Count"])) for row in rows))
    pct_width = len("Percentage")

    header = (
        f"{'Class':<{class_width}}  "
        f"{'Count':>{count_width}}  "
        f"{'Percentage':>{pct_width}}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['Class']:<{class_width}}  "
            f"{row['Count']:>{count_width}}  "
            f"{row['Percentage']:>{pct_width}.2f}"
        )


def save_table(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Class", "Count", "Percentage"])
        writer.writeheader()
        writer.writerows(rows)


def print_split_summary(split_game_counters, split_class_counters):
    total_games = sum(split_game_counters.values())
    total_events = sum(
        sum(count for label, count in counter.items() if label in INCLUDED_LABELS)
        for counter in split_class_counters.values()
    )

    print("\nSplit summary\n")
    print("Split       Games  Events  Game %  Event %")
    print("------------------------------------------")
    for split in sorted(split_game_counters):
        games = split_game_counters[split]
        events = sum(
            count
            for label, count in split_class_counters[split].items()
            if label in INCLUDED_LABELS
        )
        game_pct = (games / total_games) * 100 if total_games else 0.0
        event_pct = (events / total_events) * 100 if total_events else 0.0
        print(f"{split:<10}{games:>6}{events:>8}{game_pct:>8.2f}{event_pct:>9.2f}")


def save_split_summary(split_game_counters, split_class_counters, output_path):
    split_output_path = output_path.with_name(f"{output_path.stem}_splits{output_path.suffix}")
    total_games = sum(split_game_counters.values())
    total_events = sum(
        sum(count for label, count in counter.items() if label in INCLUDED_LABELS)
        for counter in split_class_counters.values()
    )

    with split_output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["Split", "Games", "Events", "GamePercentage", "EventPercentage"],
        )
        writer.writeheader()
        for split in sorted(split_game_counters):
            games = split_game_counters[split]
            events = sum(
                count
                for label, count in split_class_counters[split].items()
                if label in INCLUDED_LABELS
            )
            writer.writerow(
                {
                    "Split": split,
                    "Games": games,
                    "Events": events,
                    "GamePercentage": (games / total_games) * 100 if total_games else 0.0,
                    "EventPercentage": (events / total_events) * 100 if total_events else 0.0,
                }
            )

    return split_output_path


def print_split_tables(split_class_counters):
    for split in sorted(split_class_counters):
        print(f"\nTop classes for split: {split}\n")
        print_table(create_table(split_class_counters[split]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a class distribution table from SoccerNet Labels-v2.json files."
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=str(DEFAULT_DATASET_PATH),
        help=f"Path to the dataset root. Defaults to {DEFAULT_DATASET_PATH}",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"CSV output path. Defaults to {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--split-manifest",
        default=str(DEFAULT_SPLIT_MANIFEST_PATH),
        help=(
            "CSV manifest with split metadata. "
            f"Defaults to {DEFAULT_SPLIT_MANIFEST_PATH}"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    split_manifest_path = Path(args.split_manifest).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    split_map = load_split_map(split_manifest_path, dataset_path)
    class_counter, num_games, split_class_counters, split_game_counters = process_dataset(
        dataset_path, split_map=split_map
    )
    if not class_counter:
        raise RuntimeError(f"No annotations found under: {dataset_path}")

    rows = create_table(class_counter)

    print(f"\nProcessed {num_games} label files from {dataset_path}\n")
    print_table(rows)
    print_split_summary(split_game_counters, split_class_counters)
    print_split_tables(split_class_counters)

    # Save to CSV for LaTeX / Excel
    save_table(rows, output_path)
    split_output_path = save_split_summary(split_game_counters, split_class_counters, output_path)

    print(f"\nSaved table to {output_path}")
    print(f"Saved split summary to {split_output_path}")
