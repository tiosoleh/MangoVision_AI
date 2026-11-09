from pathlib import Path
from collections import Counter

for split in ["train", "val", "test"]:
    folder = Path("dataset/labels") / split
    counter = Counter()

    for file in folder.glob("*.txt"):
        for line in file.read_text().splitlines():
            if line.strip():
                class_id = int(line.split()[0])
                counter[class_id] += 1

    print(f"\n{split.upper()}:")
    print(dict(sorted(counter.items())))