"""Create deterministic, leaf-disjoint PlantVillageVQA splits directly from the release ZIP."""

import argparse
from collections import Counter, defaultdict, deque
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
from zipfile import ZipFile

LEVELS = {
    category: level
    for level, categories in enumerate((
        ("Existence & Sanity Check", "Plant Species Identification", "General Health Assessment"),
        ("Visual Attribute Grounding", "Detailed Verification"),
        ("Specific Disease Identification", "Comprehensive Description", "Causal Reasoning",
         "Counterfactual Reasoning"),
    ), 1)
    for category in categories
}
CROPS = (
    "apple", "blueberry", "cherry", "corn", "grape", "orange", "peach", "pepper bell", "potato",
    "raspberry", "soybean", "squash", "strawberry", "tomato",
)
CONDITIONS = {
    "healthy": r"\bhealthy\b|no (?:signs of disease|symptoms of pathology)",
    "citrus greening": r"citrus greening|huanglongbing|\bhlb\b|liberibacter asiaticus",
    "tomato yellow leaf curl virus": r"yellow leaf curl|\btylcv\b",
    "tomato mosaic virus": r"tomato mosaic virus|\btomv\b",
    "late blight": r"late blight|phytophthora infestans",
    "early blight": r"early blight|alternaria solani",
    "bacterial spot": r"bacterial spot",
    "septoria leaf spot": r"septoria leaf spot",
    "spider mites": r"spider mites?",
    "target spot": r"target spot",
    "leaf mold": r"leaf mold|olive-green mold",
    "powdery mildew": r"powdery mildew",
    "black rot": r"black rot",
    "esca": r"esca|black measles",
    "isariopsis leaf spot": r"isariopsis leaf spot",
    "leaf scorch": r"leaf scorch",
    "common rust": r"common rust",
    "northern leaf blight": r"northern leaf blight",
    "apple scab": r"apple scab",
    "gray leaf spot": r"cercospora leaf spot|gr[ae]y leaf spot",
    "cedar apple rust": r"cedar apple rust",
}


def condition(text):
    matches = [label for label, pattern in CONDITIONS.items() if re.search(pattern, text.lower())]
    return matches[0] if len(matches) == 1 else None


def labels(rows):
    crops, conditions = set(), set()
    for row in rows:
        category, answer = row["category"], row["answer"].lower()
        if category == "Plant Species Identification" and answer in CROPS:
            crops.add(answer)
        if category in {"Specific Disease Identification", "Comprehensive Description"}:
            if label := condition(answer):
                conditions.add(label)
            for crop in CROPS:
                if re.search(rf"\b{crop} leaf\b", answer):
                    crops.add(crop)
    return (next(iter(crops)) if len(crops) == 1 else "unknown",
            next(iter(conditions)) if len(conditions) == 1 else "unknown")


def load(archive):
    members = {}
    for member in archive.infolist():
        name = Path(member.filename)
        if name.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            key = name.name.lower()
            if key in members:
                raise ValueError(f"Duplicate ZIP image basename: {key}")
            members[key] = member
    groups = defaultdict(list)
    ids = set()
    annotation, = [name for name in archive.namelist() if Path(name).name == "PlantVillageVQA.csv"]
    with io.TextIOWrapper(archive.open(annotation), encoding="utf-8-sig", newline="") as stream:
        for index, raw in enumerate(csv.DictReader(stream)):
            image_id = raw["image_id"].strip().lower()
            if Path(raw["image_path"]).name.lower() != image_id:
                raise ValueError(f"ID and path disagree for annotation {index}")
            member = members[image_id]
            # Identical bytes share CRC and size. Treat collisions conservatively as one group.
            key = (member.CRC, member.file_size)
            category = raw["question_type"].strip()
            groups[key].append({
                "row_id": index, "image_id": image_id, "category": category, "level": LEVELS[category],
                "question": raw["question"].strip(), "answer": raw["answer"].strip(),
            })
            ids.add(image_id)
    return members, groups, ids


def partition(groups, rng):
    strata = defaultdict(list)
    for key, rows in sorted(groups.items()):
        crop, health = labels(rows)
        for row in rows:
            row.update(crop=crop, condition=health)
        strata[crop, health].append(key)
    pools = {"train": [], "dev": [], "test": []}
    for _, keys in sorted(strata.items()):
        rng.shuffle(keys)
        held = max(1, len(keys) // 6) if len(keys) >= 3 else 0
        pools["dev"].extend(keys[:held])
        pools["test"].extend(keys[held:2 * held])
        pools["train"].extend(keys[2 * held:])
    return pools


def sample(keys, groups, count, rng):
    buckets = defaultdict(list)
    for key in keys:
        for row in groups[key]:
            if row["question"] and row["answer"]:
                buckets[row["crop"], row["condition"], row["category"]].append(row)
    order = sorted(buckets, key=lambda key: (len(buckets[key]), key))
    turns = deque()
    for key in order:
        rng.shuffle(buckets[key])
        turns.append(buckets[key])
    result, seen = [], set()
    # Reserve one row per observed crop/condition stratum and per QA category.
    represented_strata, represented_categories = set(), set()
    for crop, health, category in order:
        stratum = (crop, health)
        if stratum not in represented_strata or category not in represented_categories:
            row = buckets[crop, health, category].pop()
            seen.add((row["image_id"], row["question"], row["answer"]))
            result.append(row)
            represented_strata.add(stratum)
            represented_categories.add(category)
    while turns and len(result) < count:
        bucket = turns.popleft()
        while bucket:
            row = bucket.pop()
            identity = (row["image_id"], row["question"], row["answer"])
            if identity not in seen:
                seen.add(identity)
                result.append(row)
                break
        if bucket:
            turns.append(bucket)
    if len(result) != count or {row["level"] for row in result} != {1, 2, 3}:
        raise ValueError("Sample could not meet requested size and all three QA levels")
    # Interleave categories so a short smoke run exercises multiple skills.
    by_category = defaultdict(list)
    for row in result:
        by_category[row["category"]].append(row)
    ordered = []
    while by_category:
        for category in sorted(by_category):
            ordered.append(by_category[category].pop())
            if not by_category[category]:
                del by_category[category]
    return ordered


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=int, default=4000)
    p.add_argument("--dev", type=int, default=300)
    p.add_argument("--test", type=int, default=350)
    a = p.parse_args()
    rng = random.Random(a.seed)
    with ZipFile(a.archive) as archive:
        members, groups, ids = load(archive)
        pools = partition(groups, rng)
        selected = {split: sample(keys, groups, getattr(a, split), rng) for split, keys in pools.items()}
        ownership = {}
        statistics = {}
        a.out.resolve().mkdir(parents=True, exist_ok=False)
        images = a.out.resolve() / "images"
        images.mkdir()
        for split, rows in selected.items():
            with (a.out / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    image_id = row["image_id"]
                    path = images / image_id
                    if not path.exists():
                        path.write_bytes(archive.read(members[image_id]))
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    if ownership.setdefault(digest, split) != split:
                        raise ValueError(f"Duplicate image bytes across splits: {image_id}")
                    if split == "test":
                        record = {**row, "image": f"images/{image_id}"}
                    else:
                        record = {
                            "images": [str(path)],
                            "messages": [
                                {"role": "user", "content": "<image>" + row["question"]},
                                {"role": "assistant", "content": row["answer"]},
                            ],
                        }
                    print(json.dumps(record, ensure_ascii=False), file=stream)
            statistics[split] = {
                "qa_pairs": len(rows),
                "unique_image_ids": len({row["image_id"] for row in rows}),
                "image_ids": sorted({row["image_id"] for row in rows}),
                "categories": dict(Counter(row["category"] for row in rows)),
                "inferred_crop_condition_counts": dict(Counter(
                    f"{row['crop']}/{row['condition']}" for row in rows
                )),
            }
    with a.archive.open("rb") as stream:
        archive_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    (a.out / "split.json").write_text(json.dumps({
        "archive_sha256": archive_sha, "seed": a.seed, "source_qa_pairs": sum(map(len, groups.values())),
        "source_image_ids_with_annotations": len(ids), "splits": statistics,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
