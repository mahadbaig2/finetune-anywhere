"""Ask the local Qwen VLM about a single leaf photo."""

import argparse
from pathlib import Path

from evaluate import query

QUESTIONS = [
    ("Crop", "What plant species is shown in this image?"),
    ("Visible symptoms", "Which visible features or symptoms are present on this leaf?"),
    ("Possible diagnosis", "What is the most likely condition and what visible evidence supports it? "
                           "Say if the image alone is insufficient."),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("photo", type=Path)
    p.add_argument("--port", type=int, default=8172)
    args = p.parse_args()
    for label, question in QUESTIONS:
        answer, _ = query(args.port, args.photo, question, 160)
        print(f"{label}\n{answer}\n")


if __name__ == "__main__":
    main()
