#!/usr/bin/env python
"""Build a tiny T5-tokenized short-QA dataset for ELF flow overfitting.

The saved dataset matches the existing text conditional loader:
  - condition_input_ids: prompt tokens
  - input_ids: target tokens, optionally with EOS appended
  - input / target / index: human-readable metadata for probes
"""

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


FACTS = [
    ("What color is the sky on a clear day?", "blue"),
    ("What animal says meow?", "cat"),
    ("What animal says bark?", "dog"),
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What planet do humans live on?", "Earth"),
    ("What do bees make?", "honey"),
    ("What do people use to tell time?", "clock"),
    ("What season comes after spring?", "summer"),
    ("What shape has three sides?", "triangle"),
    ("What gas do humans need to breathe?", "oxygen"),
    ("What is frozen water called?", "ice"),
    ("What is the opposite of hot?", "cold"),
    ("What is the opposite of up?", "down"),
    ("What is the opposite of left?", "right"),
    ("What is the opposite of yes?", "no"),
    ("What is the first month of the year?", "January"),
    ("What is the last month of the year?", "December"),
    ("What is the day after Monday?", "Tuesday"),
    ("What is the day before Friday?", "Thursday"),
    ("What fruit is yellow and curved?", "banana"),
    ("What fruit is red and often used in pies?", "apple"),
    ("What vehicle runs on rails?", "train"),
    ("What vehicle flies in the sky?", "airplane"),
    ("What do you drink when you are thirsty?", "water"),
    ("What do you wear on your feet?", "shoes"),
    ("What room is used for cooking?", "kitchen"),
    ("What tool cuts paper?", "scissors"),
    ("What device takes photographs?", "camera"),
    ("What language is commonly spoken in Spain?", "Spanish"),
    ("What is two plus two?", "4"),
    ("What is five plus seven?", "12"),
]

CLASSIFY = [
    ("Classify the sentiment: I loved the movie.", "positive"),
    ("Classify the sentiment: The food was terrible.", "negative"),
    ("Classify the sentiment: The book was okay.", "neutral"),
    ("Answer yes or no: Is fire hot?", "yes"),
    ("Answer yes or no: Can fish usually fly?", "no"),
    ("Answer yes or no: Is ice cold?", "yes"),
    ("Answer yes or no: Is the moon made of cheese?", "no"),
    ("Choose animal or object: tiger", "animal"),
    ("Choose animal or object: table", "object"),
    ("Choose animal or object: dolphin", "animal"),
    ("Choose animal or object: pencil", "object"),
    ("Choose fruit or color: orange", "fruit"),
    ("Choose fruit or color: purple", "color"),
    ("Choose fruit or color: grape", "fruit"),
    ("Choose fruit or color: green", "color"),
    ("Choose indoor or outdoor: bedroom", "indoor"),
    ("Choose indoor or outdoor: garden", "outdoor"),
    ("Choose indoor or outdoor: office", "indoor"),
    ("Choose indoor or outdoor: beach", "outdoor"),
    ("Return only the label. Text: The answer is correct.", "correct"),
    ("Return only the label. Text: The answer is wrong.", "wrong"),
    ("Return only the label. Text: The system passed.", "pass"),
    ("Return only the label. Text: The system failed.", "fail"),
    ("Classify as math or language: seven plus eight", "math"),
    ("Classify as math or language: write a sentence", "language"),
    ("Classify as math or language: divide ten by two", "math"),
    ("Classify as math or language: spell the word cat", "language"),
    ("Pick small or large: elephant", "large"),
    ("Pick small or large: ant", "small"),
    ("Pick small or large: whale", "large"),
    ("Pick small or large: coin", "small"),
    ("Answer true or false: Water is wet.", "true"),
]

TRANSFORMS = [
    ("Repeat exactly: alpha", "alpha"),
    ("Repeat exactly: bravo", "bravo"),
    ("Repeat exactly: comet", "comet"),
    ("Repeat exactly: delta", "delta"),
    ("Lowercase this word: HELLO", "hello"),
    ("Lowercase this word: WORLD", "world"),
    ("Lowercase this word: TRAIN", "train"),
    ("Lowercase this word: MODEL", "model"),
    ("Uppercase this word: apple", "APPLE"),
    ("Uppercase this word: river", "RIVER"),
    ("Uppercase this word: quiet", "QUIET"),
    ("Uppercase this word: token", "TOKEN"),
    ("Return the first word: red blue green", "red"),
    ("Return the first word: cat dog bird", "cat"),
    ("Return the last word: red blue green", "green"),
    ("Return the last word: cat dog bird", "bird"),
    ("Complete the pair: left -> right, up ->", "down"),
    ("Complete the pair: day -> night, hot ->", "cold"),
    ("Complete the pair: open -> close, start ->", "finish"),
    ("Complete the pair: question -> answer, problem ->", "solution"),
    ("Write the short form of United States.", "US"),
    ("Write the short form of European Union.", "EU"),
    ("Write the short form of artificial intelligence.", "AI"),
    ("Write the short form of New York.", "NY"),
    ("Return only the number of words: red blue", "2"),
    ("Return only the number of words: one two three", "3"),
    ("Return only the number of letters in cat.", "3"),
    ("Return only the number of letters in apple.", "5"),
    ("Return the missing letter: a b c _", "d"),
    ("Return the missing letter: w x y _", "z"),
    ("Return the next number: 2 4 6", "8"),
    ("Return the next number: 3 6 9", "12"),
]


def arithmetic_examples():
    rows = []
    for a, b in [
        (1, 1), (2, 3), (4, 5), (6, 7), (8, 2), (9, 4), (10, 5), (11, 6),
        (12, 8), (13, 7), (14, 3), (15, 9), (16, 4), (17, 5), (18, 6), (19, 1),
    ]:
        rows.append((f"Compute {a} plus {b}. Return only the number.", str(a + b)))
    for a, b in [
        (5, 2), (7, 3), (9, 4), (10, 6), (12, 5), (14, 9), (15, 8), (18, 7),
        (20, 11), (21, 12), (24, 10), (25, 13), (30, 15), (32, 14), (40, 17), (50, 25),
    ]:
        rows.append((f"Compute {a} minus {b}. Return only the number.", str(a - b)))
    return rows


COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange", "black", "white",
    "pink", "gray", "brown", "silver", "gold", "violet", "indigo", "cyan",
]

ANIMALS = [
    "cat", "dog", "tiger", "lion", "bear", "horse", "cow", "sheep", "goat",
    "whale", "dolphin", "shark", "eagle", "sparrow", "owl", "ant", "bee",
    "butterfly", "rabbit", "fox", "zebra", "giraffe", "monkey", "panda",
]

OBJECTS = [
    "table", "chair", "pencil", "phone", "camera", "clock", "book", "lamp",
    "shoe", "cup", "plate", "door", "window", "key", "coin", "bottle",
    "paper", "scissors", "bag", "box", "desk", "bed", "mirror", "brush",
]

FRUITS = [
    "apple", "banana", "grape", "orange", "peach", "pear", "melon", "lemon",
    "lime", "mango", "kiwi", "plum", "cherry", "berry", "apricot", "fig",
]

PLACES = [
    "kitchen", "bedroom", "office", "garden", "beach", "school", "library",
    "park", "station", "airport", "market", "museum", "clinic", "hotel",
]

WORDS = COLORS + ANIMALS + OBJECTS + FRUITS + PLACES + [
    "alpha", "bravo", "comet", "delta", "echo", "forest", "river", "mountain",
    "summer", "winter", "spring", "autumn", "north", "south", "east", "west",
    "quiet", "loud", "happy", "sad", "early", "late", "open", "close",
]

CATEGORY_WORDS = {
    "animal": ANIMALS,
    "object": OBJECTS,
    "fruit": FRUITS,
    "color": COLORS,
    "place": PLACES,
}

OPPOSITES = [
    ("hot", "cold"), ("up", "down"), ("left", "right"), ("yes", "no"),
    ("open", "closed"), ("start", "finish"), ("day", "night"),
    ("inside", "outside"), ("early", "late"), ("big", "small"),
    ("fast", "slow"), ("happy", "sad"), ("true", "false"),
]


def generated_examples(num_needed, seed=1234):
    rng = random.Random(seed)
    rows = []
    seen = set()
    seen_questions = {}

    def add(question, answer):
        if question in seen_questions:
            return
        key = (question, answer)
        if key in seen:
            return
        seen_questions[question] = answer
        seen.add(key)
        rows.append((question, answer))

    for a in range(0, 250):
        for b in range(0, 40):
            add(f"Compute {a} plus {b}. Return only the number.", str(a + b))
            add(f"What is {a} plus {b}?", str(a + b))
            add(f"Return the sum of {a} and {b}.", str(a + b))
            if a >= b:
                add(f"Compute {a} minus {b}. Return only the number.", str(a - b))
                add(f"What is {a} minus {b}?", str(a - b))
            add(f"Return the larger number: {a}, {b}.", str(max(a, b)))
            add(f"Return the smaller number: {a}, {b}.", str(min(a, b)))
            add(f"Answer yes or no: Is {a} greater than {b}?", "yes" if a > b else "no")
            add(f"Answer yes or no: Is {a} equal to {b}?", "yes" if a == b else "no")
            if len(rows) >= num_needed * 2:
                break
        if len(rows) >= num_needed * 2:
            break

    for n in range(0, 2000):
        add(f"Is {n} even? Answer yes or no.", "yes" if n % 2 == 0 else "no")
        add(f"Is {n} odd? Answer yes or no.", "yes" if n % 2 == 1 else "no")
        add(f"Return double {n}.", str(n * 2))
        if n % 2 == 0:
            add(f"Return half of {n}.", str(n // 2))

    for word in WORDS:
        add(f"Repeat exactly: {word}", word)
        add(f"Uppercase this word: {word}", word.upper())
        add(f"Lowercase this word: {word.upper()}", word.lower())
        add(f"Return the number of letters in {word}.", str(len(word)))

    category_words = {
        label: [word for word in words if sum(word in values for values in CATEGORY_WORDS.values()) == 1]
        for label, words in CATEGORY_WORDS.items()
    }
    for label, words in category_words.items():
        for word in words:
            add(f"Choose the category for {word}: animal, object, fruit, color, or place.", label)
            add(f"Classify this word: {word}.", label)

    for first in WORDS:
        for second in WORDS:
            if first == second:
                continue
            third = rng.choice(WORDS)
            if third in {first, second}:
                continue
            add(f"Return the first word: {first} {second} {third}", first)
            add(f"Return the last word: {first} {second} {third}", third)
            add(f"Return the middle word: {first} {second} {third}", second)
            if len(rows) >= num_needed * 3:
                break
        if len(rows) >= num_needed * 3:
            break

    for left, right in OPPOSITES:
        add(f"What is the opposite of {left}?", right)
        add(f"Complete the pair: {left} -> {right}, {right} ->", left)
        add(f"Answer yes or no: Is {left} the opposite of {right}?", "yes")
        add(f"Answer yes or no: Is {left} the same as {right}?", "no")

    rng.shuffle(rows)
    return rows[:num_needed]


def build_examples(num_examples):
    pairs = FACTS + CLASSIFY + TRANSFORMS + arithmetic_examples()
    if num_examples > len(pairs):
        pairs = pairs + generated_examples(num_examples - len(pairs))

    rows = []
    for idx, (question, answer) in enumerate(pairs[:num_examples]):
        rows.append({
            "index": idx,
            "input": f"User: {question}\nAssistant:",
            "target": answer,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--probe_jsonl", required=True)
    parser.add_argument("--train_jsonl", default=None)
    parser.add_argument("--num_examples", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_input_length", type=int, default=96)
    parser.add_argument("--append_eos", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = []
    for row in build_examples(args.num_examples):
        cond_ids = tokenizer(row["input"], add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(row["target"], add_special_tokens=False)["input_ids"]
        if args.append_eos and tokenizer.eos_token_id is not None:
            target_ids = list(target_ids) + [int(tokenizer.eos_token_id)]

        total_len = len(cond_ids) + len(target_ids)
        if len(cond_ids) > args.max_input_length:
            raise ValueError(f"prompt too long at index={row['index']}: {len(cond_ids)}")
        if total_len > args.max_length:
            raise ValueError(f"sequence too long at index={row['index']}: {total_len}")

        rows.append({
            **row,
            "condition_input_ids": cond_ids,
            "input_ids": target_ids,
            "prompt_len": len(cond_ids),
            "target_len": len(target_ids),
            "total_len": total_len,
        })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(str(out_dir))

    probe_path = Path(args.probe_jsonl)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    with probe_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"input": row["input"], "output": row["target"]}, ensure_ascii=False) + "\n")

    if args.train_jsonl:
        train_path = Path(args.train_jsonl)
        train_path.parent.mkdir(parents=True, exist_ok=True)
        with train_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "num_examples": len(rows),
        "max_length": args.max_length,
        "max_input_length": args.max_input_length,
        "append_eos": bool(args.append_eos),
        "prompt_tokens": {
            "min": min(r["prompt_len"] for r in rows),
            "max": max(r["prompt_len"] for r in rows),
            "mean": sum(r["prompt_len"] for r in rows) / len(rows),
        },
        "target_tokens": {
            "min": min(r["target_len"] for r in rows),
            "max": max(r["target_len"] for r in rows),
            "mean": sum(r["target_len"] for r in rows) / len(rows),
        },
        "total_tokens": {
            "min": min(r["total_len"] for r in rows),
            "max": max(r["total_len"] for r in rows),
            "mean": sum(r["total_len"] for r in rows) / len(rows),
        },
    }
    (out_dir / "short_qa_overfit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote dataset: {out_dir}")
    print(f"wrote probe jsonl: {probe_path}")


if __name__ == "__main__":
    main()
