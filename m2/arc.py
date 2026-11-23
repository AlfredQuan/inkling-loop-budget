"""Build a deterministic ARC manifest and native-probe inputs."""

import argparse
import hashlib
import json
import re
from pathlib import Path


REVISION = "210d026faf9955653af8916fad021475a3f00453"
FILES = {
    "ARC-Challenge/train": "e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb",
    "ARC-Challenge/validation": "395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05",
    "ARC-Challenge/test": "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9",
    "ARC-Easy/train": "b315db8a4be597dc7daa50a4e70d48dd7c990c32085629e6ccd8c926beaa80b5",
    "ARC-Easy/validation": "ed890ff1e4cef7a7140d3a30dcea3ed2c9d467c6458f447ad9ef0176d8dcbb74",
    "ARC-Easy/test": "4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177",
}
CAP = {"train": 256, "validation": 96, "test": 192}
LETTERS = "ABCD"
PREFIX = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>"
SUFFIX = "<|end_message|><|message_assistant|><|final|><|content_text|>"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def render(question, options):
    choices = "\n".join(f"{letter}. {text}" for letter, text in zip(LETTERS, options))
    return PREFIX + "Choose the correct answer. Reply with one letter (A, B, C, or D).\n\n" + question + "\n" + choices + SUFFIX


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def build(source, out):
    import pyarrow.parquet as pq

    all_rows, multiplicity = {}, {}
    for key, expected in FILES.items():
        file = source / f"{key}-00000-of-00001.parquet"
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset checksum mismatch: {file}")
        valid = []
        set_name, split = key.split("/")
        for row in pq.read_table(file).to_pylist():
            q, options = row["question"], row["choices"]["text"]
            source_labels, gold = row["choices"]["label"], row["answerKey"]
            norm = normalized(q)
            if (
                len(options) != 4 or len(set(source_labels)) != 4 or gold not in source_labels
                or len(set(map(normalized, options))) != 4 or not norm
                or any("<|" in text for text in [q, *options])
            ):
                continue
            multiplicity[norm] = multiplicity.get(norm, 0) + 1
            valid.append((row["id"], norm, q.strip(), options, source_labels.index(gold)))
        all_rows[set_name, split] = valid

    out.mkdir(parents=True, exist_ok=True)
    records, native = [], []
    for i, letter in enumerate(LETTERS):
        values = ["300", "100", "200", "400"]
        values[i] = "2"
        formatted = render("What is 1+1?", values)
        name = f"canary-{letter}"
        records.append({"id": name, "split": "canary", "label": letter, "subset": "canary", "prompt_sha256": digest(formatted)})
        native.append((name, formatted))
    for (subset, split), rows in all_rows.items():
        candidates = sorted(
            (item for item in rows if multiplicity[item[1]] == 1),
            key=lambda item: digest(f"loops-arc-v1:{subset}:{split}:{item[0]}"),
        )
        if len(candidates) < CAP[split]:
            raise ValueError(f"Not enough unique rows in {subset}/{split}")
        for arc_id, _, question, options, answer in candidates[: CAP[split]]:
            order = sorted(range(4), key=lambda index: digest(f"options-v1:{subset}:{arc_id}:{index}"))
            formatted = render(question, [options[i] for i in order])
            name = digest(f"arc:{subset}:{arc_id}")[:20]
            records.append({
                "id": name, "arc_id": arc_id, "split": split, "subset": subset,
                "label": LETTERS[order.index(answer)], "prompt_sha256": digest(formatted),
            })
            native.append((name, formatted))
    manifest = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    (out / "manifest.jsonl").write_bytes(manifest.encode())
    (out / "prompts.tsv").write_bytes("".join(f"{i}\t{s.encode().hex()}\n" for i, s in native).encode())
    print(f"prepared {len(records) - 4} benchmark questions, plus four format checks")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    build(a.source, a.out)
