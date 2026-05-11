# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import numpy as np


BOX_RE = re.compile(r"(?:\\boxed|boxed)\{([^}]*)\}")
HASH_RE = re.compile(r"####\s*(-?[0-9][0-9,\.]*)")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def _strip_commas(s: str) -> str:
    return s.replace(",", "").strip()

def extract_answer(text: str) -> str:
    text = str(text)

    m = BOX_RE.search(text)
    if m is not None:
        return _strip_commas(m.group(1))

    m = HASH_RE.search(text)
    if m is not None:
        return _strip_commas(m.group(1))

    nums = NUM_RE.findall(text)
    if nums:
        return _strip_commas(nums[-1])

    return "[invalid]"

def calculate_metrics(df):
    preds = df["predicted_answer"].astype(str).tolist()

    if "answer" not in df.columns:
        raise KeyError(f"gsm8k scorer expects df['answer'], got columns={list(df.columns)}")
    golds = df["answer"].astype(str).tolist()

    correct = [extract_answer(p) == extract_answer(g) for p, g in zip(preds, golds)]
    acc = float(np.mean(correct)) if len(correct) else 0.0

    # Track how often the model emitted a parseable final answer.
    answered = sum(a != "[invalid]" for a in map(extract_answer, preds))
    return {"accuracy": round(acc * 100, 2), "answered": answered, "total": len(df)}
