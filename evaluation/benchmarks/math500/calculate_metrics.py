# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import re


def extract_boxed(pred_answer):
    text = str(pred_answer)
    marker = "boxed{"
    start = text.find(marker)
    if start < 0:
        return None

    i = start + len(marker)
    depth = 1
    chars = []
    while i < len(text):
        char = text[i]
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars)
            chars.append(char)
        else:
            chars.append(char)
        i += 1

    return None


def normalize_answer(answer):
    if answer is None:
        return None
    answer = str(answer)
    answer = answer.replace(r"\left", "").replace(r"\right", "")
    return re.sub(r"\s+", "", answer)


def score_aime(pred_answer, true_answer):
    return normalize_answer(extract_boxed(pred_answer)) == normalize_answer(true_answer)


def calculate_metrics(df: pd.DataFrame) -> dict:
    correct = 0
    answered = 0
    for index, row in df.iterrows():
        correct += score_aime(row["predicted_answer"], row["answer"])
        answered += "boxed{" in row["predicted_answer"]
    return {"correct": correct, "answered": answered, "accuracy": correct / len(df), "total": len(df)}
