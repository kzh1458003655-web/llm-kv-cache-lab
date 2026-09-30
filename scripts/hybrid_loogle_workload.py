"""Build an in-memory synthetic cache-pressure sequence from public LooGLE rows.

The prompts combine public document text with a hand-crafted hot/scan arrival
sequence. Contexts are truncated, so this is not an original LooGLE evaluation,
not a production trace, and must not be used to judge question-answer quality.
The function does not write source text or prompts to disk.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


def make_loogle_requests(
    source_file: str | Path,
    tokenizer: Any,
    context_tokens: int = 1400,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read LooGLE ``shortdep_qa.jsonl`` and return eight requests in memory.

    The first document in source order with at least five distinct questions
    and one consistent context is selected as hot. Its first five distinct
    questions supply warm1–3, hot_return, and hot_verify. The next three
    context-consistent documents whose first question is distinct from all
    selected hot questions supply scan0–2.

    Returns ``(requests, metadata)``. Metadata contains source SHA-256,
    complete-row/document counts, partial-line count, and public dataset
    provenance, but no local paths or source/prompt text. Only a malformed,
    non-newline-terminated final record is accepted as a partial line.
    """
    if context_tokens < 1:
        raise ValueError("context_tokens must be positive")
    source = Path(source_file)
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    digest = hashlib.sha256()
    complete_row_count = 0
    partial_line_count = 0

    try:
        with source.open("rb") as stream:
            while True:
                raw = stream.readline()
                if not raw:
                    break
                digest.update(raw)
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    if not raw.endswith(b"\n"):
                        partial_line_count += 1
                        break
                    raise ValueError(
                        f"invalid complete JSONL row at line {complete_row_count + 1}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError("each complete JSONL row must be an object")
                missing = [field for field in ("doc_id", "context", "question", "id") if field not in row]
                if missing:
                    raise ValueError(f"source row is missing required fields: {', '.join(missing)}")
                if not all(isinstance(row[field], str) for field in ("context", "question")):
                    raise ValueError("source context and question fields must be strings")
                groups.setdefault(str(row["doc_id"]), []).append(row)
                complete_row_count += 1
    except OSError as exc:
        raise ValueError(f"cannot read dataset source {source}: {exc}") from exc

    if not groups:
        raise ValueError("source contains no complete dataset records")

    hot_doc_id: str | None = None
    hot_rows: list[dict[str, Any]] | None = None
    hot_questions: list[str] | None = None
    hot_context: str | None = None
    hot_position = -1
    for position, (doc_id, rows) in enumerate(groups.items()):
        contexts = {row["context"] for row in rows}
        if len(contexts) != 1:
            continue
        unique_rows: list[dict[str, Any]] = []
        seen_questions: set[str] = set()
        for row in rows:
            question = row["question"]
            if question not in seen_questions:
                seen_questions.add(question)
                unique_rows.append(row)
        if len(unique_rows) >= 5:
            hot_doc_id = doc_id
            hot_rows = unique_rows[:5]
            hot_questions = [row["question"] for row in hot_rows]
            hot_context = next(iter(contexts))
            hot_position = position
            break

    if hot_doc_id is None or hot_rows is None or hot_questions is None or hot_context is None:
        raise ValueError("no document has five distinct questions and a consistent context")

    selected_cold: list[tuple[str, dict[str, Any], str]] = []
    used_questions = set(hot_questions)
    for position, (doc_id, rows) in enumerate(groups.items()):
        if position <= hot_position:
            continue
        contexts = {row["context"] for row in rows}
        if len(contexts) != 1:
            continue
        first_row = rows[0]
        question = first_row["question"]
        if question in used_questions:
            continue
        selected_cold.append((doc_id, first_row, next(iter(contexts))))
        used_questions.add(question)
        if len(selected_cold) == 3:
            break
    if len(selected_cold) != 3:
        raise ValueError("fewer than three following documents have a consistent context and distinct first question")

    def decode_prefix(context: str) -> str:
        token_ids = tokenizer.encode(context, add_special_tokens=False)
        return tokenizer.decode(token_ids[:context_tokens])

    hot_prefix = decode_prefix(hot_context)
    cold_prefixes = [(doc_id, row, decode_prefix(context)) for doc_id, row, context in selected_cold]

    request_specs: list[tuple[str, str, dict[str, Any], str]] = [
        ("warm1", hot_doc_id, hot_rows[0], hot_prefix),
        ("warm2", hot_doc_id, hot_rows[1], hot_prefix),
        ("warm3", hot_doc_id, hot_rows[2], hot_prefix),
    ]
    request_specs.extend(
        (f"scan{index}", doc_id, row, prefix)
        for index, (doc_id, row, prefix) in enumerate(cold_prefixes)
    )
    request_specs.extend([
        ("hot_return", hot_doc_id, hot_rows[3], hot_prefix),
        ("hot_verify", hot_doc_id, hot_rows[4], hot_prefix),
    ])

    requests: list[dict[str, Any]] = []
    prompt_hashes: set[str] = set()
    question_hashes: set[str] = set()
    for request_id, doc_id, row, prefix in request_specs:
        question = row["question"]
        document = f"Document ID: {doc_id}\n\n{prefix}"
        prompt = (
            "Read the document and answer the question.\n"
            f"{document}\nQuestion: {question}\nAnswer:"
        )
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        question_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
        if prompt_sha256 in prompt_hashes:
            raise ValueError("selected records produced a duplicate prompt")
        if question_sha256 in question_hashes:
            raise ValueError("selected records do not have distinct questions")
        prompt_hashes.add(prompt_sha256)
        question_hashes.add(question_sha256)
        requests.append({
            "request_id": request_id,
            "doc_id": doc_id,
            "dataset_record_id": row["id"],
            "prompt": prompt,
            "prompt_sha256": prompt_sha256,
            "context_prefix_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
            "question_sha256": question_sha256,
        })

    metadata = {
        "source_sha256": digest.hexdigest(),
        "complete_row_count": complete_row_count,
        "partial_line_count": partial_line_count,
        "doc_count": len(groups),
        "source_dataset_url": "https://huggingface.co/datasets/bigainlco/LooGLE",
        "source_revision": "4b50b4cb9333b48b3f7ddfd307c1d13369be5d1e",
        "context_tokens": context_tokens,
    }
    return requests, metadata
