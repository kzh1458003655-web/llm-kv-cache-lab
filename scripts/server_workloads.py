"""Build deterministic in-memory LooGLE workloads for serving experiments.

These workloads use real public document text with a hand-crafted arrival
sequence. They are not production traces or QA evaluations; context truncation
means they are not the original LooGLE evaluation. This module never writes
source text or prompts to disk.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any


SOURCE_DATASET_URL = "https://huggingface.co/datasets/bigainlco/LooGLE"
SOURCE_REVISION = "4b50b4cb9333b48b3f7ddfd307c1d13369be5d1e"
_HOT_SCAN_SCENE = "hot_scan"
_ZIPF_SCENES = {"mixed_zipf", "roomy_control"}


def build_workload(
    source_file: str | Path,
    tokenizer: Any,
    scene: str,
    seed: int = 20260930,
    count: int = 128,
    context_tokens: int = 1400,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return in-memory request rows and path-free source/run metadata.

    Input must be valid JSONL except for an optional incomplete final line.
    Rows are selected without replacement. ``hot_scan`` emits fixed eight-row
    groups (three hot questions, three one-off documents, then two hot
    questions). ``mixed_zipf`` and ``roomy_control`` use the same seeded
    weighted document selection, with document weights based on first-seen
    rank: ``1 / (rank + 1) ** 1.2``.
    """
    if scene not in {_HOT_SCAN_SCENE, *_ZIPF_SCENES}:
        raise ValueError(f"unsupported scene {scene!r}; choose hot_scan, mixed_zipf, or roomy_control")
    if count <= 0:
        raise ValueError("count must be positive")
    if scene == _HOT_SCAN_SCENE and count % 8 != 0:
        raise ValueError("hot_scan count must be a multiple of 8")
    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")

    source = Path(source_file)
    source_hash = hashlib.sha256()
    complete_row_count = 0
    partial_line_count = 0
    seen_source_record_ids: set[str] = set()
    # Keep one context per document rather than retaining repeated copies from
    # every question row. Questions and source IDs remain associated with it.
    docs: OrderedDict[str, dict[str, Any]] = OrderedDict()

    try:
        with source.open("rb") as stream:
            line_number = 0
            while True:
                raw = stream.readline()
                if not raw:
                    break
                line_number += 1
                source_hash.update(raw)
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    if not raw.endswith(b"\n"):
                        partial_line_count += 1
                        break
                    raise ValueError(f"invalid complete JSONL record at line {line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"record at line {line_number} must be a JSON object")
                missing = [key for key in ("id", "doc_id", "context", "question") if key not in record]
                if missing:
                    raise ValueError(
                        f"record at line {line_number} is missing fields: {', '.join(missing)}"
                    )
                if not isinstance(record["context"], str) or not isinstance(record["question"], str):
                    raise ValueError(f"record at line {line_number} has non-string context/question")
                record_id_key = _record_id_key(record["id"])
                complete_row_count += 1
                if record_id_key in seen_source_record_ids:
                    continue
                seen_source_record_ids.add(record_id_key)
                doc_id = str(record["doc_id"])
                doc = docs.get(doc_id)
                if doc is None:
                    doc = {
                        "context": record["context"],
                        "context_consistent": True,
                        "questions": [],
                        "seen_questions": set(),
                    }
                    docs[doc_id] = doc
                elif doc["context"] != record["context"]:
                    doc["context_consistent"] = False
                question = record["question"]
                if question not in doc["seen_questions"]:
                    doc["seen_questions"].add(question)
                    doc["questions"].append({
                        "question": question,
                        "record_id": record["id"],
                    })
    except OSError as exc:
        raise ValueError(f"cannot read JSONL source: {type(exc).__name__}: {exc}") from exc

    if not docs:
        raise ValueError("source contains no complete JSONL records")

    # Only documents with a single consistent context can be used as one
    # shared-prefix family.
    eligible_docs = [
        doc_id for doc_id, doc in docs.items()
        if doc["context_consistent"] and doc["questions"]
    ]
    if not eligible_docs:
        raise ValueError("source has no document with a consistent context")

    # Cache tokenized prefixes once per doc. No original context is included
    # in metadata or written to disk; prompt strings live only in returned rows.
    document_prefixes: dict[str, str] = {}
    document_prefix_hashes: dict[str, str] = {}

    rng = random.Random(seed)
    doc_first_seen_rank = {doc_id: rank for rank, doc_id in enumerate(docs)}
    remaining: dict[str, list[dict[str, Any]]] = {
        doc_id: list(docs[doc_id]["questions"])
        for doc_id in eligible_docs
    }
    used_questions: set[str] = set()
    used_record_ids: set[str] = set()
    seen_prompts: set[str] = set()
    selected: list[tuple[str, dict[str, Any]]] = []

    def available_questions(doc_id: str) -> list[dict[str, Any]]:
        return [
            qa for qa in remaining[doc_id]
            if qa["question"] not in used_questions
            and _record_id_key(qa["record_id"]) not in used_record_ids
        ]

    def take_question(doc_id: str, qa: dict[str, Any]) -> None:
        # Remove the selected QA row by object identity; no question is reused.
        remaining[doc_id].remove(qa)
        used_questions.add(qa["question"])
        used_record_ids.add(_record_id_key(qa["record_id"]))
        selected.append((doc_id, qa))

    if scene == _HOT_SCAN_SCENE:
        if count % 8 != 0:  # Keep this invariant close to the generation loop.
            raise ValueError("hot_scan count must be a multiple of 8")
        for _segment_index in range(count // 8):
            hot_candidates = [
                doc_id for doc_id in eligible_docs
                if len(available_questions(doc_id)) >= 5
            ]
            if not hot_candidates:
                raise ValueError("not enough unused QA rows to form another hot_scan segment")
            hot_doc = rng.choice(hot_candidates)
            hot_qas = rng.sample(available_questions(hot_doc), 5)
            hot_question_set = {qa["question"] for qa in hot_qas}
            # Hold the five unique hot questions aside while selecting the
            # three one-off docs, then emit them in the requested order.
            for qa in hot_qas[:3]:
                take_question(hot_doc, qa)
            cold_selection: list[tuple[str, dict[str, Any]]] = []
            cold_question_set: set[str] = set()
            while len(cold_selection) < 3:
                selected_cold_docs = {doc_id for doc_id, _ in cold_selection}
                cold_candidates = {
                    doc_id: [
                        qa for qa in available_questions(doc_id)
                        if qa["question"] not in hot_question_set
                        and qa["question"] not in cold_question_set
                    ]
                    for doc_id in eligible_docs
                    if doc_id != hot_doc and doc_id not in selected_cold_docs
                }
                cold_candidates = {
                    doc_id: options for doc_id, options in cold_candidates.items() if options
                }
                if not cold_candidates:
                    break
                cold_doc = rng.choice(list(cold_candidates))
                qa = rng.choice(cold_candidates[cold_doc])
                cold_selection.append((cold_doc, qa))
                cold_question_set.add(qa["question"])
            if len(cold_selection) != 3:
                raise ValueError("not enough distinct cold documents with unused QA rows")
            for doc_id, qa in cold_selection:
                take_question(doc_id, qa)
            for qa in hot_qas[3:]:
                # May have been selected among the first three only in an
                # invalidated trace, which is prevented by the fixed slicing.
                take_question(hot_doc, qa)
    else:
        for _ in range(count):
            candidates = [
                doc_id for doc_id in eligible_docs
                if available_questions(doc_id)
            ]
            if not candidates:
                raise ValueError("not enough distinct unused QA rows for requested workload count")
            weights = [
                1.0 / ((doc_first_seen_rank[doc_id] + 1) ** 1.2)
                for doc_id in candidates
            ]
            doc_id = rng.choices(candidates, weights=weights, k=1)[0]
            qa = rng.choice(available_questions(doc_id))
            take_question(doc_id, qa)

    rows: list[dict[str, Any]] = []
    for index, (doc_id, qa) in enumerate(selected):
        if doc_id not in document_prefixes:
            token_ids = tokenizer.encode(docs[doc_id]['context'], add_special_tokens=False)
            decoded_context = tokenizer.decode(token_ids[:context_tokens])
            document = f"Document ID: {doc_id}\n\n{decoded_context}"
            document_prefixes[doc_id] = document
            document_prefix_hashes[doc_id] = hashlib.sha256(document.encode('utf-8')).hexdigest()
        if scene == _HOT_SCAN_SCENE:
            segment_index, slot = divmod(index, 8)
            if slot < 3:
                phase_id = f"warm{slot + 1}"
            elif slot < 6:
                phase_id = f"scan{slot - 3}"
            else:
                phase_id = "hot_return" if slot == 6 else "hot_verify"
            scene_request_id = f"segment{segment_index:03d}-{phase_id}"
        else:
            scene_request_id = f"request{index:04d}"
        prompt = (
            f"{document_prefixes[doc_id]}\n"
            f"Question: {qa['question']}\nAnswer:"
        )
        if prompt in seen_prompts:
            raise ValueError("selected records produced a duplicate complete prompt")
        seen_prompts.add(prompt)
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        rows.append({
            "request_id": scene_request_id,
            "doc_id": doc_id,
            "dataset_record_id": qa["record_id"],
            "prompt": prompt,
            "prompt_sha256": prompt_sha256,
            "question_sha256": hashlib.sha256(qa["question"].encode("utf-8")).hexdigest(),
            "context_prefix_sha256": document_prefix_hashes[doc_id],
            "expected_prompt_tokens": len(prompt_tokens),
        })

    if len(rows) != count:
        raise ValueError(f"internal workload count mismatch: expected {count}, got {len(rows)}")
    metadata = {
        "source_sha256": source_hash.hexdigest(),
        "complete_row_count": complete_row_count,
        "partial_line_count": partial_line_count,
        "doc_count": len(docs),
        "source_dataset_url": SOURCE_DATASET_URL,
        "source_revision": SOURCE_REVISION,
        "scene": scene,
        "seed": seed,
        "count": count,
        "context_tokens": context_tokens,
    }
    return rows, metadata


def _record_id_key(record_id: Any) -> str:
    """Create a stable comparable key for scalar or structured JSON IDs."""
    return json.dumps(record_id, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
