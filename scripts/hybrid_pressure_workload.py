"""Build a synthetic cache-pressure request sequence for mechanism checks.

The cold-document scan is deliberately synthetic and is not evidence about
real-world traffic or optimization benefit.
"""

from __future__ import annotations

import hashlib
from typing import Any

from hybrid_prefix_probe import make_document


def make_pressure_requests(cold_count: int = 3) -> list[dict[str, Any]]:
    """Return hot, cold-scan, and hot-return prompts in fixed request order.

    Every document begins with a distinct document ID. The three warm
    requests, each cold scan request, and the two return requests all use
    different questions, so every complete prompt is unique. This synthetic
    scan checks mechanism behavior only; it is not a realistic workload.
    """
    if cold_count < 0:
        raise ValueError("cold_count must be non-negative")

    hot_doc_id = "DOC-HOT-001"
    hot_document = f"Document ID: {hot_doc_id}\n\n{make_document('HOT')}"
    specs: list[tuple[str, str, str, str]] = [
        ("warm1", "warm1", hot_doc_id,
         "Which types of records appear in this fictional collection?"),
        ("warm2", "warm2", hot_doc_id,
         "How should reviewers handle uncertain details in the catalog?"),
        ("warm3", "warm3", hot_doc_id,
         "What should be preserved when two sources disagree?"),
    ]

    cold_documents: dict[str, str] = {}
    for index in range(cold_count):
        request_id = f"scan{index}"
        doc_id = f"DOC-COLD-{index:03d}"
        cold_documents[request_id] = (
            f"Document ID: {doc_id}\n\n{make_document(f'COLD-{index:03d}')}"
        )
        specs.append((
            request_id,
            request_id,
            doc_id,
            f"What preservation detail is described in cold scan document {index}?",
        ))

    specs.extend([
        ("hot_return", "hot_return", hot_doc_id,
         "Why does the fictional archive retain the original source wording?"),
        ("hot_verify", "hot_verify", hot_doc_id,
         "Which step lets another reader repeat the same search?"),
    ])

    rows: list[dict[str, Any]] = []
    for name, request_id, doc_id, question in specs:
        document = hot_document if doc_id == hot_doc_id else cold_documents[request_id]
        prompt = (
            "Read the following fictional archive description and answer the question briefly.\n\n"
            f"{document}\n\nQuestion: {question}\nAnswer:"
        )
        rows.append({
            "name": name,
            "request_id": request_id,
            "doc_id": doc_id,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        })
    return rows
