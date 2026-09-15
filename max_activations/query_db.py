#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Query helpers for the DuckDB activation database.

All connections are opened read-only. Text decoding re-encodes stored text
with the same parameters as build_activation_db.py so that token_pos values
map correctly.
"""

from typing import List, Dict, Optional
import duckdb


def _decode_context(text: str, token_pos: int, tokenizer,
                    context_window: int = 20, max_length: int = 512):
    """
    Re-encode stored text and decode the token at token_pos plus surrounding
    context. Uses identical tokenization parameters as build_activation_db.py.

    Returns:
        (token_str, context_before, context_after)
    """
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    ids = enc["input_ids"]
    seq_len = len(ids)
    if token_pos >= seq_len:
        return "", "", ""

    token_str = tokenizer.decode([ids[token_pos]], skip_special_tokens=False)
    start = max(0, token_pos - context_window)
    end   = min(seq_len, token_pos + 1 + context_window)

    context_before = tokenizer.decode(ids[start:token_pos], skip_special_tokens=False) if start < token_pos else ""
    context_after  = tokenizer.decode(ids[token_pos + 1:end], skip_special_tokens=False) if token_pos + 1 < end else ""

    return token_str, context_before, context_after


def _enrich_rows(rows: List[tuple], db_path: str, tokenizer,
                 context_window: int, max_length: int) -> List[Dict]:
    """
    Fetch text for each row from the DB and decode token context.

    rows columns: activation, token_pos, data_id, bucket
    """
    if not rows:
        return []

    # Batch-fetch all needed texts
    data_ids = list({r[2] for r in rows})
    con_ro = duckdb.connect(db_path, read_only=True)
    placeholders = ", ".join(str(d) for d in data_ids)
    text_rows = con_ro.execute(
        f"SELECT data_id, text FROM dataset_index WHERE data_id IN ({placeholders})"
    ).fetchall()
    con_ro.close()

    id_to_text = {r[0]: r[1] for r in text_rows}

    result = []
    for activation, token_pos, data_id, bucket in rows:
        text = id_to_text.get(data_id, "")
        if tokenizer is not None and text:
            token_str, ctx_before, ctx_after = _decode_context(
                text, token_pos, tokenizer, context_window, max_length)
        else:
            token_str = ctx_before = ctx_after = None

        result.append({
            "activation":    float(activation),
            "token_pos":     int(token_pos),
            "data_id":       int(data_id),
            "bucket":        bucket,
            "text":          text,
            "token":         token_str,
            "context_before": ctx_before,
            "context_after": ctx_after,
        })
    return result


def get_top_activations(db_path: str, layer: int, feature: int,
                         n: int = 50, tokenizer=None,
                         context_window: int = 20,
                         max_length: int = 512) -> List[Dict]:
    """
    Return up to n highest activations for (layer, feature), ordered by
    activation descending.  Falls back to all buckets if bucket='top' has
    fewer than n rows.
    """
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute(
        """
        SELECT activation, token_pos, data_id, bucket
        FROM activations
        WHERE layer = ? AND feature = ? AND bucket = 'top'
        ORDER BY activation DESC
        LIMIT ?
        """,
        [layer, feature, n],
    ).fetchall()

    if len(rows) < n:
        rows = con.execute(
            """
            SELECT activation, token_pos, data_id, bucket
            FROM activations
            WHERE layer = ? AND feature = ?
            ORDER BY activation DESC
            LIMIT ?
            """,
            [layer, feature, n],
        ).fetchall()
    con.close()

    return _enrich_rows(rows, db_path, tokenizer, context_window, max_length)


def get_bottom_activations(db_path: str, layer: int, feature: int,
                            n: int = 50, tokenizer=None,
                            context_window: int = 20,
                            max_length: int = 512) -> List[Dict]:
    """
    Return up to n lowest activations for (layer, feature), ordered by
    activation ascending.  Falls back to all buckets if needed.
    """
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute(
        """
        SELECT activation, token_pos, data_id, bucket
        FROM activations
        WHERE layer = ? AND feature = ? AND bucket = 'bottom'
        ORDER BY activation ASC
        LIMIT ?
        """,
        [layer, feature, n],
    ).fetchall()

    if len(rows) < n:
        rows = con.execute(
            """
            SELECT activation, token_pos, data_id, bucket
            FROM activations
            WHERE layer = ? AND feature = ?
            ORDER BY activation ASC
            LIMIT ?
            """,
            [layer, feature, n],
        ).fetchall()
    con.close()

    return _enrich_rows(rows, db_path, tokenizer, context_window, max_length)


def get_spectrum(db_path: str, layer: int, feature: int,
                 tokenizer=None,
                 context_window: int = 20,
                 max_length: int = 512) -> List[Dict]:
    """
    Return all stored rows for (layer, feature) covering all buckets
    (top, bottom, random), ordered by activation ascending.
    """
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute(
        """
        SELECT activation, token_pos, data_id, bucket
        FROM activations
        WHERE layer = ? AND feature = ?
        ORDER BY activation ASC
        """,
        [layer, feature],
    ).fetchall()
    con.close()

    return _enrich_rows(rows, db_path, tokenizer, context_window, max_length)


def get_feature_stats(db_path: str, layer: int, feature: int) -> Dict:
    """
    Return aggregate statistics for (layer, feature) broken down by bucket.

    Returns dict with keys: n_top, n_bottom, n_random, max_activation,
    min_activation, mean_top, mean_bottom.
    """
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute(
        """
        SELECT
            bucket,
            COUNT(*)           AS n,
            MAX(activation)    AS max_act,
            MIN(activation)    AS min_act,
            AVG(activation)    AS mean_act
        FROM activations
        WHERE layer = ? AND feature = ?
        GROUP BY bucket
        """,
        [layer, feature],
    ).fetchall()

    global_row = con.execute(
        """
        SELECT MAX(activation), MIN(activation)
        FROM activations
        WHERE layer = ? AND feature = ?
        """,
        [layer, feature],
    ).fetchone()
    con.close()

    stats = {
        "n_top": 0, "n_bottom": 0, "n_random": 0,
        "max_activation": None, "min_activation": None,
        "mean_top": None, "mean_bottom": None,
    }

    for bucket, n, max_act, min_act, mean_act in rows:
        if bucket == "top":
            stats["n_top"]    = n
            stats["mean_top"] = float(mean_act) if mean_act is not None else None
        elif bucket == "bottom":
            stats["n_bottom"]    = n
            stats["mean_bottom"] = float(mean_act) if mean_act is not None else None
        elif bucket == "random":
            stats["n_random"] = n

    if global_row:
        stats["max_activation"] = float(global_row[0]) if global_row[0] is not None else None
        stats["min_activation"] = float(global_row[1]) if global_row[1] is not None else None

    return stats
