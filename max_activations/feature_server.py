#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
"""
Feature activation server backed by the DuckDB activation database.

Serves top, bottom, and random activations per feature with decoded token
context. Requires a tokenizer (same model used during build) to decode text.

Usage:
    python max_activations/feature_server.py
    python max_activations/feature_server.py --db max_activations/Qwen_Qwen3-1.7B.duckdb --model Qwen/Qwen3-1.7B --port 8001 --context_window 20
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import sys
import argparse
import time
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb

# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Feature activation server (DuckDB)")
parser.add_argument("--port",        type=int,   default=8001)
parser.add_argument("--db",          type=str,   default=None,
                    help="Path to .duckdb file (default: auto-detect in max_activations/)")
parser.add_argument("--model",       type=str,   default=None,
                    help="HuggingFace model name for tokenizer (default: inferred from db filename)")
parser.add_argument("--top_n",       type=int,   default=20,
                    help="Number of top/bottom/random activations to return per feature")
parser.add_argument("--context_window", type=int, default=20)
parser.add_argument("--max_length",  type=int,   default=512)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Auto-detect DB path
# ---------------------------------------------------------------------------
def find_db_files():
    db_dir = os.path.dirname(os.path.abspath(__file__))
    return sorted(
        os.path.join(db_dir, f)
        for f in os.listdir(db_dir)
        if f.endswith(".duckdb")
    )

if args.db is None:
    found = find_db_files()
    if not found:
        print("ERROR: No .duckdb files found in max_activations/. Pass --db explicitly.")
        sys.exit(1)
    args.db = found[0]
    print(f"Auto-selected DB: {args.db}")

if not os.path.exists(args.db):
    print(f"ERROR: DB not found: {args.db}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Infer model name from db filename if not given
# ---------------------------------------------------------------------------
if args.model is None:
    stem = os.path.splitext(os.path.basename(args.db))[0]  # e.g. Qwen_Qwen3-1.7B
    args.model = stem.replace("_", "/", 1)                  # e.g. Qwen/Qwen3-1.7B
    print(f"Inferred model name: {args.model}")


# ---------------------------------------------------------------------------
# Load tokenizer (optional)
# ---------------------------------------------------------------------------
from transformers import AutoTokenizer
print(f"Loading tokenizer: {args.model} ...")
try:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
except Exception as e:
    print(f"ERROR: Could not load tokenizer for {args.model}: {e}")
    sys.exit(1)
print("Tokenizer loaded.")


# ---------------------------------------------------------------------------
# Token decode helper
# ---------------------------------------------------------------------------
def decode_context(text, token_pos, context_window, max_length):
    if tokenizer is None or not text:
        return None, None, None
    enc = tokenizer(text, add_special_tokens=False, truncation=True,
                    max_length=max_length, return_tensors=None)
    ids = enc["input_ids"]
    seq_len = len(ids)
    if token_pos >= seq_len:
        return "", "", ""
    token_str   = tokenizer.decode([ids[token_pos]], skip_special_tokens=False)
    start       = max(0, token_pos - context_window)
    end         = min(seq_len, token_pos + 1 + context_window)
    ctx_before  = tokenizer.decode(ids[start:token_pos], skip_special_tokens=False) if start < token_pos else ""
    ctx_after   = tokenizer.decode(ids[token_pos + 1:end], skip_special_tokens=False) if token_pos + 1 < end else ""
    return token_str, ctx_before, ctx_after


# ---------------------------------------------------------------------------
# DB query helpers
# ---------------------------------------------------------------------------
def get_db_info(con):
    """Return layer range and feature count."""
    row = con.execute(
        "SELECT MIN(layer), MAX(layer), COUNT(DISTINCT feature) FROM activations"
    ).fetchone()
    return {"min_layer": row[0], "max_layer": row[1], "num_features": row[2]}


def get_layer_summary(con, layer):
    """Per-feature summary for a whole layer (max activation + top token)."""
    rows = con.execute(
        """
        SELECT feature, MAX(activation) AS max_act, MIN(activation) AS min_act
        FROM activations
        WHERE layer = ?
        GROUP BY feature
        ORDER BY feature
        """,
        [layer],
    ).fetchall()
    features = []
    for feature, max_act, min_act in rows:
        features.append({
            "feature":        feature,
            "max_activation": round(float(max_act), 4),
            "min_activation": round(float(min_act), 4),
        })
    return features


def get_feature_rows(con, layer, feature, n, bucket):
    order = "DESC" if bucket == "top" else "ASC"
    rows = con.execute(
        f"""
        SELECT activation, token_pos, data_id
        FROM activations
        WHERE layer = ? AND feature = ? AND bucket = ?
        ORDER BY activation {order}
        LIMIT ?
        """,
        [layer, feature, bucket, n],
    ).fetchall()
    return rows


def fetch_texts(con, data_ids):
    if not data_ids:
        return {}
    placeholders = ", ".join(str(d) for d in data_ids)
    rows = con.execute(
        f"SELECT data_id, text FROM dataset_index WHERE data_id IN ({placeholders})"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def build_activation_list(con, layer, feature, n, bucket):
    rows = get_feature_rows(con, layer, feature, n, bucket)
    if not rows:
        return []

    data_ids   = list({r[2] for r in rows})
    id_to_text = fetch_texts(con, data_ids)

    result = []
    for activation, token_pos, data_id in rows:
        text = id_to_text.get(data_id, "")
        token, ctx_before, ctx_after = decode_context(
            text, token_pos, args.context_window, args.max_length)

        # Full document split at token_pos for the "show full text" toggle
        full_before = full_after = None
        if tokenizer is not None and text:
            enc = tokenizer(text, add_special_tokens=False, truncation=True,
                            max_length=args.max_length, return_tensors=None)
            ids = enc["input_ids"]
            if token_pos < len(ids):
                full_before = tokenizer.decode(ids[:token_pos],     skip_special_tokens=False)
                full_after  = tokenizer.decode(ids[token_pos + 1:], skip_special_tokens=False)

        result.append({
            "activation":     round(float(activation), 6),
            "token_pos":      int(token_pos),
            "data_id":        int(data_id),
            "bucket":         bucket,
            "token":          token,
            "context_before": ctx_before,
            "context_after":  ctx_after,
            "full_before":    full_before,
            "full_after":     full_after,
        })
    return result


def search_features(con, layer, phrase, bucket, limit):
    """
    Find features whose top-activating documents contain the phrase,
    ranked by number of matching rows. Returns a snippet around the
    first occurrence of the phrase for each feature.
    """
    rows = con.execute(
        """
        SELECT a.feature, COUNT(DISTINCT a.data_id) AS hits, MAX(a.activation) AS max_act,
               ARG_MAX(d.text, a.activation) AS sample_text
        FROM activations a
        JOIN dataset_index d ON a.data_id = d.data_id
        WHERE a.layer = ?
          AND a.bucket = ?
          AND d.text ILIKE ?
        GROUP BY a.feature
        ORDER BY hits DESC, max_act DESC
        LIMIT ?
        """,
        [layer, bucket, f"%{phrase}%", limit],
    ).fetchall()

    results = []
    for feature, hits, max_act, text in rows:
        snippet = None
        if text:
            idx = text.lower().find(phrase.lower())
            if idx >= 0:
                start = max(0, idx - 60)
                end   = min(len(text), idx + len(phrase) + 60)
                snippet = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
        results.append({
            "feature":        feature,
            "hits":           hits,
            "max_activation": round(float(max_act), 4),
            "snippet":        snippet,
            "phrase":         phrase,
        })
    return results


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html if isinstance(html, bytes) else html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        con    = self.server.con
        t0     = time.time()

        try:
            if parsed.path == "/feature":
                layer   = params.get("layer",   [None])[0]
                feature = params.get("feature", [None])[0]
                n       = int(params.get("n", [str(args.top_n)])[0])

                if layer is None:
                    self._send_json({"error": "Missing required parameter: layer"})
                    return

                layer = int(layer)

                if feature is None:
                    # Layer summary
                    features = get_layer_summary(con, layer)
                    elapsed  = (time.time() - t0) * 1000
                    print(f"layer summary  layer={layer}  {len(features)} features  {elapsed:.1f}ms")
                    self._send_json({"layer": layer, "features": features})
                else:
                    # Full top/bottom/random for one feature
                    feature = int(feature)
                    top    = build_activation_list(con, layer, feature, n, "top")
                    bottom = build_activation_list(con, layer, feature, n, "bottom")
                    random = build_activation_list(con, layer, feature, n, "random")
                    elapsed = (time.time() - t0) * 1000
                    print(f"feature detail layer={layer} feature={feature}  "
                          f"top={len(top)} bot={len(bottom)} rnd={len(random)}  {elapsed:.1f}ms")
                    self._send_json({
                        "layer":   layer,
                        "feature": feature,
                        "top":     top,
                        "bottom":  bottom,
                        "random":  random,
                    })

            elif parsed.path == "/search":
                layer  = params.get("layer",  [None])[0]
                phrase = params.get("phrase", [None])[0]
                bucket = params.get("bucket", ["top"])[0]
                limit  = int(params.get("limit", ["50"])[0])

                if layer is None or not phrase:
                    self._send_json({"error": "Missing required parameters: layer, phrase"})
                    return

                results = search_features(con, int(layer), phrase, bucket, limit)
                elapsed = (time.time() - t0) * 1000
                print(f"search  layer={layer} phrase={phrase!r} bucket={bucket}  {len(results)} hits  {elapsed:.1f}ms")
                self._send_json({"layer": int(layer), "phrase": phrase,
                                 "bucket": bucket, "results": results})

            elif parsed.path == "/info":
                info    = get_db_info(con)
                elapsed = (time.time() - t0) * 1000
                print(f"info  {elapsed:.1f}ms")
                self._send_json(info)

            elif parsed.path == "/" or parsed.path == "/viewer":
                viewer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "feature_viewer.html")
                if os.path.exists(viewer_path):
                    with open(viewer_path, "rb") as f:
                        self._send_html(f.read())
                else:
                    self._send_html(b"<p>feature_viewer.html not found next to server.</p>")

            else:
                self._send_json({"error": "Unknown endpoint"}, status=404)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, status=500)

    def log_message(self, fmt, *a):
        pass  # silence default access log; we print manually above


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class FeatureServer(HTTPServer):
    def __init__(self, addr, handler, db_path):
        super().__init__(addr, handler)
        self.con = duckdb.connect(db_path, read_only=True)
        print(f"Connected to: {db_path}")


def main():
    port = int(os.environ.get("WEBAPP_PORT", args.port))
    server = FeatureServer(("", port), Handler, args.db)
    print()
    print("=" * 60)
    print(f"  Feature Activation Server  —  http://localhost:{port}")
    print("=" * 60)
    print(f"  GET /info                         DB metadata")
    print(f"  GET /feature?layer=N              layer summary")
    print(f"  GET /feature?layer=N&feature=M    top/bottom/random")
    print(f"  GET /feature?layer=N&feature=M&n=50  (custom n)")
    print(f"  GET /                             open HTML viewer")
    print()
    print(f"  Tokenizer : loaded ({args.model})")
    print(f"  Default n : {args.top_n} per bucket")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.con.close()
        server.shutdown()


if __name__ == "__main__":
    main()
