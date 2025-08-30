"""
review_preprocess.py
--------------------
Single-file text preprocessing tool for review datasets.

Features:
- Lowercasing
- URL removal
- @/# symbol stripping (keeps words)
- Non-alphanumeric cleanup
- Tokenization
- Stopword removal (keeps negations by default)
- Optional stemming (NLTK PorterStemmer if available)
- Reads JSONL (one JSON per line) or JSON array files
- CLI to output CSV or Parquet

Usage examples:
    python review_preprocess.py --in reviews.jsonl --out cleaned.csv --text-col text
    python review_preprocess.py --in reviews.json --out cleaned.parquet --stem --remove-numbers
"""

from __future__ import annotations
import argparse
import json
import os
import re
from typing import Iterable, List, Optional
import pandas as pd
import argparse, json, os, re
from typing import Iterable, List, Optional
import pandas as pd
from json import JSONDecodeError




_BASE_STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at",
    "be","because","been","before","being","below","between","both","but","by","can","can't","cannot",
    "could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few",
    "for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll",
    "he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll",
    "i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most",
    "mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our",
    "ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should",
    "shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves",
    "then","there","there's","these","they","they'd","they'll","they're","they've","this","those",
    "through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're",
    "we've","were","weren't","what","what's","when","when's","where","where's","which","while","who",
    "who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're",
    "you've","your","yours","yourself","yourselves"
}
_NEGATIONS = {"no","not","nor","never"}

# regex patterns
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_MENTION_HASHTAG_RE = re.compile(r"[@#]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")

# optional stemmer
try:
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()
    _STEM_OK = True
except:
    _STEMMER = None
    _STEM_OK = False

def _tokenize(s: str) -> List[str]:
    return s.split() if s else []

def preprocess_text(
    text: str,
    *,
    keep_negations: bool = True,
    do_stem: bool = False,
    extra_stopwords: Optional[Iterable[str]] = None,
    remove_numbers: bool = False,
) -> str:
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = _URL_RE.sub(" ", s)
    s = _MENTION_HASHTAG_RE.sub(" ", s)
    s = s.replace("’", "'")
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()

    toks = _tokenize(s)

    stopwords = set(_BASE_STOPWORDS)
    if keep_negations:
        stopwords -= _NEGATIONS
    if extra_stopwords:
        stopwords |= {str(w).lower() for w in extra_stopwords}

    toks = [t for t in toks if t and t not in stopwords]
    if remove_numbers:
        toks = [t for t in toks if not t.isdigit()]
    if do_stem and _STEM_OK:
        toks = [_STEMMER.stem(t) for t in toks]

    return " ".join(toks)

def preprocess_dataframe(df: pd.DataFrame, *, text_col="text", out_col="text_clean", **kwargs):
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found")
    df2 = df.copy()
    df2[out_col] = df2[text_col].apply(lambda s: preprocess_text(s, **kwargs))
    return df2

def _read_jsonl(path: str) -> pd.DataFrame:
    import json, os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    if not os.path.isfile(path):
        raise IsADirectoryError(f"Input path is not a file: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}") from e

    if not rows:
        raise ValueError("JSONL file is empty or invalid.")
    return pd.DataFrame(rows)


# def read_any_json(path: str) -> pd.DataFrame:
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Input file not found: {path}")
#     if not os.path.isfile(path):
#         raise IsADirectoryError(f"Input path is not a file: {path}")

#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             obj = json.load(f)
#     except JSONDecodeError as e:
#         if "Extra data" in str(e):
#             raise ValueError(
#                 "This file appears to contain multiple JSON objects (likely JSONL), "
#                 "but your pipeline is configured for JSON arrays only.\n"
#                 "Fix by converting to a JSON array (e.g., wrap in [ ... ]) "
#                 "or switch to a JSONL-capable reader."
#             ) from e
#         raise

#     if isinstance(obj, list):
#         return pd.DataFrame(obj)

#     if isinstance(obj, dict):
#         for key in ("data", "items", "reviews"):
#             val = obj.get(key, None)
#             if isinstance(val, list):
#                 return pd.DataFrame(val)
#         return pd.DataFrame([obj])

#     raise ValueError(
#         "Unsupported JSON structure: expected a list of objects, or a dict "
#         "containing a list under 'data', 'items', or 'reviews'."
#     )




def write_table(df: pd.DataFrame, out_path: str):
    ext = os.path.splitext(out_path)[1].lower()
    if ext==".csv":
        df.to_csv(out_path,index=False)
    elif ext==".parquet":
        df.to_parquet(out_path,index=False)
    else:
        df.to_csv(out_path,index=False)
        print(f"Unknown extension {ext}, wrote CSV")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
    "--jsonl",
    action="store_true",
    help="Treat input as JSONL (one JSON object per line)."
)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--out-col", default="text_clean")
    ap.add_argument("--stem", action="store_true")
    ap.add_argument("--keep-negations", action="store_true")
    ap.add_argument("--no-keep-negations", dest="keep_negations", action="store_false")
    ap.add_argument("--remove-numbers", action="store_true")
    args = ap.parse_args()

    df = _read_jsonl(args.inp)
    df2 = preprocess_dataframe(df, text_col=args.text_col, out_col=args.out_col,
                               do_stem=args.stem, keep_negations=args.keep_negations,
                               remove_numbers=args.remove_numbers)
    write_table(df2, args.out)
    print(f"Wrote cleaned data to: {args.out}")

if __name__=="__main__":
    main()
