# Snapshot-based Dupaco knowledge system

## What it does

`solution.py` implements the assessment's `build()` contract. The returned object exposes `ingest(snapshot)`, `answer(question, as_of=None)`, and `conflicts()`. It works entirely from the supplied website snapshots: no network requests and no language-model API calls.

## Indexing and retrieval

Each snapshot is indexed independently. The index splits Markdown into text passages and table rows, adds page titles and URL tokens, and ranks them with BM25. The answer path reranks relevant passages, with extra preference for official product and rate pages when the question asks for financial values.

For structured topics (mortgages, auto and personal loans, certificates, savings, checking, card APRs, branch schedules, grants, membership, and contact details), the implementation reads relevant snapshot passages and checks specific qualifiers before answering. These rules aim to avoid mixing APR with APY, confusing products, or substituting a different loan term or branch.

## Time handling

With no `as_of` date, `answer` uses the latest ingested snapshot. With `as_of`, it uses the newest snapshot at or before that date. Source records carry their URL and snapshot ID. Snapshot selection does not itself prove that every page inside a crawl contains up-to-date information: dated notices and published rate sheets still need to be interpreted cautiously.

## Conflicts

`conflicts()` reports conflicting certificate APYs and auto-loan APRs across snapshots when the facts refer to equivalent product qualifiers. It also compares auto-loan terms and overlapping model-year ranges across pages in the latest snapshot, preferring newer effective dates and official rate sources when those details exist.

## Running it

From the project root:

```bash
python3.11 -m pip install -r submission/requirements.txt
python3.11 harness/run_eval.py \
  --submission submission \
  --snapshots snapshots \
  --eval eval/public_eval.jsonl \
  --out results
```

The harness automatically extracts the supplied snapshot archives if needed.

## Verified public results

The author's local public evaluation reached **0.9643 (96.43%)** on 40 supplied questions, with **zero runtime errors**, full consistency score, and full speed score. These figures are not hidden-test estimates. The code can still miss unseen phrasing, uncommon branches, or newly introduced page layouts.

## Additional local validation

The implementation was also checked with locally constructed tests:

| Test | Result |
|---|---|
| Reworded questions | 10/10 |
| Historical-rate paraphrases | 4/4 |
| Certificate product attribution | 6/6 |
| Snapshot date selection | 15/15 |

These tests checked question phrasing, historical rates, product distinctions, and snapshot selection. They are author-created checks, not part of the official public or hidden evaluation.

## Dependencies and files

- `solution.py`: entry point and implementation
- `requirements.txt`: `rank-bm25==0.2.2` (which also installs NumPy)
- `NOTES.md`: runtime and environment details

Python's standard library handles the remaining functionality. The harness supplies the `contract` definitions.
