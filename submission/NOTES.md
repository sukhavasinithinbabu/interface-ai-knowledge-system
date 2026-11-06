# Implementation notes

## Runtime

- Python 3.11
- Dependency: rank-bm25==0.2.2
- No external APIs, network requests, or LLM services
- Uses the assessment's provided contract definitions

## Approach

The implementation indexes each website snapshot separately
and uses BM25 retrieval with question-specific reranking.

It handles historical questions by selecting the latest
available snapshot on or before the requested date.

For structured financial information, the answer logic
checks product types, loan terms, model years, and rate
qualifiers before returning values.

Answers include supporting source URLs and snapshot IDs.
The conflict-reporting logic compares matching products
and qualifiers across snapshots.

## Local evaluation

Public evaluation: 96.43% across 40 questions.

Additional author-created checks:
- Reworded questions: 10/10
- Historical paraphrases: 4/4
- Certificate attribution: 6/6
- Snapshot selection: 15/15

These results are from local testing, not the hidden evaluation.

## Limitations

Retrieval can miss uncommon question wording or information
presented in unfamiliar website layouts.

Historical answers reflect the supplied snapshot evidence,
which may not capture every change between crawls.
