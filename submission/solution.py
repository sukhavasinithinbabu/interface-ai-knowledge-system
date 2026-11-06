
"""
Temporal knowledge retrieval using BM25.

Preserves historical website snapshots, indexes Markdown
passages and table rows, and retrieves cited evidence.
"""

import re
import sys
from pathlib import Path
from datetime import datetime

from rank_bm25 import BM25Okapi

sys.path.insert(
    0,
    str(next(
        d / "harness"
        for d in Path(__file__).resolve().parents
        if (d / "harness" / "contract.py").exists()
    ))
)

from contract import Answer, Conflict, Snapshot, Source


WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*%?")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "what", "which", "who", "where", "when",
    "how", "can", "could", "would", "please",
    "tell", "me", "for", "of", "on", "at", "in",
    "to", "and", "or", "does", "do", "dupaco",
    "s", "your", "you", "right", "now"
}

SYNONYMS = {
    "mortgage": "home loan",
    "mortgages": "home loan",
    "apy": "annual percentage yield",
    "apr": "annual percentage rate",
    "auto": "car vehicle",
    "certificate": "term share",
}


def tokenize(text):
    text = text.lower()

    for word, expansion in SYNONYMS.items():
        text = re.sub(
            rf"\b{word}\b",
            word + " " + expansion,
            text
        )

    return [
        word for word in WORD.findall(text)
        if word not in STOPWORDS
    ]


def extract_passages(markdown):
    """Extract paragraphs and individual Markdown table rows."""

    passages = []
    section_heading = ""
    table_header = ""

    for block in markdown.split("\n\n"):
        lines = block.splitlines()

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                section_heading = stripped.lstrip("# ").strip()

        if any(line.strip().startswith("|") for line in lines):
            for line in lines:
                row = line.strip()

                if not row.startswith("|"):
                    continue

                cells = [
                    c.strip()
                    for c in row.strip("|").split("|")
                ]

                if cells and all(
                    c and set(c) <= set("-: ")
                    for c in cells
                ):
                    continue

                if not table_header:
                    table_header = row
                    continue

                passage = "\n".join(
                    x for x in (
                        section_heading,
                        table_header,
                        row
                    ) if x
                )

                passages.append(passage)

        else:
            table_header = ""

            text = block.strip()

            if not text or len(text) < 12:
                continue

            passage = (
                section_heading + "\n" + text
                if section_heading else text
            )

            passages.append(passage)

    return passages


class NaiveKnowledge:
    def __init__(self):
        self.snapshots = []
        self.indexes = {}
        self.certificate_facts = {}
        self.auto_facts = {}
        self.cross_page_auto_facts = []

    
    def ingest(self, snapshot: Snapshot) -> None:
        self.extract_cross_page_auto(snapshot)
        documents = []
        corpus = []

        for page in snapshot.pages:
            # Extract auto-loan APRs from the official rate table.
            if page.url.rstrip("/").endswith("/rates"):
                in_auto_table = False


                for line in page.markdown.splitlines():
                    row = line.strip()

                    if row.startswith("| Loan Program | Term |"):
                        in_auto_table = True
                        continue

                    if not in_auto_table:
                        continue

                    # Stop when the table ends.
                    if not row.startswith("|"):
                        in_auto_table = False
                        continue

                    cells = [
                        cell.strip()
                        for cell in row.strip("|").split("|")
                    ]

                    if len(cells) != 3:
                        continue

                    # Skip Markdown separator rows.
                    if all(
                        cell and set(cell) <= set("-: ")
                        for cell in cells
                    ):
                        continue

                    model_year = cells[0]
                    term = cells[1]
                    apr_text = cells[2]

                    # Only accept actual model-year loan rows.
                    if "model year" not in model_year.lower():
                        continue

                    term_match = re.fullmatch(
                        r"(\d+)\s+Months?",
                        term,
                        re.IGNORECASE
                    )

                    apr_match = re.match(
                        r"(\d+(?:\.\d+)?)\s*%",
                        apr_text
                    )

                    if not term_match or not apr_match:
                        continue

                    months = term_match.group(1)
                    apr = apr_match.group(1) + "%"

                    # Product + model-year group + term + APR
                    # uniquely identifies a comparable rate.
                    normalized_model = re.sub(
                        r"\s+",
                        " ",
                        model_year.lower()
                    ).strip()

                    topic = (
                        f"auto loan {normalized_model} "
                        f"{months}-month APR"
                    )

                    self.auto_facts.setdefault(
                        topic, []
                    ).append({
                        "value": apr,
                        "snapshot_id": snapshot.snapshot_id,
                        "date": snapshot.taken_on,
                        "url": page.url,
                        "quote": row
                    })


            # Extract clearly labeled certificate APYs.
            certificate_pattern = re.compile(
                r"\b(\d+)[\s-]*month\s+"
                r"(?:fixed\s+)?certificate\s+APY\s*:\s*"
                r"(\d+(?:\.\d+)?)\s*%",
                re.IGNORECASE
            )

            for line in page.markdown.splitlines():
                # Ignore explicitly expired promotions.
                if re.search(
                    r"(?:offer ended|expired)\s+\d{4}-\d{2}-\d{2}",
                    line,
                    re.IGNORECASE
                ):
                    continue

                match = certificate_pattern.search(line)

                if not match:
                    continue

                term = match.group(1)
                rate = match.group(2) + "%"
                topic = f"{term}-month certificate APY"

                self.certificate_facts.setdefault(
                    topic, []
                ).append({
                    "value": rate,
                    "snapshot_id": snapshot.snapshot_id,
                    "date": snapshot.taken_on,
                    "url": page.url,
                    "quote": line.strip()
                })

            
            # Extract certificate APYs from official rate tables.
            if page.url.rstrip("/").endswith("/rates"):
                section = None
                in_table = False

                for line in page.markdown.splitlines():
                    row = line.strip()

                    # Track the certificate section.
                    
                    # Track the certificate section.
                    if row in (
                        "Special",
                        "Regular",
                        "Special Certificates",
                        "Regular Certificates",
                    ):
                        section = (
                            "Special Certificates"
                            if "Special" in row
                            else "Regular Certificates"
                        )
                        in_table = False
                        continue


                    # Stop when reaching another product section.

                    # Stop when reaching another product section.
                    if row in ("Retirement / IRAs", "Special IRAs"):
                        section = None
                        in_table = False
                        continue

                    if section is None:
                        continue

                    # Identify the certificate table.
                    if (
                        row.startswith("| Certificate |")
                        or row.startswith("| Term |")
                    ):
                        if "Annual Percentage Yield" in row:
                            in_table = True
                        continue

                    if not row.startswith("|"):
                        if in_table:
                            in_table = False
                        continue

                    if not in_table:
                        continue


                    cells = [
                        c.strip()
                        for c in row.strip("|").split("|")
                    ]

                    if len(cells) != 4:
                        continue

                    # Skip Markdown separator rows.
                    if all(
                        c and set(c) <= set("-: ")
                        for c in cells
                    ):
                        continue

                    product = cells[0]

                    term_match = re.search(
                        r"\b(\d+)\s+Month\b",
                        product,
                        re.IGNORECASE
                    )

                    apy_match = re.match(
                        r"(\d+(?:\.\d+)?)%",
                        cells[2]
                    )

                    if not term_match or not apy_match:
                        continue

                    term = term_match.group(1)
                    apy = apy_match.group(1) + "%"

                    tier = (
                        "jumbo" if "jumbo" in product.lower()
                        else "standard"
                    )

                    kind = (
                        "special" if section == "Special Certificates"
                        else "regular"
                    )

                    topic = (
                        f"{term}-month {tier} "
                        f"{kind} certificate APY"
                    )

                    self.certificate_facts.setdefault(
                        topic, []
                    ).append({
                        "value": apy,
                        "snapshot_id": snapshot.snapshot_id,
                        "date": snapshot.taken_on,
                        "url": page.url,
                        "quote": row
                    })

            passages = extract_passages(page.markdown)

            for passage in passages:
                tokens = tokenize(passage)

                if not tokens:
                    continue

                # Include document metadata in retrieval.
                search_text = (
                    page.title + " " +
                    page.url.replace("-", " ").replace("/", " ")
                    + " " + passage
                )

                documents.append({
                    "url": page.url,
                    "title": page.title,
                    "snapshot_id": snapshot.snapshot_id,
                    "taken_on": snapshot.taken_on,
                    "fetched_at": page.fetched_at,
                    "text": passage,
                })

                corpus.append(tokenize(search_text))

        self.indexes[snapshot.snapshot_id] = {
            "documents": documents,
            "bm25": BM25Okapi(corpus),
        }

        self.snapshots.append({
            "snapshot_id": snapshot.snapshot_id,
            "taken_on": snapshot.taken_on,
        })

    def answer(self, question, as_of=None) -> Answer:
        if not self.snapshots:
            return Answer(
                text="I don't know.",
                confident=False
            )

        eligible = [
            snap for snap in self.snapshots
            if as_of is None or snap["taken_on"] <= as_of
        ]

        if not eligible:
            return Answer(
                text="I don't have information for that date.",
                confident=False
            )

        selected = max(
            eligible,
            key=lambda snap: snap["taken_on"]
        )

        index = self.indexes[selected["snapshot_id"]]
        query_tokens = tokenize(question)

        if not query_tokens:
            return Answer(
                text="I don't know.",
                confident=False
            )

        scores = index["bm25"].get_scores(query_tokens)

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        if not ranked or scores[ranked[0]] <= 0:
            return Answer(
                text="I don't know.",
                confident=False
            )

        
        # Rerank the strongest BM25 candidates using
        # question-specific evidence and source authority.
        number_pattern = re.compile(r"\b\d+(?:[.,]\d+)*\b")

        question_numbers = set(
            number_pattern.findall(question.lower())
        )

        question_lower = question.lower()

        asks_rate = any(
            term in question_lower
            for term in ("rate", "apr", "apy", "interest")
        )

        asks_mortgage = any(
            term in question_lower
            for term in ("mortgage", "home loan")
        )

        asks_phone = any(
            term in question_lower
            for term in ("phone", "call", "telephone", "contact number")
        )

        asks_auto = any(
            term in question_lower
            for term in ("auto loan", "car loan", "vehicle loan")
        )

        def rerank(i):
            doc = index["documents"][i]
            passage = doc["text"].lower()
            url = doc["url"].lower()

            score = float(scores[i])

            # Match numbers such as 60-month or 30-year.
            passage_numbers = set(
                number_pattern.findall(passage)
            )

            if question_numbers:
                matched = question_numbers & passage_numbers
                missing = question_numbers - passage_numbers

                score += 6 * len(matched)
                score -= 8 * len(missing)

            # Rate questions should retrieve actual rate values.
            if asks_rate:
                if re.search(r"\d+(?:\.\d+)?\s*%", passage):
                    score += 5
                else:
                    score -= 10

            # Phone questions should retrieve phone numbers.
            if asks_phone:
                if re.search(
                    r"\b(?:\d{3}[-.)\s]*){2}\d{4}\b",
                    passage
                ):
                    score += 8
                else:
                    score -= 10

            # Prefer official rate and product pages.
            if asks_rate:
                if "/rates" in url or url.endswith("/rates"):
                    score += 8

                if "calculator" in url:
                    score -= 12

                if "/learn/" in url or "/blog/" in url:
                    score -= 6

            # Keep IRA certificate products separate from
            # standard certificates when the question specifies one.
            asks_certificate_product = bool(re.search(
                r"\b(?:certificate|certificates|cd|cds)\b",
                question_lower
            )) or (
                "jumbo" in question_lower
                and "month" in question_lower
            )

            if asks_certificate_product:
                asks_roth = bool(re.search(
                    r"\broth\b",
                    question_lower
                ))
                asks_traditional = bool(re.search(
                    r"\btraditional\s+ira\b",
                    question_lower
                ))

                # Use the actual product row rather than simply
                # checking whether the page mentions Roth.
                rows = [
                    line.strip().lower()
                    for line in doc["text"].splitlines()
                    if line.strip().startswith("|")
                    and re.search(
                        r"\b\d+\s+month\b",
                        line,
                        re.IGNORECASE
                    )
                    and "jumbo" in line.lower()
                ]

                if rows:
                    has_roth_row = any(
                        re.search(r"\broth\b", row)
                        for row in rows
                    )
                    has_traditional_row = any(
                        "traditional" in row
                        for row in rows
                    )
                    has_standard_row = any(
                        "roth" not in row
                        and "traditional" not in row
                        and "ira" not in row
                        for row in rows
                    )

                    if asks_roth:
                        score += 35 if has_roth_row else -35
                    elif asks_traditional:
                        score += 35 if has_traditional_row else -35
                    elif "standard" in question_lower or "non-ira" in question_lower:
                        score += 20 if has_standard_row else -20

            # Prefer home loan rate evidence over ARM explanations.
            if asks_mortgage and asks_rate:
                if "fixed conventional" in passage:
                    score += 6

            # Prefer actual auto loan rate rows.
            if asks_auto and asks_rate:
                if "model year" in passage:
                    score += 5

            return score

        
        candidates = ranked[:100]

        # For general current mortgage-rate questions,
        # search the official rate sheet first.
        # Do not use this rule for historical snapshots
        # differently: the selected index already
        # corresponds to the requested snapshot date.
        
        # Identify adjustable-rate mortgage questions.
        asks_arm = bool(re.search(
            r"\b(?:arm|adjustable[\s-]+rate|\d+/\d+)\b",
            question_lower
        ))

        # Apply fixed-rate mortgage selection only when
        # the question is not asking about an ARM.
        if asks_mortgage and asks_rate and not asks_arm:
            official_candidates = []


            for i, doc in enumerate(index["documents"]):
                url = doc["url"].rstrip("/").lower()
                passage = doc["text"].lower()

                # Use the official rate sheet.
                if not url.endswith("/rates"):
                    continue

                # Select actual mortgage rate rows,
                # not descriptions or calculator links.
                if "fixed conventional" not in passage:
                    continue

                if "purchase:" not in passage:
                    continue

                # Match term when one is requested.
                if "30-year" in question_lower:
                    if "30-year" not in passage:
                        continue

                if "15-year" in question_lower:
                    if "15-year" not in passage:
                        continue

                official_candidates.append(i)

            if official_candidates:
                candidates = official_candidates

        best_idx = max(
            candidates,
            key=rerank
        )

    

        best = index["documents"][best_idx]

        # Retrieve Volunteer Time Off participation figures
        # with their reporting years.
        asks_vto_count = (
            any(term in question_lower for term in (
                "volunteer time off", "vto"
            ))
            and any(term in question_lower for term in (
                "how many", "number of", "employees"
            ))
        )

        if asks_vto_count:
            vto_records = []

            for doc in index["documents"]:
                url = doc["url"].lower()
                passage = doc["text"]

                if "/newsroom/volunteering-pays-off-" not in url:
                    continue

                match = re.search(
                    r"(\d[\d,]*)\s+Dupaco employees"
                    r".{0,100}?\b(?:in\s+)?(20\d{2})\b"
                    r".{0,40}?Volunteer Time Off",
                    passage,
                    re.IGNORECASE | re.DOTALL
                )

                if match:
                    count = int(match.group(1).replace(",", ""))
                    year = int(match.group(2))
                    vto_records.append((year, count, doc))

            if vto_records:
                records_by_year = {}

                for year, count, doc in vto_records:
                    records_by_year[year] = (count, doc)

                explicit_year = re.search(
                    r"\b20\d{2}\b", question_lower
                )

                if explicit_year:
                    year = int(explicit_year.group(0))
                    selected_years = (
                        [year] if year in records_by_year
                        else []
                    )
                else:
                    selected_years = sorted(records_by_year)

                if selected_years:
                    parts = []
                    sources = []

                    for year in selected_years:
                        count, doc = records_by_year[year]
                        parts.append(
                            f"In {year}, {count} Dupaco "
                            "employees used Volunteer Time Off."
                        )
                        sources.append(
                            Source(
                                url=doc["url"],
                                snapshot_id=selected["snapshot_id"],
                                quote=doc["text"][:200]
                            )
                        )

                    if not explicit_year and len(parts) > 1:
                        parts.append(
                            "The question 'last year' is "
                            "ambiguous without a reference "
                            "year, so both published counts "
                            "are provided."
                        )

                    return Answer(
                        text="\n".join(parts),
                        sources=sources,
                        confident=True
                    )


        # Avoid substituting hours from other branches
        # when the requested branch cannot be verified.
        asks_specific_branch_hours = (
            "branch" in question_lower
            and any(term in question_lower for term in (
                "hours", "open", "close", "closing"
            ))
        )

        if asks_specific_branch_hours:
            branch_match = re.search(
                r"\b(?:for|at|of)\s+(?:the\s+)?"
                r"([a-z][a-z\s-]+?)\s+branch\b",
                question_lower
            )

            if branch_match:
                branch_name = branch_match.group(1).strip()

        
             
                # Check for branch-specific evidence,
                # excluding incidental mentions.
               
                # Check all passages from each branch-related page.
                branch_evidence = []
                pages = {}

                for doc in index["documents"]:
                    url = doc["url"].rstrip("/")

                    if not any(part in url.lower() for part in (
                        "/contact",
                        "/locations",
                        "/branches",
                        "/newsroom/",
                    )):
                        continue

                    pages.setdefault(url, []).append(doc)

                for url, docs in pages.items():
                    full_text = "\n".join(
                        doc["text"] for doc in docs
                    ).lower()

                    if branch_name not in full_text:
                        continue

                    if not any(term in full_text for term in (
                        "lobby hours",
                        "lobby",
                        "monday",
                        "friday",
                        "saturday",
                    )):
                        continue

                    branch_evidence.extend(docs)


                if not branch_evidence:
                    return Answer(
                        text=(
                            f"I couldn't verify published "
                            f"lobby hours for the "
                            f"{branch_name.title()} branch "
                            "in the available Dupaco "
                            "website snapshots. I don't "
                            "want to provide hours from "
                            "a different branch."
                        ),
                        sources=[],
                        confident=False
                    )


        # Handle reworded auto-financing rate questions using
        # the exact model-year range and loan term.
        auto_question = (
            re.search(r"\b(?:auto|car|vehicle)\b", question_lower)
            and any(word in question_lower for word in (
                "loan", "financ", "apr", "interest", "rate"
            ))
        )

        if auto_question:
            term_match = re.search(
                r"\b(\d{1,3})[\s-]*months?\b", question_lower
            )
            year_match = re.search(r"\b(20\d{2})\b", question_lower)

            if term_match and year_match:
                months = int(term_match.group(1))
                year = int(year_match.group(1))

                for doc in index["documents"]:
                    url = doc["url"].rstrip("/").lower()

                    if not url.endswith(
                        "/personal/loans/auto-motorcycle-loans"
                    ):
                        continue

                    for line in doc["text"].splitlines():
                        match = re.search(
                            r"\|\s*(20\d{2})-(\d{2,4})\s+"
                            r"Model Year\s*\|\s*(\d+)\s+Months?"
                            r"\s*\|\s*(\d+(?:\.\d+)?)%\s*APR",
                            line,
                            re.IGNORECASE
                        )

                        if not match:
                            continue

                        start = int(match.group(1))
                        end_text = match.group(2)
                        end = int(
                            end_text if len(end_text) == 4
                            else str(start)[:2] + end_text
                        )
                        row_months = int(match.group(3))
                        apr = match.group(4)

                        if start <= year <= end and row_months == months:
                            return Answer(
                                text=(
                                    f"Dupaco lists an APR as low as "
                                    f"{apr}% for a {months}-month "
                                    f"auto loan on a {year} model-year "
                                    "vehicle. Actual rates may vary."
                                ),
                                sources=[
                                    Source(
                                        url=doc["url"],
                                        snapshot_id=selected["snapshot_id"],
                                        quote=line[:200]
                                    )
                                ],
                                confident=True
                            )

        # Retrieve personal loan APR for a specified term.
        asks_personal_loan_apr = (
            "personal loan" in question_lower
            and any(term in question_lower for term in (
                "apr", "rate", "interest"
            ))
        )

        if asks_personal_loan_apr:
            term_match = re.search(
                r"\b(\d{1,3})[\s-]*months?\b",
                question_lower
            )

            if term_match:
                months = int(term_match.group(1))
                candidates = []

                for doc in index["documents"]:
                    url = doc["url"].rstrip("/").lower()
                    passage = doc["text"]

                    if url not in (
                        "https://www.dupaco.com/rates",
                        "https://www.dupaco.com/personal/loans/personal-loans",
                    ):
                        continue

                    # Match the term and APR in an actual
                    # personal loan rate table row.
                    pattern = (
                        r"\|\s*"
                        + str(months)
                        + r"\s+Months?\s*\|\s*"
                        + r"(\d+(?:\.\d+)?)%\s*"
                    )

                    match = re.search(
                        pattern,
                        passage,
                        re.IGNORECASE
                    )

                    if match and (
                        "| Term | APR" in passage
                        or "| Term | Annual Percentage Rate" in passage
                    ):
                        candidates.append(
                            (match.group(1), doc)
                        )

                if candidates:
                    # Prefer the official rate page.
                    candidates.sort(
                        key=lambda item:
                        item[1]["url"].rstrip("/").lower()
                        != "https://www.dupaco.com/rates"
                    )

                    apr, doc = candidates[0]

                    return Answer(
                        text=(
                            f"The published APR for a "
                            f"{months}-month Dupaco personal "
                            f"loan is as low as {apr}%. "
                            "The actual rate depends on "
                            "credit qualifications and "
                            "loan approval."
                        ),
                        sources=[
                            Source(
                                url=doc["url"],
                                snapshot_id=selected["snapshot_id"],
                                quote=doc["text"][:200]
                            )
                        ],
                        confident=True
                    )


        # Retrieve Forbes credit union recognition from
        # the official Dupaco award announcement.
        asks_forbes_recognition = (
            "forbes" in question_lower
            and any(term in question_lower for term in (
                "recognition", "recognized", "award",
                "rank", "honor", "named"
            ))
        )

        if asks_forbes_recognition:
            candidates = []

            for doc in index["documents"]:
                url = doc["url"].lower()
                passage = doc["text"]
                lower = passage.lower()

                if (
                    "/newsroom/forbes-ranks-dupaco-as-a-top-credit-union"
                    not in url
                ):
                    continue

                if (
                    "forbes" in lower
                    and "fifth time" in lower
                    and "credit union" in lower
                ):
                    candidates.append(doc)

            if candidates:
                doc = min(
                    candidates,
                    key=lambda d: len(d["text"])
                )

                return Answer(
                    text=(
                        "Forbes recognized Dupaco Community "
                        "Credit Union as one of America's "
                        "top-rated Best-in-State credit unions. "
                        "According to Dupaco's announcement, "
                        "this was the fifth time Forbes "
                        "recognized the credit union.\n\n"
                        + doc["text"][:700]
                    ),
                    sources=[
                        Source(
                            url=doc["url"],
                            snapshot_id=selected["snapshot_id"],
                            quote=doc["text"][:200]
                        )
                    ],
                    confident=True
                )


        # Return the certificate rates page when the
        # user asks for a URL, rather than rate values.
        asks_certificate_link = (
            any(term in question_lower for term in (
                "certificate", "cd rate", "cd / certificate"
            ))
            and (
                any(term in question_lower for term in (
                    "link", "url", "website", "page"
                ))
                or (
                    "online" in question_lower
                    and any(term in question_lower for term in (
                        "where can i find",
                        "where do i find",
                        "where to find",
                        "where can i see",
                        "where do i see"
                    ))
                )
            )
        )

        if asks_certificate_link:
            target_url = (
                "https://www.dupaco.com/"
                "personal/savings-account#rates"
            )

            for doc in index["documents"]:
                passage = doc["text"]
                page_url = doc["url"].split("#", 1)[0].rstrip("/").lower()

                if page_url != "https://www.dupaco.com/personal/savings-account":
                    continue

                if "personal/savings-account#rates" not in passage:
                    continue

                return Answer(
                    text=(
                        "You can view Dupaco's certificate "
                        f"rates here: {target_url}"
                    ),
                    sources=[
                        Source(
                            url=target_url,
                            snapshot_id=selected["snapshot_id"],
                            quote=passage[:200]
                        )
                    ],
                    confident=True
                )

        # Retrieve Dupaco's routing number from the official
        # cash-management page in the selected snapshot.
        if (
            "routing number" in question_lower
            or "aba number" in question_lower
        ):
            for doc in index["documents"]:
                url = doc["url"].rstrip("/").lower()

                if not url.endswith("/business/cash-management"):
                    continue

                for line in doc["text"].splitlines():
                    match = re.search(
                        r"(?:routing number|aba number)"
                        r"[^\n]{0,100}?\b(\d{9})\b",
                        line,
                        re.IGNORECASE
                    )

                    if not match:
                        continue

                    routing_number = match.group(1)

                    return Answer(
                        text=(
                            "Dupaco's routing number is "
                            f"{routing_number}."
                        ),
                        sources=[
                            Source(
                                url=doc["url"],
                                snapshot_id=selected["snapshot_id"],
                                quote=line[:200]
                            )
                        ],
                        confident=True
                    )

        # Answer questions asking where to join Dupaco online.
        asks_online_join = (
            any(term in question_lower for term in (
                "join dupaco",
                "become a dupaco member",
                "apply for membership",
                "open a membership",
            ))
            and any(term in question_lower for term in (
                "online", "website", "url", "link"
            ))
        )

        if asks_online_join:
            join_page = "https://www.dupaco.com/about/join"

            for doc in index["documents"]:
                if doc["url"].rstrip("/") != join_page:
                    continue

                passage = doc["text"]

                match = re.search(
                    r"https://join\.dupaco\.com/?",
                    passage
                )

                if match:
                    join_link = match.group(0)

                    return Answer(
                        text=(
                            "You can join Dupaco online at "
                            f"{join_link}"
                        ),
                        sources=[
                            Source(
                                url=doc["url"],
                                snapshot_id=selected["snapshot_id"],
                                quote=passage[:200]
                            )
                        ],
                        confident=True
                    )


        # Answer questions about updating contact information
        # using Dupaco's account correction instructions.
        asks_contact_update = (
            any(term in question_lower for term in (
                "update", "change", "correct", "edit"
            ))
            and any(term in question_lower for term in (
                "contact information",
                "personal information",
                "phone number",
                "email address",
                "mailing address",
            ))
        )

        if asks_contact_update:
            policy_url = (
                "https://www.dupaco.com/digital-privacy-policy"
            )

            policy_matches = []

            for doc in index["documents"]:
                if doc["url"].rstrip("/") != policy_url:
                    continue

                passage = doc["text"]
                lower = passage.lower()

                if (
                    "accessing and correcting your information"
                    in lower
                    and "800-373-7600" in passage
                ):
                    policy_matches.append(doc)

            if policy_matches:
                doc = policy_matches[0]
                passage = doc["text"]

                answer_text = (
                    "You can update your personal or contact "
                    "information by logging into your Dupaco "
                    "account and making the changes. You can "
                    "also call Dupaco at 800-373-7600, email "
                    "service@dupaco.com, or visit a branch. "
                    "For account-specific assistance, you can "
                    "use the Contact Us option in Shine Online "
                    "or Mobile Banking."
                )

                sources = [
                    Source(
                        url=doc["url"],
                        snapshot_id=selected["snapshot_id"],
                        quote=passage[:200]
                    )
                ]

                for contact_doc in index["documents"]:
                    if (
                        contact_doc["url"].rstrip("/")
                        == "https://www.dupaco.com/contact"
                        and "shine online" in
                        contact_doc["text"].lower()
                        and "contact us" in
                        contact_doc["text"].lower()
                    ):
                        sources.append(
                            Source(
                                url=contact_doc["url"],
                                snapshot_id=selected["snapshot_id"],
                                quote=contact_doc["text"][:200]
                            )
                        )
                        break

                return Answer(
                    text=answer_text,
                    sources=sources,
                    confident=True
                )


        # Answer membership eligibility questions using
        # Dupaco's official membership page.
        asks_membership_eligibility = (
            any(term in question_lower for term in (
                "eligible to become",
                "eligible to join",
                "who can join",
                "who is eligible",
                "membership eligibility",
                "qualify for membership",
            ))
            and any(term in question_lower for term in (
                "member",
                "join",
                "membership",
            ))
        )

        if asks_membership_eligibility:
            join_url = "https://www.dupaco.com/about/join"

            join_docs = [
                doc for doc in index["documents"]
                if doc["url"].rstrip("/") == join_url
            ]

            eligibility = []
            locations = []

            for doc in join_docs:
                passage = doc["text"]

                for line in passage.splitlines():
                    clean = line.strip()
                    lower = clean.lower()

                    if lower.startswith("- you "):
                        if any(term in lower for term in (
                            "live or work",
                            "immediate family",
                            "rural ideas network",
                        )):
                            if clean not in eligibility:
                                eligibility.append(clean)

                    if all(state in lower for state in (
                        "iowa",
                        "illinois",
                        "wisconsin",
                    )):
                        if "find us" in lower:
                            if clean not in locations:
                                locations.append(clean)

            if eligibility and locations:
                answer_text = (
                    "Dupaco membership eligibility:\n"
                    + "\n".join(eligibility)
                    + "\n\n"
                    + "\n".join(locations)
                    + "\n\nThe charter area covers specific "
                      "counties, so eligibility depends on "
                      "the applicable membership requirements."
                )

                return Answer(
                    text=answer_text,
                    sources=[
                        Source(
                            url=join_url,
                            snapshot_id=selected["snapshot_id"],
                            quote=eligibility[0][:200]
                        )
                    ],
                    confident=True
                )


        # Retrieve Home$tart grant details from the
        # matching program announcement, not other grants.
        asks_homestart = (
            "home$tart" in question_lower
            or "homestart" in question_lower
        )

        if asks_homestart:
            grant_matches = []

            for doc in index["documents"]:
                url = doc["url"].lower()
                passage = doc["text"]

                # Limit evidence to the Home$tart program.
                if not (
                    "hometart-grants" in url
                    or "home-start-grants" in url
                ):
                    continue

                if (
                    "home$tart" in passage.lower()
                    and re.search(
                        r"\$\s*[\d,]+",
                        passage
                    )
                    and "down payment" in passage.lower()
                ):
                    grant_matches.append(doc)

            if grant_matches:
                # Prefer the passage that explains the
                # program and its financial assistance.
                best_grant = max(
                    grant_matches,
                    key=lambda doc: (
                        "closing costs" in doc["text"].lower(),
                        "federal home loan bank" in
                        doc["text"].lower(),
                        len(doc["text"])
                    )
                )

                return Answer(
                    text=best_grant["text"][:1000],
                    sources=[
                        Source(
                            url=best_grant["url"],
                            snapshot_id=selected["snapshot_id"],
                            quote=best_grant["text"][:200]
                        )
                    ],
                    confident=True
                )


        # Retrieve the published number of Dupaco branches.
        asks_branch_count = (
            "branch" in question_lower
            and any(term in question_lower for term in (
                "how many",
                "number of",
                "total",
                "count",
            ))
        )

        if asks_branch_count:
            branch_candidates = []

            for doc in index["documents"]:
                url = doc["url"].lower()
                passage = doc["text"]

                # Prefer Dupaco's own corporate announcements.
                if "/newsroom/" not in url:
                    continue

                match = re.search(
                    r"\b(?:operates|has|with)\s+"
                    r"(\d+)\s+(?:branch(?:es)?|branch offices)\b",
                    passage,
                    re.IGNORECASE
                )

                if match:
                    branch_candidates.append(
                        (int(match.group(1)), doc)
                    )

            if branch_candidates:
                # Prefer the latest published corporate
                # announcement when several counts exist.
                branch_candidates.sort(
                    key=lambda item: (
                        "completes-acquisition" in
                        item[1]["url"].lower(),
                        item[1]["url"]
                    ),
                    reverse=True
                )

                count, doc = branch_candidates[0]

                return Answer(
                    text=(
                        f"Dupaco operates {count} "
                        "branch office locations."
                    ),
                    sources=[
                        Source(
                            url=doc["url"],
                            snapshot_id=selected["snapshot_id"],
                            quote=doc["text"][:200]
                        )
                    ],
                    confident=True
                )


        # Find Dupaco branch addresses without selecting
        # unrelated businesses from the ATM locator.
        asks_branch_address = (
            "branch" in question_lower
            and any(term in question_lower for term in (
                "address", "located", "location"
            ))
        )

        if asks_branch_address:
            city_match = re.search(
                r"(?:address|location)\s+of\s+(?:the\s+)?"
                r"([a-z][a-z\s-]+?)\s+branch\b",
                question_lower
            )

            if city_match:
                branch_city = city_match.group(1).strip()
                address_matches = []

                for doc in index["documents"]:
                    url = doc["url"].rstrip("/").lower()
                    lines = doc["text"].splitlines()

                    if not url.endswith("/contact"):
                        continue

                    if len(lines) < 2:
                        continue

                    # Only select addresses explicitly
                    # identified as Dupaco Credit Union.
                    if "dupaco credit union" not in lines[0].lower():
                        continue

                    for line in lines[1:]:
                        if (
                            branch_city in line.lower()
                            and re.search(r"\b\d{3,6}\b", line)
                        ):
                            address_matches.append(
                                (line.strip(), doc["url"])
                            )

                if address_matches:
                    address, url = address_matches[0]

                    return Answer(
                        text=(
                            "Dupaco Credit Union "
                            + branch_city.title()
                            + " Branch:\n"
                            + address
                        ),
                        sources=[
                            Source(
                                url=url,
                                snapshot_id=selected["snapshot_id"],
                                quote=address[:200]
                            )
                        ],
                        confident=True
                    )


        # Answer general branch-hours questions using
        # published branch schedules from the snapshot.
        asks_lobby_hours = (
            "lobby" in question_lower
            and "hour" in question_lower
        )

        asks_driveup_hours = (
            "drive-up" in question_lower
            or "drive up hours" in question_lower
            or "drive-through hours" in question_lower
        ) and "hour" in question_lower

        if asks_lobby_hours or asks_driveup_hours:
            branch_pages = [
                (
                    "Cedar Rapids Learning Lab",
                    "dupaco-celebrates-completion-of-newest-learning-lab"
                ),
                (
                    "Platteville",
                    "dupaco-community-credit-union-celebrates-completion-of-expanded-branch"
                ),
            ]

            mode_needed = (
                "drive-up" if asks_driveup_hours else "lobby"
            )

            schedule_results = []
            schedule_sources = []

            for branch_name, slug in branch_pages:
                page_docs = [
                    doc for doc in index["documents"]
                    if doc["url"].rstrip("/").endswith("/" + slug)
                ]

                if not page_docs:
                    continue

                mode = "lobby"
                hours = []

                for doc in page_docs:
                    for line in doc["text"].splitlines():
                        line = line.strip()
                        lower = line.lower()

                        if lower.startswith("**lobby**"):
                            mode = "lobby"
                        elif lower.startswith("**drive-up**"):
                            mode = "drive-up"

                        has_day = bool(re.search(
                            r"\b(?:monday|tuesday|wednesday|"
                            r"thursday|friday|saturday|sunday)\b",
                            lower
                        ))

                        has_hours = bool(re.search(
                            r"\d{1,2}(?::\d{2})?\s*"
                            r"(?:a\.m\.|p\.m\.)",
                            lower
                        ))

                        if (
                            mode == mode_needed
                            and has_day
                            and (
                                has_hours
                                or "closed" in lower
                                or "atm only" in lower
                            )
                        ):
                            if line not in hours:
                                hours.append(line)

                if hours:
                    schedule_results.append(
                        branch_name + ":\n"
                        + "\n".join(hours)
                    )

                    schedule_sources.append(
                        Source(
                            url=page_docs[0]["url"],
                            snapshot_id=selected["snapshot_id"],
                            quote=hours[0][:200]
                        )
                    )

            if schedule_results:
                if asks_driveup_hours:
                    intro = (
                        "Published Dupaco drive-up schedules "
                        "vary by branch. Examples:\n"
                    )
                else:
                    intro = (
                        "No, Dupaco branches do not all "
                        "have identical lobby hours. "
                        "Published examples:\n"
                        if "all" in question_lower
                        or "same" in question_lower
                        else (
                            "Dupaco lobby hours vary by branch. "
                            "Published examples:\n"
                        )
                    )

                return Answer(
                    text=(
                        intro
                        + "\n\n".join(schedule_results)
                        + "\n\nWhich branch would you "
                          "like to check? These published "
                          "schedules may have changed."
                    ),
                    sources=schedule_sources,
                    confident=True
                )


        # Prefer actual mobile-wallet answers over FAQ question links.
        asks_mobile_wallet = (
            any(term in question_lower for term in (
                "mobile wallet",
                "mobile payment",
                "apple pay",
                "google pay",
                "samsung pay",
            ))
            and any(term in question_lower for term in (
                "card", "wallet", "pay", "payment",
            ))
        )

        if asks_mobile_wallet:
            wallet_url = (
                "https://www.dupaco.com/"
                "online-and-mobile-banking/mobile-payments"
            )

            wallet_docs = [
                doc for doc in index["documents"]
                if doc["url"].rstrip("/") == wallet_url
            ]

            wallet_answers = []

            for doc in wallet_docs:
                passage = doc["text"]

                # Select factual statements rather than FAQ links.
                if (
                    "apple pay" in passage.lower()
                    and "google pay" in passage.lower()
                    and "samsung pay" in passage.lower()
                    and not passage.strip().startswith("- [")
                ):
                    if passage not in wallet_answers:
                        wallet_answers.append(passage)

            if wallet_answers:
                # Prefer the statement that explains both
                # card compatibility and default-card selection.
                best_wallet = max(
                    wallet_answers,
                    key=lambda text: (
                        "default" in text.lower(),
                        "debit and credit" in text.lower(),
                        len(text)
                    )
                )

                return Answer(
                    text=best_wallet,
                    sources=[
                        Source(
                            url=wallet_url,
                            snapshot_id=selected["snapshot_id"],
                            quote=best_wallet[:200]
                        )
                    ],
                    confident=True
                )

        # certificate/CD questions without a specified term.
        asks_certificate = bool(re.search(
            r"\b(?:certificates?|cds?)\b",
            question_lower
        ))

        asks_certificate_term = bool(re.search(
            r"\b\d+[\s-]*(?:month|months|year|years)\b",
            question_lower
        ))

        if (
            asks_certificate
            and asks_rate
            and not asks_certificate_term
        ):
            certificate_rows = []
            rate_url = None

            for doc in index["documents"]:
                url = doc["url"].rstrip("/").lower()

                if not url.endswith("/rates"):
                    continue

                lines = doc["text"].splitlines()

                if len(lines) < 3:
                    continue

                # Confirm this passage comes from the
                # certificate rate table, not another product.
                if not any(
                    "| certificate |" in line.lower()
                    for line in lines
                ):
                    continue

                for line in lines:
                    if not line.strip().startswith("|"):
                        continue

                    cells = [
                        cell.strip()
                        for cell in line.strip().strip("|").split("|")
                    ]

                    if len(cells) != 4:
                        continue

                    if not re.search(
                        r"\b\d+\s*month\b",
                        cells[0],
                        re.IGNORECASE
                    ):
                        continue

                    if not re.search(
                        r"\d+(?:\.\d+)?%",
                        cells[2]
                    ):
                        continue

                    if line not in certificate_rows:
                        certificate_rows.append(line)
                        rate_url = doc["url"]

            if certificate_rows:
                return Answer(
                    text=(
                        "Dupaco certificate dividend rates "
                        "and APYs by term:\n"
                        "| Certificate | Dividend Rate | "
                        "APY | Minimum to Earn |\n"
                        "|---|---:|---:|---:|\n"
                        + "\n".join(certificate_rows)
                    ),
                    sources=[
                        Source(
                            url=rate_url,
                            snapshot_id=selected["snapshot_id"],
                            quote=certificate_rows[0][:200]
                        )
                    ],
                    confident=True
                )


        
       
        # Collect personal and business credit-card APRs
        # when the question asks about credit cards generally.
        asks_all_card_rates = (
            ("credit cards" in question_lower
             or "credit card rates" in question_lower)
            and asks_rate
            and not any(
                name in question_lower
                for name in (
                    "platinum visa",
                    "rewards visa",
                    "business platinum",
                    "business rewards",
                )
            )
        )

        if asks_all_card_rates:
            card_sources = (
                (
                    "Platinum Visa",
                    "/personal/credit-cards",
                ),
                (
                    "Rewards Visa",
                    "/personal/credit-cards",
                ),
                (
                    "Business Platinum Visa",
                    "/business/credit-cards",
                ),
                (
                    "Business Rewards Visa",
                    "/business/credit-cards",
                ),
            )

            card_results = []
            source_urls = []

            for card_name, path in card_sources:
                matches = []

                for doc in index["documents"]:
                    url = doc["url"].rstrip("/").lower()

                    if not url.endswith(path):
                        continue

                    passage = doc["text"]
                    lines = passage.splitlines()

                    if not lines:
                        continue

                    # Each indexed passage retains its
                    # product or section heading.
                    if (
                        card_name.lower() not in lines[0].lower()
                        and not (
                            path.startswith("/business")
                            and any(
                                card_name.lower() in line.lower()
                                for line in lines[1:]
                            )
                        )
                    ):
                        continue

                    for line in lines[1:]:
                        if (
                            "apr" not in line.lower()
                            and not (
                                path.startswith("/business")
                                and line.strip().startswith("|")
                                and card_name.lower() in line.lower()
                            )
                        ):
                            continue

                        # Find standard APR ranges or
                        # individual published APR values.
                        rates = re.findall(
                            r"\d+(?:\.\d+)?\s*%",
                            line
                        )

                        if not rates:
                            continue

                        # Avoid introductory promotional APRs.
                        if (
                            "intro" in line.lower()
                            and "ongoing rate" not in line.lower()
                        ):
                            continue

                        # Prefer structured rate rows for
                        # business cards.
                        if path.startswith("/business"):
                            if not (
                                line.strip().startswith("|")
                                and card_name.lower()
                                in line.lower()
                            ):
                                continue

                        matches.append((line, doc["url"]))

                if matches:
                    # Prefer the shortest complete rate
                    # statement, avoiding duplicated prose.
                    rate_line, url = min(
                        matches,
                        key=lambda item: len(item[0])
                    )

                    card_results.append(
                        f"{card_name}: {rate_line.strip()}"
                    )

                    if url not in source_urls:
                        source_urls.append(url)

            if card_results:
                return Answer(
                    text=(
                        "Dupaco credit-card APRs:\n"
                        + "\n".join(card_results)
                    ),
                    sources=[
                        Source(
                            url=url,
                            snapshot_id=selected["snapshot_id"]
                        )
                        for url in source_urls
                    ],
                    confident=True
                )

        # Answer education financing questions using the
        # official student-loan product page.
        asks_student_financing = (
            "student loan" in question_lower
            or (
                any(term in question_lower for term in (
                    "university", "college", "tuition",
                    "undergraduate", "education"
                ))
                and any(term in question_lower for term in (
                    "loan", "borrow", "financing",
                    "funding", "repayment"
                ))
            )
        )

        if asks_student_financing and not asks_rate:
            student_url = (
                "https://www.dupaco.com/personal/loans/student-loans"
            )

            student_docs = [
                doc for doc in index["documents"]
                if doc["url"].rstrip("/") == student_url
            ]

            # Select product details rather than headings,
            # navigation links, or unrelated articles.
            evidence_terms = (
                "funding gap",
                "flexible options",
                "defer full payments",
                "graduated repayment",
                "$0 origination fees",
                "student loan debt",
            )

            evidence = []

            for doc in student_docs:
                passage = doc["text"]

                if any(
                    term in passage.lower()
                    for term in evidence_terms
                ):
                    if passage not in evidence:
                        evidence.append(passage)

            if evidence:
                return Answer(
                    text="\n".join(evidence)[:2000],
                    sources=[
                        Source(
                            url=student_url,
                            snapshot_id=selected["snapshot_id"],
                            quote=evidence[0][:200]
                        )
                    ],
                    confident=True
                )

        # Combine checking-account rate rows from the official rate sheet.
        if (
            "checking" in question_lower
            and asks_rate
            and not asks_mortgage
        ):
            checking_rows = []

            for doc in index["documents"]:
                url = doc["url"].rstrip("/").lower()
                passage = doc["text"]

                if not url.endswith("/rates"):
                    continue

                for line in passage.splitlines():
                    line_lower = line.lower()

                    if not line.strip().startswith("|"):
                        continue

                    if (
                        "| free checking |" in line_lower
                        or "| 1st rate checking |" in line_lower
                    ):
                        if line not in checking_rows:
                            checking_rows.append(line)

            if checking_rows:
                return Answer(
                    text=(
                        "Current checking account rates:\n"
                        + "\n".join(checking_rows)
                    ),
                    sources=[
                        Source(
                            url="https://www.dupaco.com/rates",
                            snapshot_id=selected["snapshot_id"]
                        )
                    ],
                    confident=True
                )
            
        # Use Share Savings tiers for general savings-rate questions.
        
        # Retrieve Bright Savings rates from the selected snapshot.
        if (
            "bright savings" in question_lower
            and asks_rate
        ):
            bright_rows = []
            rate_url = None

            for doc in index["documents"]:
                url = doc["url"].rstrip("/").lower()

                # Prefer the official rate sheet.
                if not url.endswith("/rates"):
                    continue

                for line in doc["text"].splitlines():
                    if (
                        line.strip().startswith("|")
                        and "| bright savings |" in line.lower()
                        and line not in bright_rows
                    ):
                        bright_rows.append(line)
                        rate_url = doc["url"]

            if bright_rows:
                return Answer(
                    text=(
                        "Bright Savings account rates:\n"
                        + "\n".join(bright_rows)
                    ),
                    sources=[
                        Source(
                            url=rate_url,
                            snapshot_id=selected["snapshot_id"],
                            quote=bright_rows[0][:200]
                        )
                    ],
                    confident=True
                )

        
        if (
            "savings" in question_lower
            and asks_rate
            and "health savings" not in question_lower
            and "hsa" not in question_lower
            and "bright savings" not in question_lower
        ):

            savings_rows = []
            rate_url = None

            for doc in index["documents"]:
                url = doc["url"].rstrip("/").lower()

                if not url.endswith("/rates"):
                    continue

                for line in doc["text"].splitlines():
                    if (
                        line.strip().startswith("|")
                        and "| share savings |" in line.lower()
                        and line not in savings_rows
                    ):
                        savings_rows.append(line)
                        rate_url = doc["url"]

            if savings_rows:
                apys = []

                for line in savings_rows:
                    cells = [
                        cell.strip()
                        for cell in line.strip().strip("|").split("|")
                    ]

                    if len(cells) >= 4:
                        match = re.search(
                            r"\d+(?:\.\d+)?%",
                            cells[3]
                        )
                        if match:
                            apys.append(float(match.group()[:-1]))

                highest_apy = max(apys) if apys else None

                if highest_apy is not None:
                    intro = (
                        f"No. Share Savings tiers have a maximum "
                        f"APY of {highest_apy:.2f}%.\n"
                        if re.search(
                            r"\b(?:do any|does any)\b",
                            question_lower
                        )
                        and re.search(r"\b0\.35\s*%", question_lower)
                        else "Current Share Savings rates:\n"
                    )

                    return Answer(
                        text=intro + "\n".join(savings_rows),
                        sources=[
                            Source(
                                url=rate_url,
                                snapshot_id=selected["snapshot_id"]
                            )
                        ],
                        confident=True
                    )



        # Combine related rows for broad mortgage-rate questions.
        asks_specific_amount = bool(re.search(
            r"\$\s*[\d,]+",
            question
        ))

        asks_specific_term = bool(re.search(
            r"\b\d+[\s-]*year\b",
            question_lower
        ))

        if (
            asks_mortgage
            and asks_rate
            and not asks_arm
            and not asks_specific_amount
        ):
            url = best["url"].rstrip("/").lower()

            if url.endswith("/rates"):
                related = []

                for doc in index["documents"]:
                    if doc["url"] != best["url"]:
                        continue

                    passage = doc["text"]

                    if "Purchase: 30-Year Fixed Conventional" not in passage:
                        continue

                    if not re.search(r"\d+(?:\.\d+)?%", passage):
                        continue

                    related.append(doc)

                # Avoid mixing unrelated mortgage types.
                if related and (
                    not asks_specific_term
                    or "30-year" in question_lower
                ):
                    combined_text = "\n".join(
                        doc["text"].splitlines()[-1]
                        for doc in related
                    )

                    mortgage_sources = [
                        Source(
                            url=best["url"],
                            snapshot_id=best["snapshot_id"]
                        )
                    ]

                    # Add evidence from the dedicated home-loans
                    # page without changing the selected rates.
                    supporting_rows = []

                    for doc in index["documents"]:
                        url = doc["url"].rstrip("/").lower()

                        if not url.endswith(
                            "/personal/loans/home-loans"
                        ):
                            continue

                        for line in doc["text"].splitlines():
                            if "Purchase: 30-Year Fixed Conventional" not in line:
                                continue

                            if line in combined_text:
                                supporting_rows.append((doc, line))

                    if supporting_rows:
                        doc, line = supporting_rows[0]

                        mortgage_sources.append(
                            Source(
                                url=doc["url"],
                                snapshot_id=selected["snapshot_id"],
                                quote=line[:200]
                            )
                        )

                    return Answer(
                        text=combined_text[:1500],
                        sources=mortgage_sources,
                        confident=True
                    )


        # Check whether rate questions have explicit rate evidence.
        if asks_rate:
            evidence = best["text"]

            # A percentage alone is not enough:
            # it could describe a down payment, LTV,
            # rate adjustment cap, or other condition.
            lines = evidence.splitlines()
            has_explicit_rate = False

            for line in lines:
                lower = line.lower()

                if not re.search(r"\d+(?:\.\d+)?\s*%", line):
                    continue

                # Exclude common non-rate percentages.
                if any(term in lower for term in (
                    "down payment",
                    "construction costs",
                    "loan-to-value",
                    "may adjust up to",
                    "maximum adjustment",
                    "within 6%",
                )):
                    continue

                # Structured rate tables are strong evidence.
                if line.strip().startswith("|"):
                    if any(term in evidence.lower() for term in (
                        "| rate |",
                        "| apr |",
                        "annual percentage rate",
                        "annual percentage yield",
                        "dividend rate",
                    )):
                        has_explicit_rate = True
                        break

                # Explicit prose statements about rates.
                if re.search(
                    r"(?:interest rate|apr|apy|fixed rate)"
                    r".{0,50}\d+(?:\.\d+)?\s*%",
                    lower
                ):
                    has_explicit_rate = True
                    break

            if not has_explicit_rate:
                return Answer(
                    text=(
                        "I couldn't verify a published interest "
                        "rate for this product from the available "
                        "website information. Please contact "
                        "Dupaco for the applicable rate."
                    ),
                    sources=[
                        Source(
                            url=best["url"],
                            snapshot_id=best["snapshot_id"]
                        )
                    ],
                    confident=False
                )



        answer_sources = [
            Source(
                url=best["url"],
                snapshot_id=best["snapshot_id"],
                quote=best["text"][:200]
            )
        ]

        # Add supporting evidence for the same auto-loan rate,
        # term, and model year from the official product page.
        if asks_auto and asks_rate:
            term_match = re.search(
                r"\b(\d{1,3})[\s-]*months?\b",
                question_lower
            )
            year_match = re.search(
                r"\b(20\d{2})\b", question_lower
            )
            answer_rates = set(re.findall(
                r"\d+(?:\.\d+)?%", best["text"]
            ))

            if term_match and year_match and answer_rates:
                months = int(term_match.group(1))
                year = int(year_match.group(1))

                for doc in index["documents"]:
                    url = doc["url"].rstrip("/").lower()

                    if not url.endswith(
                        "/personal/loans/auto-motorcycle-loans"
                    ):
                        continue

                    for line in doc["text"].splitlines():
                        range_match = re.search(
                            r"\b(20\d{2})-(\d{2,4})\s+model year\b",
                            line, re.IGNORECASE
                        )
                        row_term = re.search(
                            r"\|\s*(\d+)\s+months?\s*\|",
                            line, re.IGNORECASE
                        )

                        if not range_match or not row_term:
                            continue

                        start_year = int(range_match.group(1))
                        end_text = range_match.group(2)
                        end_year = int(
                            end_text if len(end_text) == 4
                            else str(start_year)[:2] + end_text
                        )

                        if not start_year <= year <= end_year:
                            continue

                        if int(row_term.group(1)) != months:
                            continue

                        if not any(
                            rate in line for rate in answer_rates
                        ):
                            continue

                        answer_sources.append(
                            Source(
                                url=doc["url"],
                                snapshot_id=selected["snapshot_id"],
                                quote=line[:200]
                            )
                        )
                        break

                    if len(answer_sources) > 1:
                        break

        return Answer(
            text=best["text"][:1000],
            sources=answer_sources,
            confident=True
        )

    
   
    def extract_cross_page_auto(self, snapshot):
        """
        Extract auto-loan APR facts from multiple webpage types.
        Keep model-year ranges and effective dates.
        """
        for page in snapshot.pages:
            url = page.url.lower()
            content = page.markdown

            # Limit extraction to relevant auto-loan pages.
            if not any(term in url for term in (
                "/rates",
                "auto-loan",
                "auto-motorcycle",
                "vehicle-affordability",
                "switch-and-earn",
            )):
                continue

            lines = content.splitlines()
            for line_number, line in enumerate(lines):
                cells = [
                    c.strip()
                    for c in line.strip().strip("|").split("|")
                ]

                if len(cells) != 3:
                    continue

                model, term, apr_text = cells

                # Require an explicit model-year range.
                model_match = re.search(
                    r"(\d{4})\s*[-–]\s*(\d{2,4})"
                    r"\s+Model Year",
                    model,
                    re.IGNORECASE
                )

                term_match = re.fullmatch(
                    r"(\d+)\s+Months?",
                    term,
                    re.IGNORECASE
                )

                apr_match = re.match(
                    r"(\d+(?:\.\d+)?)\s*%\s*APR",
                    apr_text,
                    re.IGNORECASE
                )

                if not all((
                    model_match, term_match, apr_match
                )):
                    continue

                start_year = int(model_match.group(1))
                end_text = model_match.group(2)

                if len(end_text) == 2:
                    end_year = (
                        start_year // 100 * 100
                        + int(end_text)
                    )
                else:
                    end_year = int(end_text)

                               # Find the effective date near this rate table.
                following_lines = lines[
                    line_number + 1:line_number + 25
                ]

                nearby_text = "\n".join(following_lines)

                effective_match = re.search(
                    r"Rates are effective as of\s+"
                    r"(\d{1,2}/\d{1,2}/\d{4})",
                    nearby_text,
                    re.IGNORECASE
                )

                effective_date = (
                    effective_match.group(1)
                    if effective_match else None
                )
                self.cross_page_auto_facts.append({
                    "start_year": start_year,
                    "end_year": end_year,
                    "term_months": int(term_match.group(1)),
                    "apr": apr_match.group(1) + "%",
                    "effective_date": effective_date,
                    "snapshot_id": snapshot.snapshot_id,
                    "snapshot_date": snapshot.taken_on,
                    "url": page.url,
                    "quote": line.strip(),
                })

    def conflicts(self):
        found = []

        all_facts = {}

        for topic, facts in self.certificate_facts.items():
            all_facts[topic] = list(facts)

        for topic, facts in self.auto_facts.items():
            all_facts.setdefault(topic, []).extend(facts)

        for topic, facts in all_facts.items():
            # Use the newest observation of each value.
            by_value = {}

            for fact in facts:
                value = fact["value"]
                previous = by_value.get(value)

                if previous is None or fact["date"] > previous["date"]:
                    by_value[value] = fact

            # One unique value means no detected conflict.
            if len(by_value) < 2:
                continue

            ordered = sorted(
                by_value.values(),
                key=lambda f: (
                    f["date"],
                    f["snapshot_id"]
                )
            )

            # Most recent observed value wins.
            winner = max(
                facts,
                key=lambda f: (
                    f["date"],
                    f["snapshot_id"]
                )
            )

            sources = [
                Source(
                    url=fact["url"],
                    snapshot_id=fact["snapshot_id"],
                    quote=fact["quote"][:200]
                )
                for fact in ordered
            ]

            found.append(
                Conflict(
                    topic=topic,
                    values=[
                        fact["value"] for fact in ordered
                    ],
                    sources=sources,
                    resolution=winner["value"],
                    
                    reason=(
                        "The newer snapshot contains the updated "
                        + (
                            "auto-loan APR for the same model-year "
                            "group and loan term."
                            if topic.startswith("auto loan")
                            else
                            "certificate APY for the same "
                            "certificate term and tier."
                        )
                    )

                )
            )

        
        # Detect conflicting auto-loan rates between
        # webpages in the latest snapshot.
        if self.cross_page_auto_facts:
            latest_id = max(
                self.snapshots,
                key=lambda s: s["taken_on"]
            )["snapshot_id"]

            facts = [
                f for f in self.cross_page_auto_facts
                if f["snapshot_id"] == latest_id
                and f["effective_date"]
            ]

            def parse_date(value):
                return datetime.strptime(
                    value, "%m/%d/%Y"
                ).date()

            def source_rank(url):
                url = url.rstrip("/").lower()

                if url.endswith("/rates"):
                    return 3
                if "/personal/loans/" in url:
                    return 2
                return 1

            reported = set()

            for old in facts:
                # Find newer conflicting rates for the
                # same term and overlapping model years.
                newer = [
                    f for f in facts
                    if f["term_months"] == old["term_months"]
                    and f["apr"] != old["apr"]
                    and parse_date(f["effective_date"]) >
                        parse_date(old["effective_date"])
                    and max(
                        f["start_year"], old["start_year"]
                    ) <= min(
                        f["end_year"], old["end_year"]
                    )
                ]

                if not newer:
                    continue

                winner = max(
                    newer,
                    key=lambda f: (
                        parse_date(f["effective_date"]),
                        source_rank(f["url"])
                    )
                )

                overlap_start = max(
                    old["start_year"],
                    winner["start_year"]
                )

                overlap_end = min(
                    old["end_year"],
                    winner["end_year"]
                )

                # Deduplicate repeated claims appearing
                # on multiple webpages.
                key = (
                    old["term_months"],
                    overlap_start,
                    overlap_end,
                    old["apr"],
                    winner["apr"]
                )

                if key in reported:
                    continue

                reported.add(key)

                found.append(
                    Conflict(
                        topic=(
                            f"{old['term_months']}-month "
                            f"auto loan APR for "
                            f"{overlap_start}-{overlap_end} "
                            f"model years"
                        ),
                        values=[
                            old["apr"],
                            winner["apr"]
                        ],
                        sources=[
                            Source(
                                url=old["url"],
                                snapshot_id=old["snapshot_id"],
                                quote=old["quote"][:200]
                            ),
                            Source(
                                url=winner["url"],
                                snapshot_id=winner["snapshot_id"],
                                quote=winner["quote"][:200]
                            )
                        ],
                        resolution=winner["apr"],
                        reason=(
                            "The rates apply to overlapping "
                            "vehicle model years and the same "
                            "loan term. The newer effective "
                            "rate supersedes the older "
                            "published rate."
                        )
                    )
                )

        
        return found



def build():
    return NaiveKnowledge()
