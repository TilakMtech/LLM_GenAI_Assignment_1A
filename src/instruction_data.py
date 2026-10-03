"""Part B1 - build the instruction dataset from the cleaned domain corpus.

Two methods:
  * heuristic (default, offline, fully reproducible) - parse every policy into
    numbered sections and clauses and turn them into grounded Q&A pairs:
      - section_qa   : "What does the <policy> say about <section>?"   -> section text
      - section_sum  : "Summarise the <section> provisions ..."        -> first sentences
      - clause_rule  : "What does the <policy> require regarding <subject>?" -> clause
      - clause_cloze : "Complete this rule from the <policy>: '<first half>...'" -> clause
    Every response is copied from the policy text, so answers are grounded.
  * llm - synthetic generation with an external LLM (Anthropic or OpenAI API),
    using LLM_PROMPT_TEMPLATE below (recorded verbatim for the report).

The pairs are split 80/20 by *section group* so near-identical questions about
the same clause never land in both train and eval (no leakage).
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

from src import config

LLM_PROMPT_TEMPLATE = (
    "Read the text below and generate {n} instruction-response pairs in JSON format "
    "based ONLY on this text. Each entry must have instruction and response keys.\n"
    "The instruction must be a question an employee or HR executive would realistically ask; "
    "the response must answer it using only facts stated in the text (quote numbers, limits "
    "and conditions exactly). Return a JSON list only, with no commentary.\n\n"
    "Document: {title}\n---\n{chunk}\n---"
)

HEADING = re.compile(
    r"^(?:(?:section|rule|chapter|clause)\s+)?(?P<num>\d{1,2}(?:\.\d{1,2}){0,2})[.)]?\s+"
    r"(?P<title>[A-Z][A-Za-z0-9 ,&/()'’\-]{2,80}?)\s*(?:[.:]?\s*[—–-]{1,2}\s*(?P<rest>.*))?$",
    re.I)
CAPS_HEADING = re.compile(r"^[A-Z][A-Z0-9 &/,'()\-]{3,60}:?$")
ITEM_START = re.compile(r"^(?:\(?[a-z]\)|\(?[ivx]{1,4}\)|[ivx]{1,4}\.|\(\d{1,2}\)|[o•▪\-\*]\s)", re.I)
MODAL = re.compile(
    r"^(?P<subj>[A-Z][^.;:]{2,70}?)\s+(?P<verb>shall|must|should|will be|will|is entitled|are entitled|"
    r"is eligible|are eligible|can not|cannot|can|may|is not|are not|is required|are required)\b",
    re.I)
GENERIC_SUBJECTS = {"leave", "resource", "resources", "employee", "employees", "policy", "process",
                    "purpose", "it", "they", "he", "she", "this", "these", "staff", "company",
                    "organisation", "organization", "management", "he/she", "member", "members"}
WORDS = lambda s: len(s.split())  # noqa: E731


def document_title(path: Path, text: str) -> str:
    first = next((l.strip() for l in text.splitlines() if l.strip()), "")
    if re.search(r"polic|rule|manual|code|guideline|program|procedure", first, re.I) and WORDS(first) <= 12:
        title = first
    else:
        title = path.stem.replace("_", " ")
    title = re.sub(r"\s+", " ", title).strip(" :-")
    if title.isupper() or title.islower():
        return title.title()
    # "MyGov LEAVE POLICY" -> "MyGov Leave Policy" (keep short acronyms such as CCS)
    return re.sub(r"\b[A-Z]{4,}\b", lambda m: m.group(0).capitalize(), title)


def split_sections(text: str):
    """Return [(heading, body_lines)] using numbered or ALL-CAPS headings."""
    sections, heading, body = [], "Overview", []
    for raw in text.splitlines():
        line = raw.strip()
        match = HEADING.match(line)
        is_caps = (CAPS_HEADING.match(line) and WORDS(line) <= 8
                   and sum(ch.isalpha() for ch in line) >= 0.7 * len(line.replace(" ", "")))
        if (match and WORDS(match.group("title")) <= 10) or is_caps:
            if body:
                sections.append((heading, body))
            if match:
                heading = match.group("title").strip(" .:-")
                body = [match.group("rest")] if match.group("rest") else []
            else:
                heading, body = line.strip(":").title(), []
        elif line:
            body.append(line)
    if body:
        sections.append((heading, body))
    return sections


def split_items(lines):
    """Re-join wrapped PDF lines into list items / sentences."""
    items, current = [], ""
    for line in lines:
        if ITEM_START.match(line) and current:
            items.append(current)
            current = line
        else:
            current = f"{current} {line}".strip()
    if current:
        items.append(current)
    clauses = []
    for item in items:
        item = ITEM_START.sub("", item).strip()
        # long paragraphs -> sentences
        parts = re.split(r"(?<=[.;])\s+(?=[A-Z])", item) if WORDS(item) > 80 else [item]
        clauses.extend(p.strip() for p in parts if p.strip())
    return clauses


def truncate_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    cut = " ".join(words[:limit])
    end = max(cut.rfind(". "), cut.rfind("; "))
    return cut[:end + 1] if end > len(cut) // 2 else cut + " ..."


def first_sentences(text: str, n: int = 2) -> str:
    sentences = re.split(r"(?<=[.])\s+(?=[A-Z])", text)
    return " ".join(sentences[:n])


def clean_subject(subject: str) -> str:
    subject = re.sub(r"^(the|all|any|every|an?)\s+", "", subject.strip(), flags=re.I)
    return subject if subject[:2].isupper() else subject[0].lower() + subject[1:]


def heuristic_pairs(path: Path, max_per_doc: int = 150, seed: int = config.SEED):
    text = path.read_text(encoding="utf-8")
    title = document_title(path, text)
    pairs = []
    for s_idx, (heading, lines) in enumerate(split_sections(text)):
        group = f"{path.stem}::{s_idx}"
        body = " ".join(lines)
        body = re.sub(r"\s+", " ", body).strip()
        if WORDS(body) >= 20 and heading != "Overview":
            pairs.append({"instruction": f"What does the {title} say about {heading.lower()}?",
                          "response": truncate_words(body, 180), "type": "section_qa"})
            if WORDS(body) >= 60:
                pairs.append({"instruction": f"Summarise the '{heading}' provisions of the {title}.",
                              "response": first_sentences(body, 2), "type": "section_sum"})
        for clause in split_items(lines):
            n = WORDS(clause)
            if not 8 <= n <= 90:
                continue
            modal = MODAL.match(clause)
            subj = modal.group("subj").strip(" ,") if modal else ""
            if (modal and 1 <= WORDS(subj) <= 7 and not re.search(r"\d", subj)
                    and not re.match(r"(in|on|at|for|other|if|where|when|during|after|before|upon|as|such)\b", subj, re.I)):
                subj = clean_subject(subj)
                if subj.lower() in GENERIC_SUBJECTS:
                    question = f"Under the {title}, what is the rule for {subj} on {heading.lower()}?"
                else:
                    question = f"What does the {title} require regarding {subj}?"
                pairs.append({"instruction": question, "response": clause, "type": "clause_rule"})
            elif re.search(r"\d|eligible|entitle|not |only|within|maximum|minimum", clause, re.I) and n >= 12:
                half = " ".join(clause.split()[: max(5, n // 2)])
                pairs.append({"instruction": f"Complete this rule from the {title} "
                                             f"(section '{heading}'): \"{half} ...\"",
                              "response": clause, "type": "clause_cloze"})
        for pair in pairs:
            pair.setdefault("group", group)
            pair.setdefault("section", heading)
    for pair in pairs:
        pair["source"] = path.name
        pair["document_title"] = title
    if len(pairs) > max_per_doc:  # keep long HR manuals from dominating the mix
        random.Random(seed).shuffle(pairs)
        pairs = pairs[:max_per_doc]
    return pairs


def llm_pairs(path: Path, n_per_chunk: int = 10, chunk_words: int = 600):
    """Synthetic pairs from an external LLM (needs ANTHROPIC_API_KEY or OPENAI_API_KEY)."""
    text = path.read_text(encoding="utf-8")
    title = document_title(path, text)
    words = text.split()
    pairs = []
    for c, start in enumerate(range(0, len(words), chunk_words)):
        chunk = " ".join(words[start:start + chunk_words])
        prompt = LLM_PROMPT_TEMPLATE.format(n=n_per_chunk, title=title, chunk=chunk)
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            reply = anthropic.Anthropic().messages.create(
                model=os.environ.get("LLM_MODEL", "claude-sonnet-4-5"), max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]).content[0].text
        elif os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI
            reply = OpenAI().chat.completions.create(
                model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}]).choices[0].message.content
        else:
            raise RuntimeError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY for --method llm")
        match = re.search(r"\[.*\]", reply, re.S)
        for item in json.loads(match.group(0)) if match else []:
            if item.get("instruction") and item.get("response"):
                pairs.append({"instruction": item["instruction"].strip(),
                              "response": item["response"].strip(), "type": "llm_synthetic",
                              "group": f"{path.stem}::{c}", "section": f"chunk {c}",
                              "source": path.name, "document_title": title})
    return pairs


def grouped_split(pairs, train_fraction=config.INSTRUCTION_TRAIN_FRACTION, seed=config.SEED):
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair["group"]].append(pair)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    train, evaluation, target = [], [], train_fraction * len(pairs)
    for key in keys:
        (train if len(train) < target else evaluation).extend(groups[key])
    return train, evaluation


def build_instruction_dataset(method: str = "heuristic", corpus_dir: Path = config.DOMAIN_CORPUS_DIR,
                              output_dir: Path = config.INSTRUCTION_DIR):
    pairs, seen = [], set()
    for path in sorted(corpus_dir.glob("*.txt")):
        for pair in (llm_pairs(path) if method == "llm" else heuristic_pairs(path)):
            key = re.sub(r"\W+", " ", pair["instruction"].lower()).strip()
            if key in seen or not 3 <= WORDS(pair["response"]) <= 250:
                continue
            seen.add(key)
            pair["method"] = method
            pairs.append(pair)
    if len(pairs) < 10:
        raise ValueError(f"Only {len(pairs)} pairs generated - check the corpus.")
    train, evaluation = grouped_split(pairs)
    output_dir.mkdir(parents=True, exist_ok=True)

    def dump(rows, path):
        with path.open("w", encoding="utf-8") as handle:
            for i, row in enumerate(rows):
                handle.write(json.dumps({"id": i, **row}, ensure_ascii=False) + "\n")

    dump(pairs, output_dir / "instruction_dataset.jsonl")
    dump(train, output_dir / "instruction_train.jsonl")
    dump(evaluation, output_dir / "instruction_eval.jsonl")
    by_type = defaultdict(int)
    for pair in pairs:
        by_type[pair["type"]] += 1
    stats = {"method": method, "total_pairs": len(pairs), "train_pairs": len(train),
             "eval_pairs": len(evaluation),
             "train_fraction": round(len(train) / len(pairs), 3),
             "documents": len({p["source"] for p in pairs}),
             "pairs_by_type": dict(by_type),
             "split_unit": "section group (no clause appears in both splits)",
             "llm_prompt_template": LLM_PROMPT_TEMPLATE if method == "llm" else None}
    (config.REPORT_DIR / "instruction_dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["heuristic", "llm"], default="heuristic")
    build_instruction_dataset(parser.parse_args().method)
