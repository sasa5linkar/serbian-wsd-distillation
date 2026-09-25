from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

DEV_GOLD_FILE = "sr-elexis-WSD_0001_0120-gold.tsv"
FINAL_GOLD_FILES = (
    "sr-elexis-WSD_0301-0400-gold.tsv",
    "sr-elexis-WSD_0401-0500-gold.tsv",
    "sr-elexis-WSD_0501-0600-gold.tsv",
)
BLANK_VALUES = {"", "_", "*", "0"}
SUFFIX_RE = re.compile(r"\[\d+\]")
ENG_URL_ID_RE = re.compile(r"^se(ENG30)(\d{8})([a-z])$")
LLM_URL_ID_RE = re.compile(r"^se(LLM)(\d+)$")
CVMWE_URL_ID_RE = re.compile(r"^se(CVMWE)(\d+)$")
CV_URL_ID_RE = re.compile(r"^se(CV\d+)$")
DRJ_URL_ID_RE = re.compile(r"^seDRJ0?(\d+)_([^_]+)$")
DRS_URL_ID_RE = re.compile(r"^seDRS0?(\d+)_([^_]+)$")
RSSJ_URL_ID_RE = re.compile(r"^seRSSJ(\d+)_([^_]+)$")
CVDF_URL_ID_RE = re.compile(r"^seCVDf(\d+)_([^_]+)$")


@dataclass(slots=True)
class WebAnnoToken:
    sentence_index: int
    token_id: str
    start: int
    end: int
    text: str
    pos: str = ""
    upos: str = ""
    ne_id: str = ""
    ne_type: str = ""
    lemma: str = ""
    mwe_id: str = ""
    mwe_lemma: str = ""
    mwe_type: str = ""
    comment: str = ""
    explanation: str = ""
    sense_id: str = ""
    number_of_senses: str = ""
    origin: str = ""
    candidate_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WebAnnoSentence:
    text: str
    sentence_index: int
    tokens: list[WebAnnoToken] = field(default_factory=list)
    source_file: str = ""


def strip_webanno_suffix(value: object) -> str:
    if value is None:
        return ""
    return SUFFIX_RE.sub("", str(value)).strip()


def is_blank(value: object) -> bool:
    return strip_webanno_suffix(value) in BLANK_VALUES


def normalize_sense_id(value: object) -> str:
    text = strip_webanno_suffix(value)
    if not text or text in BLANK_VALUES:
        return ""
    if text.startswith("http:") or text.startswith("https:"):
        text = text.rstrip("/").rsplit("/", 1)[-1]

    eng_match = ENG_URL_ID_RE.match(text)
    if eng_match:
        prefix, digits, pos = eng_match.groups()
        return f"{prefix}-{digits}-{pos}"

    llm_match = LLM_URL_ID_RE.match(text)
    if llm_match:
        prefix, number = llm_match.groups()
        return f"{prefix}-{number}"

    cvmwe_match = CVMWE_URL_ID_RE.match(text)
    if cvmwe_match:
        prefix, number = cvmwe_match.groups()
        return f"{prefix}-{number}"

    cv_match = CV_URL_ID_RE.match(text)
    if cv_match:
        return cv_match.group(1)

    drj_match = DRJ_URL_ID_RE.match(text)
    if drj_match:
        digits, sense = drj_match.groups()
        return f"DRJ0-{int(digits)}/{sense}"

    drs_match = DRS_URL_ID_RE.match(text)
    if drs_match:
        digits, sense = drs_match.groups()
        if sense.isdigit() and len(sense) == 2:
            sense = f"{sense[0]}.{sense[1]}"
        return f"DRS0-{int(digits)}/{sense}"

    rssj_match = RSSJ_URL_ID_RE.match(text)
    if rssj_match:
        digits, sense = rssj_match.groups()
        return f"RSSJ-{int(digits)}/{sense}"

    cvdf_match = CVDF_URL_ID_RE.match(text)
    if cvdf_match:
        digits, sense = cvdf_match.groups()
        return f"CVDf-{int(digits)}/{sense}"

    return text


def parse_candidate_ids(value: object) -> list[str]:
    text = strip_webanno_suffix(value)
    if not text or text in BLANK_VALUES:
        return []
    normalized_delimiters = text.replace("\\;", ";").replace("|", ";")
    candidates: list[str] = []
    seen: set[str] = set()
    for part in normalized_delimiters.split(";"):
        candidate = normalize_sense_id(part)
        if candidate and candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)
    return candidates


def mark_span(sentence_text: str, spans: Iterable[tuple[int, int]], start_mark: str = "**", end_mark: str = "**") -> str:
    pieces: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        pieces.append(sentence_text[cursor:start])
        pieces.append(f"{start_mark}{sentence_text[start:end]}{end_mark}")
        cursor = end
    pieces.append(sentence_text[cursor:])
    return "".join(pieces)


def gold_files_for_split(gold_dir: Path | str, split: Literal["dev", "final", "all"]) -> list[Path]:
    directory = Path(gold_dir)
    if split == "dev":
        names = (DEV_GOLD_FILE,)
    elif split == "final":
        names = FINAL_GOLD_FILES
    elif split == "all":
        names = (DEV_GOLD_FILE, *FINAL_GOLD_FILES)
    else:
        raise ValueError(f"Unknown gold split: {split}")
    return [directory / name for name in names if (directory / name).exists()]


def iter_tsv_files(path: Path | str) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted(root.glob("*.tsv"))


def parse_webanno_tsv(path: Path | str) -> list[WebAnnoSentence]:
    path = Path(path)
    sentences: list[WebAnnoSentence] = []
    current: WebAnnoSentence | None = None
    inferred_sentence_index = 0

    def flush() -> None:
        nonlocal current
        if current is not None:
            sentences.append(current)
            current = None

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        if raw_line.startswith("#Text="):
            flush()
            inferred_sentence_index += 1
            current = WebAnnoSentence(
                text=raw_line[len("#Text=") :],
                sentence_index=inferred_sentence_index,
                source_file=str(path),
            )
            continue
        if not raw_line:
            continue
        if raw_line.startswith("#"):
            continue
        if current is None:
            continue

        columns = raw_line.split("\t")
        if len(columns) < 11:
            continue
        start_text, end_text = columns[1].split("-", 1)
        wsd = columns[11:]
        if len(wsd) >= 6:
            comment, explanation, kbid, number, origin, possible = wsd[:6]
        elif len(wsd) >= 5:
            comment = ""
            explanation, kbid, number, origin, possible = wsd[:5]
        else:
            comment = explanation = kbid = number = origin = possible = ""

        current.tokens.append(
            WebAnnoToken(
                sentence_index=current.sentence_index,
                token_id=columns[0],
                start=int(start_text),
                end=int(end_text),
                text=columns[2],
                pos=strip_webanno_suffix(columns[3]),
                upos=strip_webanno_suffix(columns[4]),
                ne_id=strip_webanno_suffix(columns[5]),
                ne_type=strip_webanno_suffix(columns[6]),
                lemma=strip_webanno_suffix(columns[7]),
                mwe_id=strip_webanno_suffix(columns[8]),
                mwe_lemma=strip_webanno_suffix(columns[9]),
                mwe_type=strip_webanno_suffix(columns[10]),
                comment=strip_webanno_suffix(comment),
                explanation=strip_webanno_suffix(explanation),
                sense_id=normalize_sense_id(kbid),
                number_of_senses=strip_webanno_suffix(number),
                origin=strip_webanno_suffix(origin),
                candidate_ids=parse_candidate_ids(possible),
            )
        )

    flush()
    return sentences
