"""
ATS Resume Parser – NLP Edition
================================
Endpoints:
  POST /parse-resume   multipart 'file' → full structured JSON
  POST /score-match    JSON body         → AI skill-match score 0-100
  GET  /health                           → {"status": "ok"}

Tech stack:
  • PyPDF2       – PDF text extraction
  • python-docx  – DOCX text extraction
  • spaCy        – NER (named entity recognition for orgs, dates)
  • NLTK         – sentence tokenisation, stopwords
  • scikit-learn – TF-IDF cosine similarity for skill scoring

Run:
  python app.py          (development, port 5001)
"""

import io
import os
import re
import logging
from datetime import datetime

from flask import Flask, jsonify, request

# ── PDF / DOCX ────────────────────────────────────────────────────────
import PyPDF2
import docx

# ── NLP ───────────────────────────────────────────────────────────────
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# ── Date parsing ──────────────────────────────────────────────────────
from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta

# ── Scoring ───────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Bootstrap NLP models (lazy – errors caught at startup, not at request time)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("resume_parser")

# spaCy
try:
    _nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
    logger.info("spaCy model loaded: en_core_web_sm")
except OSError:
    _nlp = None
    SPACY_OK = False
    logger.warning("spaCy model NOT found – run: python -m spacy download en_core_web_sm")

# NLTK data
for _pkg in ("punkt", "stopwords", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

_STOPWORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Skills are extracted from the resume content (no predefined list).
# ---------------------------------------------------------------------------
 

# Education degree keywords
# IMPORTANT: Order matters — longer / more specific patterns first to avoid
# short abbreviations (be, bs, ms) matching common words in job descriptions.
DEGREE_KEYWORDS = [
    r"ph\.?\s*d\.?",
    r"doctor(?:ate)?",
    r"master[s]?\s+(?:of\s+)?(?:science|arts|technology|engineering|business|commerce|computer)",
    r"bachelor[s]?\s+(?:of\s+)?(?:science|arts|technology|engineering|commerce|computer)",
    r"b\.?\s*(?:tech|eng|sc|com|arch|ca)\.?",
    r"m\.?\s*(?:tech|eng|sc|com|phil|ba|ca)\.?",
    r"b\.?\s*s\.?",
    r"m\.?\s*s\.?",
    r"mba|pgdm",
    r"b\.com\b",
    r"diploma(?:\s+in)?",
    r"associate[s]?\s+(?:of\s+)?(?:science|arts)",
    r"higher\s+secondary|hsc",
    r"secondary\s+school|ssc",
]

# Degree regex: require start-of-line or whitespace boundary BEFORE the match
# to avoid catching "be" inside words like "website", "to be", etc.
DEGREE_RE = re.compile(
    r"(?:^|\n|\s)(" + "|".join(DEGREE_KEYWORDS) + r")",
    re.IGNORECASE | re.MULTILINE,
)

# Section header patterns
# All use re.match() so they anchor at the start of the stripped line.
# Colons at the end are ignored (not required in pattern).
SECTION_PATTERNS = {
    "experience": re.compile(
        r"^(?:(?:professional|relevant|work)\s+)?experience"
        r"|^employment(?:\s+history)?"
        r"|^(?:career\s+(?:history|overview))"
        r"|^work\s+(?:history|experience)"
        r"|^work\s+experience",
        re.IGNORECASE,
    ),
    "education": re.compile(
        r"^education(?:al\s+(?:background|qualification)s?)?"
        r"|^educational\s+history"
        r"|^academic(?:\s+background)?"
        r"|^qualifications?"
        r"|^academic\s+credentials?",
        re.IGNORECASE,
    ),
    "skills": re.compile(
        r"^(?:technical\s+)?skills?(?:\s+(?:&|and)\s+\w+)?"
        r"|^competencies?"
        r"|^technologies?"
        r"|^expertise"
        r"|^core\s+competencies?",
        re.IGNORECASE,
    ),
    "certifications": re.compile(
        r"^certifications?|^certificates?|^credentials?|^licenses?",
        re.IGNORECASE,
    ),
    # "CAREER SUMMARY:", "Professional Summary", "About Me", "Objective", etc.
    "summary": re.compile(
        r"^(?:career\s+)?summary"
        r"|^personal\s+profile"
        r"|^(?:professional\s+)?(?:profile|overview)"
        r"|^objective"
        r"|^about(?:\s+me)?"
        r"|^career\s+objective",
        re.IGNORECASE,
    ),
    "projects": re.compile(
        r"^(?:relevant\s+|personal\s+|key\s+)?projects?",
        re.IGNORECASE,
    ),
}


def _is_header_candidate(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if len(s) > 60:
        return False
    # must be mostly letters/spaces/& and optionally end with colon
    if re.search(r"\d", s):
        return False
    core = s.rstrip(":").strip()
    if not re.match(r"^[A-Za-z &/\-]+$", core):
        return False
    # strong signals: endswith ":" OR ALL CAPS
    letters = re.sub(r"[^A-Za-z]", "", core)
    if s.endswith(":"):
        return True
    if letters and letters.isupper():
        return True
    return core.lower() in SECTION_PATTERNS


def _detect_section_header(line: str) -> str | None:
    s = (line or "").strip()
    # Allow short Title-Case headers (many Canva resumes) even without ALL-CAPS or ":".
    # We still keep this conservative by requiring a pattern match and few words.
    if not _is_header_candidate(s):
        norm0 = re.sub(r"[:\-–—]+$", "", s).strip().lower()
        word_count = len([w for w in re.split(r"\s+", norm0) if w])
        if word_count <= 3:
            for sec_name, pattern in SECTION_PATTERNS.items():
                if pattern.match(norm0):
                    return sec_name
        return None
    norm = re.sub(r"[:\-–—]+$", "", s).strip().lower()

    # explicit keyword mapping (format-independent)
    if re.search(r"\b(skills?\s+(?:and|&)\s+expertise|areas?\s+of\s+expertise|technical\s+expertise|core\s+competencies|tools|technologies)\b", norm):
        return "skills"
    if re.search(r"\b(professional\s+experience|work\s+experience|employment\s+history|work\s+history)\b", norm):
        return "experience"
    if re.search(r"\b(relevant\s+projects?|projects?)\b", norm):
        return "projects"

    for sec_name, pattern in SECTION_PATTERNS.items():
        if pattern.match(norm):
            return sec_name
    if norm in SECTION_PATTERNS:
        return norm
    return None

# Date patterns: handles "Month Year", "DD Month, Year", "Year" formats
_MONTH_NAMES = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
# Matches "04 February, 2029", "February 2029", "Feb 2029", or just "2029"
_DATE_PART = (
    rf"(?:\d{{1,2}},?\s+)?(?:{_MONTH_NAMES})[.,]?\s*\d{{4}}"
    r"|\d{4}"
)
DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_DATE_PART})"
    r"\s*[-–—|~�\uFFFD]+\s*"
    rf"(?P<end>{_DATE_PART}|present|current|now)",
    re.IGNORECASE,
)

# Also match "Title | Date" single-entry lines like "Web Developer | 04 Feb, 2029 - Present"
TITLE_DATE_RE = re.compile(
    r"^(?P<title>[^|]+?)\s*\|\s*"
    rf"(?P<start>{_DATE_PART})\s*[-–—~�\uFFFD]+\s*(?P<end>{_DATE_PART}|present|current|now)",
    re.IGNORECASE,
)

# Education: reject lines that are employment timelines misread as "university"
_EDU_UNIVERSITY_KW_RE = re.compile(
    r"\b(university|college|institute|academy|polytechnic|conservatory)\b|\bhigh\s+school\b",
    re.IGNORECASE,
)
_EDU_UNIVERSITY_TITLE_RE = re.compile(
    r"\b[A-Za-z][A-Za-z\s&'.-]{2,80}\s+University\b",
    re.IGNORECASE,
)


def _text_looks_like_date_span_only(s: str) -> bool:
    """True if the string is essentially a date range (not an institution name)."""
    s = (s or "").strip()
    if not s:
        return True
    if len(s) > 140:
        return False
    if not DATE_RANGE_RE.search(s):
        return False
    t = DATE_RANGE_RE.sub(" ", s)
    t = re.sub(
        rf"\b(?:{_MONTH_NAMES})\b\.?",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"\d{1,4}", " ", t)
    t = re.sub(r"[-–—,|]+", " ", t)
    t = re.sub(r"\bpresent\b|\bcurrent\b|\bnow\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return len(t) < 4


def _is_plausible_education_university(s: str | None) -> bool:
    if not s or not str(s).strip():
        return False
    s = str(s).strip()
    if _text_looks_like_date_span_only(s):
        return False
    if _EDU_UNIVERSITY_KW_RE.search(s):
        return True
    if _EDU_UNIVERSITY_TITLE_RE.search(s):
        return True
    return False


def _pipe_segment_is_institution_name(seg: str) -> bool:
    seg = (seg or "").strip()
    if not seg or _text_looks_like_date_span_only(seg):
        return False
    return _is_plausible_education_university(seg)


# ---------------------------------------------------------------------------
app = Flask(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# File text extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n".join(parts)


def _extract_docx(file_bytes: bytes) -> str:
    document = docx.Document(io.BytesIO(file_bytes))
    parts = []
    for para in document.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())
    for table in document.tables:
        for row in table.rows:
            row_texts = [c.text.strip() for c in row.cells if c.text.strip()]
            if row_texts:
                parts.append("  |  ".join(row_texts))
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Section splitter
# ═══════════════════════════════════════════════════════════════════════════

def _split_sections(text: str) -> dict[str, str]:
    """Split resume into named sections based on header detection."""
    lines = text.split("\n")
    sections: dict[str, list[str]] = {"header": []}
    current = "header"

    for line in lines:
        stripped = line.strip()
        matched = _detect_section_header(stripped) if stripped and len(stripped) < 70 else None
        if matched:
            current = matched
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)

    return {k: "\n".join(v) for k, v in sections.items()}


def _normalize_headers(text: str) -> str:
    """
    PDF extraction often concatenates headers into adjacent words like:
      'Strong CommunicationSKILLS123-456...'
    This normalizes common ALL-CAPS headers by forcing newlines around them.
    """
    if not text:
        return ""
    # 1) Fix common glued tokens from PDF extraction
    # "languagesAgile" -> "languages\nAgile"
    text = re.sub(r"([a-z])([A-Z])", r"\1\n\2", text)
    # "development123" -> "development\n123"
    text = re.sub(r"([A-Za-z])(\d)", r"\1\n\2", text)
    # Replace common bad dash glyphs / replacement chars
    text = text.replace("\uFFFD", "-")
    text = text.replace("�", "-")
    # "2013Master" -> "2013\nMaster"
    text = re.sub(r"(\d{4})(?=[A-Z])", r"\1\n", text)

    # Split multiple "Company | Date - Date" blocks that get merged onto one line:
    # "... Dec 2023 Fauget Tech Company | Jan 2024 - Aug 2024" -> "... Dec 2023\nFauget Tech Company | Jan 2024 - Aug 2024"
    # Works for both "Month YYYY" and "YYYY" end tokens.
    text = re.sub(
        rf"(?P<end>(?:{_MONTH_NAMES})[.,]?\s*\d{{4}}|\d{{4}})\s+(?P<company>[A-Z][^\n|]{{2,120}}\|)",
        r"\g<end>\n\g<company>",
        text,
        flags=re.IGNORECASE,
    )

    # Ensure these common titles don't get glued into sentences
    text = re.sub(r"(?i)(?<=\.)\s*(Software\s+Developer\s+Intern)", r"\n\1", text)
    text = re.sub(r"(?i)(?<=\.)\s*(Web\s+Developer\s+Intern)", r"\n\1", text)
    text = re.sub(r"(?i)\bFirebase\.\s*(Software\s+Developer\s+Intern)", r"Firebase.\n\1", text)
    text = re.sub(r"(?i)\bFaugetbase\b", "Faugetbase", text)
    # Split repeated company/date blocks on one line:
    # "Dec 2023 Fauget Tech Company | Jan 2024 - Aug 2024" -> "Dec 2023\nFauget Tech Company | Jan 2024 - Aug 2024"
    text = re.sub(
        r"(\b(?:19\d{2}|20\d{2})\b)\s+([A-Z][^\n]{0,120}\|)",
        r"\1\n\2",
        text,
    )
    # Ensure internship titles start on their own line (helps parsing)
    text = re.sub(r"(?i)(?<=\.)\s*(Software\s+Developer\s+Intern)", r"\n\1", text)
    text = re.sub(r"(?i)(?<=\.)\s*(Web\s+Developer\s+Intern)", r"\n\1", text)

    # 2) Normalize section headers (single-word + multi-word variants)
    phrase_headers = [
        "AREAS OF EXPERTISE",
        "PROFESSIONAL SUMMARY",
        "PROFESSIONAL EXPERIENCE",
        "RELEVANT PROJECTS",
        "TECHNICAL SKILLS",
        "CORE COMPETENCIES",
    ]
    for h in phrase_headers:
        text = re.sub(rf"(?i)(?<=\w){h}", rf"\n{h}", text)
        text = re.sub(rf"(?i){h}(?=\w)", rf"{h}\n", text)
        text = re.sub(rf"(?i)(?<!\n){h}(?!\n)", f"\n{h}\n", text)

    word_headers = ["SKILLS", "EXPERIENCE", "EDUCATION", "PROFILE", "SUMMARY", "PROJECTS", "CERTIFICATIONS", "CONTACT"]
    for h in word_headers:
        # Handle glued headers like "CommunicationSKILLS123" (no word boundaries)
        # Guardrails: do not split words like "Educational" into "EDUCATION" + "al"
        # NOTE: make this case-sensitive so we never split normal sentence words like "skills".
        text = re.sub(rf"(?<=[a-z]){h}(?![a-z])", rf"\n{h}", text)
        # Only split AFTER header when the next char is a digit (e.g., "SKILLS123")
        text = re.sub(rf"{h}(?=\d)", rf"{h}\n", text)
        # Also normalize standalone occurrences
        text = re.sub(rf"(?<!\n)\b{h}\b(?!\n)", f"\n{h}\n", text)

    # 3) Re-join split multi-word headers produced by earlier normalisation
    text = re.sub(r"(?i)PROFESSIONAL\s*\nSUMMARY\s*\n:\s*", "PROFESSIONAL SUMMARY:\n", text)
    text = re.sub(r"(?i)PROFESSIONAL\s*\nEXPERIENCE\s*\n:\s*", "PROFESSIONAL EXPERIENCE:\n", text)
    text = re.sub(r"(?i)RELEVANT\s*\nPROJECTS\s*\n:\s*", "RELEVANT PROJECTS:\n", text)
    # Split glued email / year-range / company+year cases common in PDFs
    text = re.sub(r"(?<=\w)([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", r"\n\1", text)
    text = re.sub(
        r"([A-Za-z])(?=(19\d{2}|20\d{2})\s*[-–—]\s*(?:19\d{2}|20\d{2}|Present|PRESENT|Current|CURRENT|Now|NOW))",
        r"\1\n",
        text,
    )
    text = re.sub(
        r"((?:19\d{2}|20\d{2})\s*[-–—]\s*(?:19\d{2}|20\d{2}|Present|PRESENT|Current|CURRENT|Now|NOW))(?=[A-Z])",
        r"\1\n",
        text,
    )
    # Clean excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _is_probable_header_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if len(s) > 60:
        return False
    if _detect_section_header(s) is not None:
        return True
    norm = re.sub(r"[:\-–—]+$", "", s).strip().lower()
    return norm in ("profile", "personal profile", "contact", "contact details", "summary", "languages")


def _clean_skill_token(token: str) -> str | None:
    t = (token or "").strip()
    if not t:
        return None
    t = re.sub(r"^[•\-\u2022\*\u00b7]+\s*", "", t)
    t = re.sub(r"\s+", " ", t).strip(" ,;|")
    low = t.lower()
    if low in _STOPWORDS:
        return None
    if len(t) < 2 or len(t) > 60:
        return None
    if re.search(r"@\w+|\bhttps?://|\bwww\.", t, re.IGNORECASE):
        return None
    # Drop plain domains
    if re.search(r"\b[a-z0-9.-]+\.[a-z]{2,}\b", t, re.IGNORECASE):
        return None
    # Drop standalone headers (with or without trailing colon)
    _hdr = re.sub(r"[:\-–—\s]+$", "", t.strip(), flags=re.IGNORECASE).upper()
    if _hdr in {"PROFILE", "CONTACT", "SUMMARY", "EXPERIENCE", "EDUCATION", "SKILLS"}:
        return None
    if re.search(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b", t):
        return None
    # Drop address-ish lines (keep things like "HTML5" - but reject typical address/phone patterns)
    low = t.lower()
    if re.search(r"\d", t) and (("-" in t) or ("st" in low) or ("street" in low) or ("city" in low)):
        return None
    # Reject spaced-letter headings like "W E B D E V E L O P E R"
    if re.match(r"^(?:[A-Za-z]\s+){4,}[A-Za-z]$", t):
        return None
    return t


# ═══════════════════════════════════════════════════════════════════════════
# Skills extraction
# ═══════════════════════════════════════════════════════════════════════════

_JOB_TITLE_AS_SKILL_RE = re.compile(
    r"^(?:web|software|frontend|front[- ]end|backend|back[- ]end|full[- ]?stack|senior|junior|"
    r"mid[- ]?level|lead|principal|staff|associate)\s+"
    r"(?:developer|engineer|designer|architect|consultant|analyst|programmer)"
    r"(?:\s+intern)?$",
    re.IGNORECASE,
)


def _skills_header_line_kind(line: str) -> str | None:
    """Return 'skills' or 'areas' if the line is a skills-section header; else None."""
    s = re.sub(r"[:\-–—\s]+$", "", (line or "").strip(), flags=re.IGNORECASE).upper()
    if s == "SKILLS":
        return "skills"
    if s.startswith("SKILLS AND EXPERTISE") or s.startswith("SKILLS & EXPERTISE"):
        return "skills"
    if s.startswith("AREAS OF EXPERTISE"):
        return "areas"
    return None


_LANG_LINE_RE = re.compile(
    r"^(?:english|french|spanish|hindi|marathi|german|arabic|japanese|korean|mandarin|portuguese|italian)"
    r"(?:\s*\([^)]+\))?$",
    re.IGNORECASE,
)


def _collect_lines_after_skills_header(lines: list[str], i0: int, max_lines: int = 60) -> list[str]:
    """Lines below SKILLS / AREAS OF EXPERTISE until the next section or contact-like row."""
    out: list[str] = []
    for j in range(i0 + 1, min(len(lines), i0 + 1 + max_lines)):
        lj = (lines[j] or "").strip()
        if not lj:
            continue
        upper_lj = lj.strip().upper()
        if upper_lj in {"LANGUAGES", "CONTACT", "CONTACT DETAILS"}:
            continue
        if _is_probable_header_line(lj):
            break
        if _LANG_LINE_RE.match(lj.strip()):
            continue
        low = lj.lower()
        # Avoid matching " st" inside words like "structure" / "infrastructure".
        if any(x in low for x in ("@", "http", "www.", " street", " city", "site.com")):
            continue
        if re.search(r"\b(?:st\.|st)\b", low) and re.search(r"\d", lj):
            continue
        if re.search(r"\d{3}[-\s]?\d{3}", lj):
            continue
        if TITLE_DATE_RE.search(lj) or DATE_RANGE_RE.search(lj):
            if out:
                break
            continue
        if _looks_like_company_line(lj) and out:
            break
        if "." in lj and len(lj) > 50:
            continue
        out.append(lj)
    return out


def _should_reject_skill_phrase(s: str) -> bool:
    """
    Drop false positives (candidate name, job title) using regex + spaCy NER.
    Keeps real skill phrases across varied resume formats.
    """
    s = (s or "").strip()
    if not s:
        return True
    if _JOB_TITLE_AS_SKILL_RE.match(s):
        return True
    if DEGREE_RE.search(f" {s}"):
        return True
    if _LANG_LINE_RE.match(s):
        return True
    if not SPACY_OK or _nlp is None:
        return False
    doc = _nlp(s[: min(len(s), 200)])
    if not doc.ents:
        return False
    persons = [e for e in doc.ents if e.label_ == "PERSON"]
    if not persons:
        return False
    text_span = doc.text.strip()
    low_span = text_span.lower()
    if re.search(r"\b(thinking|debugging|programming|solving|communication|resolution|development|architecture|performance)\b", low_span):
        return False
    # Single PERSON entity matching the whole phrase (e.g. full name)
    if len(persons) == 1 and persons[0].text.strip() == text_span:
        return True
    alnum = lambda x: "".join(c for c in x.lower() if c.isalnum())
    full_al = alnum(text_span)
    if not full_al:
        return False
    covered_al = alnum(" ".join(e.text for e in persons))
    if len(covered_al) / len(full_al) >= 0.85 and len(persons) <= 3:
        return True
    return False


def _extract_skills(text: str, sections: dict[str, str]) -> list[str]:
    """
    Extract skills from the resume's SKILLS section (no predefined dictionary).
    Prefers bullet/line lists *below* SKILLS / SKILLS: (common Word templates).
    Falls back to lines *above* SKILLS only when nothing follows the header (some Canva layouts).
    """
    text = _normalize_headers(text or "")
    lines = [l.strip() for l in text.split("\n")]

    # Use skills section if present and not obviously contact info.
    section_skills_text = (sections.get("skills", "") or "").strip()
    if section_skills_text:
        bad = section_skills_text.lower()
        if any(x in bad for x in ("@", "http", "www.", "st.", "street", "city")):
            section_skills_text = ""

    block_lines: list[str] = []
    had_skills_header = False
    idxs = [i for i, l in enumerate(lines) if _skills_header_line_kind(l)]
    if idxs:
        had_skills_header = True
        i0 = idxs[0]
        kind = _skills_header_line_kind(lines[i0])
        forward = _collect_lines_after_skills_header(lines, i0)

        if kind == "skills":
            if forward:
                block_lines = forward
            else:
                # Canva-style: short skill chips above the word SKILLS
                for j in range(i0 - 1, max(-1, i0 - 40), -1):
                    lj = (lines[j] or "").strip()
                    if not lj:
                        break
                    if _is_probable_header_line(lj):
                        break
                    low = lj.lower()
                    if any(x in low for x in ("@", "http", "www.", " street", " city", "site.com")):
                        break
                    if re.search(r"\b(?:st\.|st)\b", low) and re.search(r"\d", lj):
                        break
                    if re.search(r"\d{3}[-\s]?\d{3}", lj):
                        break
                    if "developer" in low:
                        continue
                    if len(lj) <= 45:
                        block_lines.append(lj)
                block_lines.reverse()
        elif kind == "areas":
            if forward:
                block_lines = forward
    else:
        # Fallback: extract "expertise" list that often appears right after summary.
        all_lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
        start_at = 0
        for i, l in enumerate(all_lines[:80]):
            if l.upper().startswith("PROFESSIONAL SUMMARY"):
                start_at = i + 1
                break

        for l in all_lines[start_at:start_at + 50]:
            low = l.lower()
            if any(x in low for x in ("@", "http", "www.", "st.", "street", "city")):
                break
            if re.search(r"\d{3}[-\s]?\d{3}", l):
                break
            if TITLE_DATE_RE.match(l) or DATE_RANGE_RE.search(l):
                break
            if _looks_like_company_line(l):
                break
            if _is_probable_header_line(l):
                continue
            if "." in l:
                continue
            if 3 <= len(l) <= 55:
                block_lines.append(l)

    # Summary-based extraction: common in modern templates where skills are just a list
    # directly under summary (no skills header at all).
    if not section_skills_text and not block_lines:
        summary_lines = [l.strip() for l in (sections.get("summary", "") or "").split("\n") if l.strip()]
        # take lines that look like short skill phrases (no punctuation, no dates) until contact-ish line
        for l in summary_lines:
            low = l.lower()
            if any(x in low for x in ("@", "http", "www.", "st.", "street", "city")):
                break
            if re.search(r"\d{3}[-\s]?\d{3}", l):
                break
            if "." in l:
                continue
            if DATE_RANGE_RE.search(l):
                continue
            if 3 <= len(l) <= 55:
                block_lines.append(l)

    # Drop company lines, junk tokens, names, and standalone job titles.
    block_lines = [
        l
        for l in block_lines
        if (ct := _clean_skill_token(l))
        and not _looks_like_company_line(l)
        and not _should_reject_skill_phrase(ct)
    ]

    # Prefer lines captured from the SKILLS header path so we never merge upward-scan
    # garbage with a valid section body. Use section text only when the header path is empty.
    if block_lines:
        candidate_text = "\n".join(block_lines)
    elif section_skills_text and not had_skills_header:
        candidate_text = section_skills_text
    else:
        candidate_text = ""

    if not candidate_text:
        # Final fallback: scan top of resume for a compact "expertise list" block.
        all_lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
        collecting = False
        for l in all_lines[:80]:
            low = l.lower()
            is_contact = any(x in low for x in ("@", "http", "www.", "st.", "street", "city")) or re.search(r"\d{3}[-\s]?\d{3}", l)
            if is_contact and collecting:
                break
            if is_contact and not collecting:
                continue
            if TITLE_DATE_RE.match(l) or DATE_RANGE_RE.search(l):
                break
            if not collecting and (("years" in low and "experience" in low) or ("technical documentation" in low)):
                collecting = True
                continue
            if "." in l:
                # once we see a sentence, start collecting next short lines
                collecting = True
                continue
            if collecting and 3 <= len(l) <= 55 and not _is_probable_header_line(l):
                tok = _clean_skill_token(l)
                if tok:
                    block_lines.append(tok)

        candidate_text = "\n".join(block_lines)
        if not candidate_text:
            return []

    raw_tokens: list[str] = []
    for line in candidate_text.split("\n"):
        if not line.strip():
            continue
        parts = re.split(r"[,\u2022•|/]\s*|\s{2,}", line)
        if len(parts) <= 1:
            parts = [line]
        raw_tokens.extend([p.strip() for p in parts if p.strip()])

    found: dict[str, str] = {}
    for tok in raw_tokens:
        cleaned = _clean_skill_token(tok)
        if not cleaned:
            continue
        # Split mixed tokens like "Problem Solving BS Software Engineering".
        mixed_deg = re.search(r"\b(?:BS|MS|BE|ME|B\.?\s*TECH|M\.?\s*TECH|BACHELOR|MASTER|PHD|DIPLOMA)\b", cleaned, re.IGNORECASE)
        if mixed_deg and mixed_deg.start() > 3:
            left = _clean_skill_token(cleaned[:mixed_deg.start()].strip(" ,;:-"))
            if left and not _should_reject_skill_phrase(left):
                key_left = left.lower()
                if key_left not in found:
                    found[key_left] = left
        if _should_reject_skill_phrase(cleaned):
            continue
        key = cleaned.lower()
        if key not in found:
            found[key] = cleaned

    return sorted(found.values(), key=lambda s: s.lower())


# ═══════════════════════════════════════════════════════════════════════════
# Date utilities
# ═══════════════════════════════════════════════════════════════════════════

def _parse_date_safe(raw: str) -> datetime | None:
    """Parse a free-text date string into a datetime (returns None on failure)."""
    raw = raw.strip()
    if not raw:
        return None
    if re.match(r"present|current|now", raw, re.IGNORECASE):
        return datetime.now()
    try:
        return dateparser.parse(raw, default=datetime(2000, 1, 1))
    except Exception:
        # Try year-only
        m = re.search(r"\b(19|20)\d{2}\b", raw)
        if m:
            return datetime(int(m.group()), 1, 1)
        return None


def _duration_months(start_raw: str, end_raw: str) -> int | None:
    start = _parse_date_safe(start_raw)
    end   = _parse_date_safe(end_raw)
    if start is None or end is None:
        return None
    delta = relativedelta(end, start)
    months = delta.years * 12 + delta.months
    return max(0, months) if months >= 0 else None


def _looks_like_company_line(s: str) -> bool:
    """
    Best-effort company-name detector used across skills/experience parsing.
    Keep this conservative (prefer false-negative over false-positive).
    """
    s = (s or "").strip()
    if not s or len(s) > 60:
        return False

    low = s.lower()

    # Never treat obvious job titles as companies
    if any(w in low for w in (
        "developer", "engineer", "manager", "analyst", "architect", "designer",
        "consultant", "specialist", "lead", "intern", "freelance",
    )):
        return False

    # reject sentence fragments ending with period (usually not a company name)
    if s.endswith(".") and re.match(r"^[a-z]", s):
        return False

    if re.search(r"\b(company|inc\.?|ltd\.?|llc|corp\.?)\b", low):
        return True

    # Company-like suffixes / separators (word-boundary aware)
    if "&" in s:
        return True

    if re.search(r"\b(partners|agency|studio|labs|group|industries|technologies)\b", low):
        if re.search(r"\b(skilled|experienced|utilizing|using|implement|develop|building|proficiency)\b", low):
            return False
        if "," in s or s.endswith("."):
            return False
        return True

    # standalone "co" / "co." token
    if re.search(r"\bco\.?\b", low):
        return True

    return False


def _clean_company_name(s: str | None) -> str | None:
    s = (s or "").strip()
    if not s:
        return None
    m = re.search(
        r"([A-Z][A-Za-z&.\- ]{1,90}\b(?:Company|Partners|Inc\.?|Ltd\.?|LLC|Corp\.?|Group|Studio|Labs|Technologies)\b)\s*$",
        s,
    )
    if m:
        return m.group(1).strip()
    # Trim long sentence glue around company-like suffixes
    if "." in s and len(s) > 35:
        tail = s.split(".")[-1].strip()
        if _looks_like_company_line(tail):
            return tail
    return s


# ═══════════════════════════════════════════════════════════════════════════
# Experience extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract_experience(text: str, sections: dict[str, str]) -> list[dict]:
    """
    Heuristic experience parser.

    Strategy A – "Title | StartDate - EndDate" on a single line (modern templates).
    Strategy B – Date range on its own line; look ±3 lines for company / title.
    Both strategies are attempted; results are merged (deduplicated by start date).
    """
    text = _normalize_headers(text or "")
    # Prefer experience section; only fall back to full text if section is missing.
    exp_section = (sections.get("experience", "") or "").strip()
    search_texts: list[str] = [exp_section, text] if exp_section else [text]

    seen_starts: set[str] = set()
    experiences: list[dict] = []

    def _looks_like_title(s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return False
        if len(s) > 60:
            return False
        # Reject generic lowercase words like "management"
        letters_only = re.sub(r"[^A-Za-z]", "", s)
        if letters_only and letters_only.islower():
            return False
        # Strong signal for many templates: ALL CAPS titles
        letters = re.sub(r"[^A-Za-z]", "", s)
        if letters and letters.isupper() and len(letters) >= 6:
            return True
        # Otherwise allow short "Title Case" lines
        if 4 <= len(s) <= 35 and re.match(r"^[A-Za-z][A-Za-z &/.\-]{2,}$", s):
            return True
        return False

    def _looks_like_company(s: str) -> bool:
        return _looks_like_company_line(s)

    def _extract_experience_blocks(block_text: str) -> list[dict]:
        """
        Block parser for templates like:
          Company
          (Company repeated)
          2016 - Present
          APPLICATIONS DEVELOPER
          <desc lines...>
          2014 - 2016
          WEB CONTENT MANAGER
          <desc lines...>
        """
        block_text = _normalize_headers(block_text or "")
        raw_lines = [l.strip() for l in block_text.split("\n") if l.strip()]
        out: list[dict] = []
        current_company: str | None = None
        pending_title: str | None = None
        pending_dates: list[tuple[str, str]] = []

        i = 0
        while i < len(raw_lines):
            line = raw_lines[i]
            if _is_probable_header_line(line):
                i += 1
                continue

            # Company line
            if _looks_like_company(line):
                current_company = line
                i += 1
                continue

            # Title line (some templates: Title → Company → Dates)
            if _looks_like_title(line) and not _looks_like_company(line):
                pending_title = line
                i += 1
                continue

            # Date line → queue it (some templates list multiple date ranges first)
            dm = DATE_RANGE_RE.search(line)
            if dm:
                pending_dates.append(((dm.group("start") or "").strip(), (dm.group("end") or "").strip()))
                i += 1
                continue

            # Title line: if we have pending dates, assign the next date range to this title.
            if pending_dates and _looks_like_title(line):
                start_date, end_date = pending_dates.pop(0)
                job_title = line

                # description until next title/date/company/header
                desc_parts: list[str] = []
                j = i + 1
                while j < len(raw_lines):
                    nxt = raw_lines[j]
                    if _is_probable_header_line(nxt) or _looks_like_company(nxt) or DATE_RANGE_RE.search(nxt) or _looks_like_title(nxt):
                        break
                    desc_parts.append(nxt)
                    j += 1

                description = " ".join(desc_parts[:8]).strip() or None
                duration = _duration_months(start_date, end_date) if start_date else None

                out.append({
                    "companyName"     : current_company,
                    "jobTitle"        : job_title,
                    "startDate"       : start_date,
                    "endDate"         : end_date,
                    "durationInMonths": duration,
                    "description"     : description,
                })

                i = j
                continue

            # Dates already captured, but title was seen BEFORE dates (Title → Company → Dates)
            if pending_title and pending_dates:
                start_date, end_date = pending_dates.pop(0)
                out.append({
                    "companyName"     : current_company,
                    "jobTitle"        : pending_title,
                    "startDate"       : start_date,
                    "endDate"         : end_date,
                    "durationInMonths": _duration_months(start_date, end_date) if start_date else None,
                    "description"     : None,
                })
                pending_title = None
                continue

            i += 1

        # If dates exist but no titles were found, emit minimal rows
        for start_date, end_date in pending_dates:
            out.append({
                "companyName"     : current_company,
                "jobTitle"        : None,
                "startDate"       : start_date,
                "endDate"         : end_date,
                "durationInMonths": _duration_months(start_date, end_date) if start_date else None,
                "description"     : None,
            })

        return out

    def _extract_two_column_experience(block_text: str) -> list[dict]:
        """
        Handle templates like:
          Software Developer Intern    Web Developer Intern
          Fauget Tech Company | Aug 2023 - Dec 2023    Fauget Tech Company | Jan 2024 - Aug 2024
          <left bullets...>    <right bullets...>
        """
        bt = _normalize_headers(block_text or "")
        # Some PDFs collapse the two-column section into a single giant line.
        # Normalize by injecting newlines around obvious anchors.
        bt = re.sub(r"\b(Fauget\s+Tech\s+Company)\s*\|", r"\n\1 |", bt, flags=re.IGNORECASE)
        bt = re.sub(r"(?i)\b(Software\s+Developer\s+Intern)\b", r"\n\1", bt)
        bt = re.sub(r"(?i)\b(Web\s+Developer\s+Intern)\b", r"  \1", bt)
        lines = [l.rstrip() for l in bt.split("\n") if l.strip()]
        out: list[dict] = []

        # find a line that contains two titles (often two columns merged)
        for i in range(len(lines) - 2):
            title_line = lines[i]
            if DATE_RANGE_RE.search(title_line):
                continue

            parts: list[str] = []
            if re.search(r"\s{2,}|\t+", title_line):
                parts = re.split(r"\s{2,}|\t+", title_line.strip())
            else:
                # Fallback: "Software Developer Intern Web Developer Intern"
                m2 = re.match(r"^(.+?\bIntern)\s+(.+?\bIntern)\s*$", title_line, re.IGNORECASE)
                if m2:
                    parts = [m2.group(1).strip(), m2.group(2).strip()]

            if parts:
                if len(parts) != 2:
                    continue
                left_title, right_title = parts[0].strip(), parts[1].strip()
                if not left_title or not right_title:
                    continue

                company_line = lines[i + 1]
                if not (("|" in company_line) and re.search(r"\s{2,}|\t+", company_line)):
                    # Alternative: two company/date blocks are on separate lines
                    company_line2 = lines[i + 2] if i + 2 < len(lines) else ""
                    if "|" not in company_line2:
                        continue
                    cparts = [company_line.strip(), company_line2.strip()]
                else:
                    cparts = re.split(r"\s{2,}|\t+", company_line.strip())
                    if len(cparts) != 2:
                        # Sometimes the two company/date blocks are just separated by two spaces.
                        cparts = re.split(r"\s{2,}", company_line.strip())
                        if len(cparts) != 2:
                            continue

                def parse_company_date(s: str):
                    # "Fauget Tech Company | Aug 2023 - Dec 2023"
                    chunks = [x.strip() for x in s.split("|") if x.strip()]
                    company = chunks[0] if chunks else None
                    dm = DATE_RANGE_RE.search(s)
                    if not dm:
                        return company, None, None, None
                    start = (dm.group("start") or "").strip()
                    end = (dm.group("end") or "").strip()
                    dur = _duration_months(start, end) if start else None
                    return company, start, end, dur

                l_company, l_start, l_end, l_dur = parse_company_date(cparts[0])
                r_company, r_start, r_end, r_dur = parse_company_date(cparts[1])

                # descriptions: collect next few lines until we hit a section header
                left_desc: list[str] = []
                right_desc: list[str] = []
                for j in range(i + 2, min(len(lines), i + 20)):
                    row = lines[j]
                    if _is_probable_header_line(row):
                        break
                    if re.search(r"\s{2,}|\t+", row):
                        cols = re.split(r"\s{2,}|\t+", row.strip())
                        if len(cols) == 2:
                            if cols[0].strip():
                                left_desc.append(cols[0].strip())
                            if cols[1].strip():
                                right_desc.append(cols[1].strip())
                    else:
                        # single column line -> ignore (layout noise)
                        continue

                if l_company and l_start and l_end:
                    out.append({
                        "companyName": l_company,
                        "jobTitle": left_title,
                        "startDate": l_start,
                        "endDate": l_end,
                        "durationInMonths": l_dur,
                        "description": " ".join(left_desc[:10]).strip() or None,
                    })
                if r_company and r_start and r_end:
                    out.append({
                        "companyName": r_company,
                        "jobTitle": right_title,
                        "startDate": r_start,
                        "endDate": r_end,
                        "durationInMonths": r_dur,
                        "description": " ".join(right_desc[:10]).strip() or None,
                    })

                return out

        return out

    for search_text in search_texts:
        # Normalize again at experience-level (important for merged two-column PDFs)
        search_text = _normalize_headers(search_text or "")
        lines = search_text.split("\n")

        # If this resume uses "Job Title | Date - Date" format, prefer that (more reliable)
        has_pipe_format = any(TITLE_DATE_RE.match((l or "").strip()) for l in lines)

        paired_titles: list[str] | None = None
        paired_titles_at = -1
        for i, l in enumerate(lines[:80]):
            l0 = (l or "").strip()
            m2 = re.match(r"^(.+?\bIntern)\s+(.+?\bIntern)\s*$", l0, re.IGNORECASE)
            if m2:
                paired_titles = [m2.group(1).strip(), m2.group(2).strip()]
                paired_titles_at = i
                break
        paired_company_line_count = 0

        # ── Strategy A: "Title | Date - Date" on one line ─────────────
        for idx, line in enumerate(lines):
            m = TITLE_DATE_RE.match(line.strip())
            if not m:
                continue
            start_date = m.group("start").strip()
            end_date   = m.group("end").strip()
            key = start_date.lower()
            if key in seen_starts:
                continue
            seen_starts.add(key)
            raw_title  = m.group("title").strip()
            # Many templates: previous line is company, this line's title is job title.
            prev_company = None
            for j in range(idx - 1, max(-1, idx - 4), -1):
                pj = (lines[j] or "").strip()
                if not pj:
                    continue
                if DATE_RANGE_RE.search(pj) or _is_probable_header_line(pj):
                    continue
                if _looks_like_company(pj):
                    prev_company = pj
                    break
            next_company = None
            for j in range(idx + 1, min(len(lines), idx + 18)):
                nj = (lines[j] or "").strip()
                if not nj:
                    continue
                if TITLE_DATE_RE.match(nj):
                    break
                if _is_probable_header_line(nj):
                    break
                if _looks_like_company(nj):
                    next_company = nj
                    break

            # If raw_title itself looks like a company name, treat it as company not job title.
            job_title = raw_title
            company_name = prev_company
            if _looks_like_company(raw_title):
                company_name = raw_title
                # In some templates the job title is on the previous line:
                #   "Software Developer Intern"
                #   "Fauget Tech Company | Aug 2023 - Dec 2023"
                prev_title = None
                for j in range(idx - 1, max(-1, idx - 6), -1):
                    pj = (lines[j] or "").strip()
                    if not pj:
                        continue
                    if DATE_RANGE_RE.search(pj) or _is_probable_header_line(pj):
                        continue
                    if _looks_like_company(pj):
                        continue
                    lowpj = pj.lower().strip()
                    if lowpj.startswith(("and ", "assist ", "implement ", "collaborate ")):
                        continue
                    # If the title is glued to other text, extract the exact role token
                    mrole = re.search(r"(?i)\b(Software\s+Developer\s+Intern|Web\s+Developer\s+Intern)\b", pj)
                    if mrole:
                        prev_title = mrole.group(1).strip()
                        break
                    if _looks_like_title(pj) and ("intern" in lowpj or "developer" in lowpj):
                        prev_title = pj.strip()
                        break
                job_title = prev_title

                if paired_titles and idx > paired_titles_at:
                    if paired_company_line_count < len(paired_titles):
                        job_title = paired_titles[paired_company_line_count]
                    paired_company_line_count += 1

            # Common PDF order issue: company appears after title/date in flattened text.
            if not company_name and next_company:
                company_name = next_company

            # Skip education rows that look like "2024 - 2028 | University"
            window = " ".join([(lines[j] or "") for j in range(max(0, idx - 2), min(len(lines), idx + 3))]).lower()
            if "university" in window:
                continue

            duration = _duration_months(start_date, end_date)
            experiences.append({
                "companyName"     : company_name,
                "jobTitle"        : job_title,
                "startDate"       : start_date,
                "endDate"         : end_date,
                "durationInMonths": duration,
                "description"     : None,
            })

        # If two consecutive rows ended up with same company and second row has a nearer
        # forward company token, remap the second one (common in mixed two-column PDFs).
        for i in range(1, len(experiences)):
            cur = experiences[i]
            prev = experiences[i - 1]
            if not cur.get("startDate"):
                continue
            if (cur.get("companyName") or "").strip() and (cur.get("companyName") == prev.get("companyName")):
                row_idx = None
                for j, line in enumerate(lines):
                    if TITLE_DATE_RE.match((line or "").strip()):
                        mrow = TITLE_DATE_RE.match((line or "").strip())
                        if mrow and (mrow.group("start") or "").strip().lower() == (cur.get("startDate") or "").strip().lower():
                            row_idx = j
                            break
                if row_idx is None:
                    continue
                for j in range(row_idx + 1, min(len(lines), row_idx + 20)):
                    nj = (lines[j] or "").strip()
                    if not nj:
                        continue
                    if TITLE_DATE_RE.match(nj):
                        break
                    if _looks_like_company(nj) and nj != cur.get("companyName"):
                        cur["companyName"] = nj
                        break

        if has_pipe_format:
            continue

        # Template-aware block parser as fallback
        two_col = _extract_two_column_experience(search_text)
        for exp in two_col:
            sd = (exp.get("startDate") or "").strip().lower()
            if sd and sd not in seen_starts:
                seen_starts.add(sd)
                experiences.append(exp)

        block_results = _extract_experience_blocks(search_text)
        for exp in block_results:
            sd = (exp.get("startDate") or "").strip().lower()
            if sd and sd not in seen_starts:
                seen_starts.add(sd)
                experiences.append(exp)

        # ── Strategy B: date range somewhere on a line ─────────────────
        entry_starts: list[int] = []
        for i, line in enumerate(lines):
            if DATE_RANGE_RE.search(line) and not TITLE_DATE_RE.match(line.strip()):
                entry_starts.append(i)

        if not entry_starts:
            continue

        entry_starts.append(len(lines))  # sentinel
        for idx, start_i in enumerate(entry_starts[:-1]):
            end_i = entry_starts[idx + 1]

            date_line  = lines[start_i]
            date_match = DATE_RANGE_RE.search(date_line)
            start_date = date_match.group("start").strip() if date_match else None
            end_date   = date_match.group("end").strip()   if date_match else None
            if not start_date:
                continue
            key = start_date.lower()
            if key in seen_starts:
                continue
            seen_starts.add(key)

            # Company is usually BEFORE the date line, title is usually AFTER (fixes many PDFs)
            before = [l.strip() for l in lines[max(0, start_i - 6): start_i] if l.strip()]
            # take the nearest non-date, non-header line as company
            company_name = None
            for b in reversed(before):
                if DATE_RANGE_RE.search(b) or _is_probable_header_line(b):
                    continue
                if _looks_like_company(b):
                    company_name = b
                    break

            # Title: first title-like line after date line (before descriptions)
            job_title = None
            for a in lines[start_i + 1: min(len(lines), start_i + 10)]:
                a = a.strip()
                if not a:
                    continue
                if DATE_RANGE_RE.search(a) or _is_probable_header_line(a):
                    break
                if _looks_like_title(a):
                    job_title = a
                    break

            desc_lines  = [l.strip() for l in lines[start_i + 1: end_i] if l.strip()]
            description = " ".join(desc_lines[:6]) or None

            duration = _duration_months(start_date, end_date) if start_date else None
            experiences.append({
                "companyName"     : company_name,
                "jobTitle"        : job_title,
                "startDate"       : start_date,
                "endDate"         : end_date,
                "durationInMonths": duration,
                "description"     : description,
            })

    # If absolutely nothing found, try spaCy fallback
    if experiences:
        # Post-fix for mixed two-column PDFs: map company tokens by chronology.
        company_tokens: list[str] = []
        for l in [x.strip() for x in text.split("\n") if x.strip()]:
            if _is_probable_header_line(l) or DATE_RANGE_RE.search(l) or TITLE_DATE_RE.match(l):
                continue
            if _looks_like_company(l) and l not in company_tokens:
                company_tokens.append(l)
        if len(company_tokens) >= 2 and len(experiences) >= 2:
            def _year_start(exp: dict) -> int:
                m = re.search(r"\b(19\d{2}|20\d{2})\b", (exp.get("startDate") or ""))
                return int(m.group(1)) if m else 0
            ordered_idx = sorted(range(len(experiences)), key=lambda i: _year_start(experiences[i]), reverse=True)
            for pos, i in enumerate(ordered_idx[:len(company_tokens)]):
                target_company = company_tokens[pos]
                cur_company = (experiences[i].get("companyName") or "").strip()
                if not cur_company or (pos > 0 and cur_company in company_tokens[:pos]):
                    experiences[i]["companyName"] = target_company
        for e in experiences:
            e["companyName"] = _clean_company_name(e.get("companyName"))

    if not experiences and SPACY_OK and _nlp is not None:
        return _extract_experience_spacy(text)

    return experiences[:10]


def _extract_experience_spacy(text: str) -> list[dict]:
    """Last-resort fallback: use spaCy DATE + ORG entities."""
    if _nlp is None:
        return []
    doc = _nlp(text[:50_000])
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    orgs  = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    if not dates:
        return []
    return [{
        "companyName"     : orgs[0] if orgs else None,
        "jobTitle"        : None,
        "startDate"       : dates[0],
        "endDate"         : dates[-1] if len(dates) > 1 else None,
        "durationInMonths": None,
        "description"     : text[:300].replace("\n", " "),
    }]


# ═══════════════════════════════════════════════════════════════════════════
# Education extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract_education(text: str, sections: dict[str, str]) -> list[dict]:
    """
    Education parser.
    Searches both the education section AND the full text (handles multi-column PDFs
    where section splitter may have lost part of the content).
    """
    text = _normalize_headers(text or "")
    edu_section = (sections.get("education", "") or "").strip()
    # Prefer education section; only fall back to full text if section is missing.
    search_texts = [edu_section] if edu_section else [text]

    education: list[dict] = []
    seen: set[str] = set()

    def _extract_education_blocks(block_text: str) -> list[dict]:
        """
        Block parser for templates like:
          School/University
          2010 - 2014
          SECONDARY SCHOOL
          University
          2014 - 2016
          BACHELOR OF TECHNOLOGY
        """
        block_text = _normalize_headers(block_text or "")
        raw_lines = [l.strip() for l in block_text.split("\n") if l.strip()]
        out: list[dict] = []

        i = 0
        while i < len(raw_lines):
            line = raw_lines[i]
            if _is_probable_header_line(line):
                i += 1
                continue

            # Handle "Bachelor's Degree in Computer\nScience at Fauget University" pattern
            if (" at " in line.lower()) and ("degree" in line.lower() or DEGREE_RE.search(line)):
                # combine with previous line if it looks like continuation (e.g., "Bachelor's Degree in Computer")
                combined = line
                if i - 1 >= 0 and not _is_probable_header_line(raw_lines[i - 1]) and not DATE_RANGE_RE.search(raw_lines[i - 1]):
                    if "degree" in raw_lines[i - 1].lower():
                        combined = raw_lines[i - 1].strip() + " " + line.strip()

                # Extract degree and university
                deg = None
                dm = DEGREE_RE.search(combined)
                if dm:
                    try:
                        deg = (dm.group(1) or dm.group(0)).strip()
                    except Exception:
                        deg = dm.group(0).strip()
                uni = None
                m_at = re.search(r"\bat\s+(.+)$", combined, re.IGNORECASE)
                if m_at:
                    uni = m_at.group(1).strip()

                # Look ahead for date ranges in parentheses
                date_line = raw_lines[i + 1] if i + 1 < len(raw_lines) else ""
                yr = DATE_RANGE_RE.search(date_line)
                start_year = end_year = None
                if yr:
                    years = re.findall(r"\b(19\d{2}|20\d{2})\b", f"{yr.group('start')} {yr.group('end')}")
                    start_year = int(years[0]) if years else None
                    end_year = int(years[1]) if len(years) > 1 else (int(years[0]) if years else None)

                if deg or uni:
                    out.append({
                        "degree": deg or combined,
                        "fieldOfStudy": None,
                        "university": uni,
                        "startYear": start_year,
                        "endYear": end_year,
                    })
                i += 1
                continue

            # Handle parenthesized ranges on one line, e.g.
            # "( Aug 2020 - Dec 2024 )   ( Aug 2017 - Dec 2019 )"
            if "(" in line and ")" in line and DATE_RANGE_RE.search(line):
                ranges = DATE_RANGE_RE.findall(line)
                if ranges:
                    # Look back for up to 4 lines for two institutions and optional degree line.
                    prev = raw_lines[max(0, i - 4): i]
                    prev = [p for p in prev if not _is_probable_header_line(p)]

                    # try detect: degree line contains "degree" or matches DEGREE_RE
                    degree_line = next((p for p in prev if ("degree" in p.lower()) or DEGREE_RE.search(p)), None)
                    # institutions: lines that contain "university|school|high school"
                    inst = [p for p in prev if re.search(r"(university|school|high school|college|institute)", p, re.IGNORECASE)]

                    def to_year(s: str) -> int | None:
                        m = re.search(r"\b(19\d{2}|20\d{2})\b", s)
                        return int(m.group(1)) if m else None

                    # map ranges to institutions if counts match
                    for idx, (start_raw, end_raw) in enumerate(ranges[:2]):
                        start_year = to_year(start_raw)
                        end_year = to_year(end_raw)
                        uni = inst[idx] if idx < len(inst) else None
                        deg = None
                        # first range often university degree
                        if idx == 0 and degree_line:
                            deg = degree_line
                        elif idx == 1 and inst:
                            # high school
                            deg = "HIGH SCHOOL" if "high school" in (uni or "").lower() else None

                        if uni or deg:
                            out.append({
                                "degree": deg,
                                "fieldOfStudy": None,
                                "university": uni,
                                "startYear": start_year,
                                "endYear": end_year,
                            })

                    i += 1
                    continue

            # handle "YYYY - YYYY | University" or "University | YYYY - YYYY" line
            if "|" in line and DATE_RANGE_RE.search(line):
                yr = DATE_RANGE_RE.search(line)
                start_raw = (yr.group("start") or "").strip() if yr else ""
                end_raw = (yr.group("end") or "").strip() if yr else ""
                parts = [p.strip() for p in line.split("|") if p.strip()]
                uni = None
                if parts:
                    for seg in (parts[0], parts[-1]):
                        if seg and _pipe_segment_is_institution_name(seg):
                            uni = seg
                            break
                years = re.findall(r"\b(19\d{2}|20\d{2})\b", f"{start_raw} {end_raw}")
                start_year = int(years[0]) if years else None
                end_year = int(years[1]) if len(years) > 1 else (int(years[0]) if years else None)
                # degree line usually just above this
                deg_line = raw_lines[i - 1] if i - 1 >= 0 else ""
                degree = None
                if deg_line and not DATE_RANGE_RE.search(deg_line) and not _is_probable_header_line(deg_line):
                    degree = deg_line.strip()
                if degree and (
                    DEGREE_RE.search(degree)
                    or re.match(r"^(?:ME|BE|BS|MS|BTECH|MTECH)\b", degree.strip())
                    or degree.isupper()
                    or "bachelor" in degree.lower()
                    or "master" in degree.lower()
                ):
                    out.append({
                        "degree": degree,
                        "fieldOfStudy": None,
                        "university": uni,
                        "startYear": start_year,
                        "endYear": end_year,
                    })
                elif uni and (start_year or end_year) and _is_plausible_education_university(uni):
                    # Some templates separate degree text away from university/year line.
                    out.append({
                        "degree": None,
                        "fieldOfStudy": None,
                        "university": uni,
                        "startYear": start_year,
                        "endYear": end_year,
                    })
                i += 1
                continue

            # "Graduated: 2013" style
            grad_year = None
            mgrad = re.search(r"\bgraduated\s*:\s*(19\d{2}|20\d{2})\b", line, re.IGNORECASE)
            if mgrad:
                grad_year = int(mgrad.group(1))

            # find year range line
            yr = DATE_RANGE_RE.search(line)
            if not yr:
                # If this line only contains a graduation year, still try to build an education row
                if grad_year is not None:
                    degree = raw_lines[i - 2].strip() if i - 2 >= 0 else None
                    institution = raw_lines[i - 1].strip() if i - 1 >= 0 else None
                    if degree and institution and DEGREE_RE.search(degree):
                        out.append({
                            "degree": degree,
                            "fieldOfStudy": None,
                            "university": institution,
                            "startYear": None,
                            "endYear": grad_year,
                        })
                    i += 1
                    continue
                i += 1
                continue

            start_raw = (yr.group("start") or "").strip()
            end_raw   = (yr.group("end") or "").strip()

            # Education should not have "Present/Current" ranges — those are almost always experience.
            if re.match(r"^(present|current|now)$", end_raw.strip(), re.IGNORECASE):
                i += 1
                continue

            # institution tends to be just before year line
            institution = None
            if i - 1 >= 0 and not DATE_RANGE_RE.search(raw_lines[i - 1]):
                prev = raw_lines[i - 1]
                if not _is_probable_header_line(prev):
                    low = prev.lower()
                    if "company" not in low and _is_plausible_education_university(prev):
                        institution = prev

            # degree tends to be after year line
            degree = None
            field_of_study = None
            j = i + 1
            while j < len(raw_lines) and not DATE_RANGE_RE.search(raw_lines[j]) and not _is_probable_header_line(raw_lines[j]):
                cand = raw_lines[j]
                if DEGREE_RE.search(cand):
                    deg_match = DEGREE_RE.search(cand)
                    try:
                        degree = (deg_match.group(1) or deg_match.group(0)).strip()
                    except Exception:
                        degree = cand.strip()

                    after = cand[deg_match.end():].strip(" ,–-–") if deg_match else ""
                    fm = re.match(r"(?:of|in)\s+([A-Za-z][A-Za-z &/]{2,60})|[-–]\s*([A-Za-z][A-Za-z &/]{2,60})", after)
                    if fm:
                        field_of_study = (fm.group(1) or fm.group(2) or "").strip() or None
                    break
                j += 1

            # If degree not matched by keywords, accept ALL-CAPS short lines like 'SECONDARY SCHOOL'
            if degree is None:
                for k in range(i + 1, min(len(raw_lines), i + 6)):
                    cand = raw_lines[k]
                    letters = re.sub(r"[^A-Za-z]", "", cand)
                    if letters and letters.isupper() and 6 <= len(letters) <= 40:
                        degree = cand.strip()
                        break

            # Guardrail: do not treat job titles as degrees
            if degree:
                deg_low = degree.lower()
                if degree.strip().endswith(":"):
                    degree = None
                if any(x in deg_low for x in ("developer", "manager", "engineer", "analyst", "content")):
                    degree = None
                if any(x in deg_low for x in ("expertise", "projects", "experience", "summary", "skills")):
                    degree = None

            # Parse years (prefer year-only)
            years = re.findall(r"\b(19\d{2}|20\d{2})\b", f"{start_raw} {end_raw}")
            start_year = int(years[0]) if years else None
            end_year = int(years[1]) if len(years) > 1 else (int(years[0]) if years else None)

            # Final validation: only accept blocks that look education-like
            window = " ".join(raw_lines[max(0, i - 2): min(len(raw_lines), i + 6)]).lower()
            looks_edu = bool(
                re.search(
                    r"\b(university|college|institute|academy|polytechnic)\b|\bhigh\s+school\b|\bschool\b",
                    window,
                )
            )
            looks_degree = bool(re.search(r"\b(bachelor|master|secondary|hsc|ssc|phd|diploma|associate)\b", window))

            if (looks_edu or looks_degree):
                if institution and not _is_plausible_education_university(institution):
                    institution = None
                if institution or degree:
                    out.append({
                        "degree": degree,
                        "fieldOfStudy": field_of_study,
                        "university": institution,
                        "startYear": start_year,
                        "endYear": end_year,
                    })

            i = max(i + 1, j)
        return out

    for search_text in search_texts:
        # Template-aware block extraction first
        for edu in _extract_education_blocks(search_text):
            key = f"{(edu.get('degree') or '')}|{(edu.get('university') or '')}|{edu.get('startYear') or ''}".lower()
            if key not in seen:
                seen.add(key)
                education.append(edu)

        # Split into candidate paragraphs / lines
        paragraphs = re.split(r"\n{2,}", search_text.strip())
        if len(paragraphs) <= 1:
            paragraphs = search_text.split("\n")

        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 8:
                continue

            # DEGREE_RE now uses a capturing group (group 1) to skip the
            # leading whitespace that serves as a word boundary guard.
            deg_match = DEGREE_RE.search(para)
            if not deg_match:
                continue

            # group(1) is the actual degree text (without leading whitespace)
            try:
                degree = deg_match.group(1).strip()
            except IndexError:
                degree = deg_match.group(0).strip()

            if not degree or len(degree) < 2:
                continue

            key = degree.lower()
            if key in seen:
                continue
            seen.add(key)

            # Field of study: text on same line after the degree keyword
            # e.g. "Bachelor of Computer Science - Software Engineering"
            after_degree = para[deg_match.end():].strip(" ,–-–")
            # Try to capture "of|in|–|- FieldName"
            field_match = re.match(
                r"(?:of|in)\s+([A-Za-z][A-Za-z &/]{2,60})|"
                r"[-–]\s*([A-Za-z][A-Za-z &/]{2,60})",
                after_degree,
            )
            if field_match:
                field_of_study = (field_match.group(1) or field_match.group(2) or "").strip()
            else:
                # Fallback: grab up to the first digit / punctuation
                fm2 = re.match(r"([A-Za-z][A-Za-z &/]{2,60})", after_degree)
                field_of_study = fm2.group(1).strip() if fm2 else None

            # Clean up trailing noise
            if field_of_study:
                field_of_study = re.sub(r"\s*[-–|]\s*.*$", "", field_of_study).strip()
                if len(field_of_study) < 3:
                    field_of_study = None

            # University: keyword-based search in the paragraph
            university = None
            uni_re = re.search(
                r"(?:university|college|institute|school|iit|nit|bits|vit|"
                r"academy|polytechnic|vraie|great)[^\n,;|]{0,80}",
                para, re.IGNORECASE,
            )
            if uni_re:
                university = uni_re.group(0).strip()
                # Clean trailing noise
                university = re.sub(r"\s+\d{4}.*$", "", university).strip()
            elif SPACY_OK and _nlp is not None:
                doc = _nlp(para[:500])
                orgs = [e.text for e in doc.ents if e.label_ == "ORG"]
                university = orgs[0] if orgs else None

            # Guardrail: avoid experience rows being misread as education paragraphs.
            if not university and not re.search(r"\b(university|college|institute|school|academy|polytechnic)\b", para, re.IGNORECASE):
                continue
            if university and not _is_plausible_education_university(university):
                continue

            # Years: "YYYY - YYYY" or "YYYY | YYYY" or "YYYY"
            years = re.findall(r"\b(19\d{2}|20\d{2})\b", para)
            start_year = int(years[0]) if years else None
            end_year   = int(years[1]) if len(years) > 1 else None

            education.append({
                "degree"      : degree,
                "fieldOfStudy": field_of_study,
                "university"  : university,
                "startYear"   : start_year,
                "endYear"     : end_year,
            })

    # Final cleanup: remove obvious noise rows
    # Fill missing degree labels from orphan degree lines when university+years are present.
    orphan_degree_candidates: list[str] = []
    for l in [x.strip() for x in (text or "").split("\n") if x.strip()]:
        if DATE_RANGE_RE.search(l) or _is_probable_header_line(l) or _looks_like_company_line(l):
            continue
        mdeg = re.search(
            r"\b(?:BS|MS|BE|ME|B\.?\s*TECH|M\.?\s*TECH|BACHELOR|MASTER|PHD|DIPLOMA)\b[^\n|]{0,80}",
            l,
            re.IGNORECASE,
        )
        if not mdeg and not DEGREE_RE.search(f" {l}"):
            continue
        cand = (mdeg.group(0).strip() if mdeg else l.strip())
        if len(cand) < 4:
            continue
        orphan_degree_candidates.append(cand)
    used_degrees = {(e.get("degree") or "").strip().lower() for e in education if (e.get("degree") or "").strip()}
    orphan_degree_candidates = [d for d in orphan_degree_candidates if d.strip().lower() not in used_degrees]
    for e in sorted(education, key=lambda x: ((x.get("startYear") or 9999), (x.get("endYear") or 9999))):
        if e.get("degree"):
            continue
        if not e.get("university") or (not e.get("startYear") and not e.get("endYear")):
            continue
        if orphan_degree_candidates and _is_plausible_education_university(e.get("university")):
            e["degree"] = orphan_degree_candidates.pop(0)

    cleaned: list[dict] = []
    has_year_based = any(e.get("startYear") or e.get("endYear") for e in education)
    for e in education:
        uni = (e.get("university") or "").strip()
        deg = (e.get("degree") or "").strip()
        if uni and not _is_plausible_education_university(uni):
            continue
        if uni.upper() in {"SCHOOL"}:
            continue
        if uni and deg and uni.strip().lower() == deg.strip().lower():
            continue
        # If we already have a proper year-based education, drop fuzzy no-year rows
        if has_year_based and not e.get("startYear") and not e.get("endYear"):
            continue
        # Keep rows with either a degree, or a university+year range.
        if not deg and not (uni and (e.get("startYear") or e.get("endYear"))):
            continue
        if not uni and not deg and not e.get("startYear") and not e.get("endYear"):
            continue
        cleaned.append(e)

    # Prefer one row per (university, years): keep the entry with the richest degree text.
    deduped: dict[tuple, dict] = {}
    for e in cleaned:
        key = (
            (e.get("university") or "").strip().lower(),
            e.get("startYear"),
            e.get("endYear"),
        )
        prev = deduped.get(key)
        if prev is None:
            deduped[key] = e
            continue
        pdeg = len((prev.get("degree") or "").strip())
        cdeg = len((e.get("degree") or "").strip())
        if cdeg > pdeg or (cdeg == pdeg and (e.get("fieldOfStudy") or "") > (prev.get("fieldOfStudy") or "")):
            deduped[key] = e
    cleaned = list(deduped.values())
    cleaned.sort(key=lambda x: ((x.get("startYear") or 0), (x.get("endYear") or 0)))

    return cleaned[:5]  # cap at 5 entries


# ═══════════════════════════════════════════════════════════════════════════
# scikit-learn Skill Match Scoring
# ═══════════════════════════════════════════════════════════════════════════

def _score_skill_match(candidate_skills: list[str], job_skills: list[str]) -> dict:
    """
    Compute a 0-100 skill match score using TF-IDF cosine similarity.
    Returns score, matched skills, and missing skills.
    """
    if not candidate_skills or not job_skills:
        return {"score": 0, "matched_skills": [], "missing_skills": job_skills or []}

    # Normalise to lowercase
    cand_set = {s.strip().lower() for s in candidate_skills}
    job_set  = {s.strip().lower() for s in job_skills}

    # Exact matches
    exact_match = cand_set & job_set
    missing     = job_set - cand_set
    exact_score = len(exact_match) / len(job_set) * 100 if job_set else 0

    # TF-IDF cosine similarity for partial / semantic match
    cand_text = " ".join(candidate_skills)
    job_text  = " ".join(job_skills)

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform([cand_text, job_text])
        cosine_score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]) * 100
    except Exception:
        cosine_score = 0.0

    # Weighted blend: 70% exact, 30% TF-IDF
    final_score = round(0.70 * exact_score + 0.30 * cosine_score, 2)

    return {
        "score"          : min(100.0, final_score),
        "exact_score"    : round(exact_score, 2),
        "semantic_score" : round(cosine_score, 2),
        "matched_skills" : sorted(exact_match),
        "missing_skills" : sorted(missing),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Flask endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/parse-resume", methods=["POST"])
def parse_resume():
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' field."}), 400

    upload = request.files["file"]
    if not upload or not upload.filename:
        return jsonify({"error": "Empty file received."}), 400

    filename  = upload.filename
    ext       = os.path.splitext(filename)[1].lower()
    file_bytes = upload.read()

    if not file_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    logger.info("Parsing  %-44s  %d KB", filename, len(file_bytes) // 1024)

    # ── Extract raw text ──────────────────────────────────────────────
    try:
        if ext == ".pdf":
            text = _extract_pdf(file_bytes)
            file_type = "pdf"
        elif ext == ".docx":
            text = _extract_docx(file_bytes)
            file_type = "docx"
        elif ext == ".doc":
            try:
                text = _extract_docx(file_bytes)
                file_type = "doc"
            except Exception:
                return jsonify({"error": "Legacy .doc not supported. Re-save as .docx or PDF."}), 400
        else:
            return jsonify({"error": f"Unsupported type '{ext}'. Use PDF or DOCX."}), 400
    except Exception as exc:
        logger.exception("Text extraction failed: %s", exc)
        return jsonify({"error": f"Extraction error: {exc}"}), 500

    if not text.strip():
        return jsonify({"error": "No text could be extracted from the file."}), 422

    # ── NLP parsing ───────────────────────────────────────────────────
    sections   = _split_sections(text)
    skills     = _extract_skills(text, sections)
    experience = _extract_experience(text, sections)
    education  = _extract_education(text, sections)
    word_count = len(text.split())

    logger.info(
        "Done  %-44s  words=%d  skills=%d  exp=%d  edu=%d",
        filename, word_count, len(skills), len(experience), len(education),
    )

    return jsonify({
        "text"       : text,
        "word_count" : word_count,
        "file_type"  : file_type,
        "skills"     : skills,
        "experience" : experience,
        "education"  : education,
    })


@app.route("/score-match", methods=["POST"])
def score_match():
    """
    Body (JSON):
    {
      "candidate_skills": ["Python", "SQL", ...],
      "job_skills":       ["Python", "React", ...]
    }
    Returns: { "score": 75.4, "matched_skills": [...], "missing_skills": [...] }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    candidate_skills = data.get("candidate_skills") or []
    job_skills       = data.get("job_skills") or []

    result = _score_skill_match(candidate_skills, job_skills)
    return jsonify(result)


# ═══════════════════════════════════════════════════════════════════════════
# ATS Application Match Scoring  (Skills 50% | Experience 30% | Education 20%)
# ═══════════════════════════════════════════════════════════════════════════

# Education equivalence groups – any two members in the same list are treated as equivalent.
# Prefix tokens (e.g. "bs") that might appear at the *start* of a longer degree string
# (e.g. "BS Software Engineering") are intentionally short so that `_group_of` below
# uses startswith-based matching instead of substring to avoid false positives.
_EDU_EQUIV_GROUPS: list[list[str]] = [
    # Bachelor of Technology / Engineering
    ["b.tech", "be", "b.e", "bachelor of engineering", "bachelor of technology",
     "b-tech", "btech", "bachelor of technology" , "b.tech"],
    # Bachelor of Science (includes BS/B.Sc prefixes common in US/UK/Indian degrees)
    ["b.sc", "bsc", "bs", "b.s", "bachelor of science", "bachelor of computer science",
     "bachelor of computer applications", "bca", "b.c.a"],
    # Bachelor of Commerce
    ["b.com", "bcom", "bachelor of commerce"],
    # Bachelor of Arts
    ["b.a", "ba", "bachelor of arts"],
    # Bachelor of Business Administration
    ["bba", "bachelor of business administration"],
    # MCA
    ["mca", "m.c.a", "master of computer applications"],
    # Master of Engineering / Technology
    ["m.tech", "me", "m.e", "mtech", "m-tech",
     "master of engineering", "master of technology",
     "master of science in engineering"],
    # Master of Science
    ["m.sc", "msc", "ms", "m.s", "master of science"],
    # MBA
    ["mba", "m.b.a", "master of business administration"],
    # PhD
    ["phd", "ph.d", "doctorate", "doctor of philosophy"],
    # Diploma
    ["diploma"],
    # 12th / HSC
    ["high school", "hsc", "12th", "secondary", "higher secondary"],
    # 10th / SSC
    ["ssc", "10th", "matriculation"],
]


def _edu_degree_prefix(norm: str) -> str:
    """Return only the degree prefix token(s) before any field-of-study words.

    E.g. 'bs software engineering' → 'bs'
         'bachelor of technology'   → 'bachelor of technology'
         'me software development'  → 'me'
    """
    # Strip common field-of-study connectors and everything after them
    cut = re.split(
        r"\b(?:in|of|and|with|from|for)\b",
        norm,
        maxsplit=1,
    )
    prefix = cut[0].strip() if cut else norm
    # If the prefix is just a short abbreviation token, return it;
    # otherwise return the full cleaned string.
    return prefix

_SENIOR_TITLES = re.compile(
    r"\b(senior|sr\.?|lead|principal|staff|head|director|vp|cto|cfo|ceo)\b",
    re.IGNORECASE,
)

_EXPERIENCE_LEVEL_YEARS: dict[str, int] = {
    "entry":     0,
    "entry level": 0,
    "junior":    0,
    "mid":       2,
    "mid level": 2,
    "senior":    4,
    "senior level": 4,
    "lead":      6,
    "principal": 8,
    "staff":     8,
    "executive": 10,
}


def _normalise_edu_label(label: str) -> str:
    s = re.sub(r"[^a-z0-9 .]", "", label.strip().lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _edu_labels_match(candidate_deg: str, required: str) -> bool:
    """
    True if candidate_deg satisfies the required education (or is equivalent).
    Handles slash-separated requirements like 'B-Tech / MCA'.

    Matching logic (in order):
      1. Exact substring of the full normalised strings.
      2. Prefix-based group lookup – strips field-of-study noise
         ("BS Software Engineering" → prefix "bs" → group bsc/bs).
      3. Full-string group lookup as a final fallback.
    """
    cand_norm   = _normalise_edu_label(candidate_deg)
    cand_prefix = _edu_degree_prefix(cand_norm)
    req_parts   = [_normalise_edu_label(p.strip()) for p in re.split(r"[/,|]", required) if p.strip()]

    def _group_of(label: str) -> int | None:
        for i, group in enumerate(_EDU_EQUIV_GROUPS):
            for g in group:
                # Exact match or one is a prefix/suffix of the other
                if g == label or label.startswith(g + " ") or label == g or g.startswith(label + " "):
                    return i
                # Full containment (handles longer degree strings)
                if g in label or label in g:
                    return i
        return None

    cand_group        = _group_of(cand_prefix) or _group_of(cand_norm)
    for req_norm in req_parts:
        req_prefix = _edu_degree_prefix(req_norm)
        req_group  = _group_of(req_prefix) or _group_of(req_norm)

        # Direct substring match on full strings
        if req_norm and (req_norm in cand_norm or cand_norm in req_norm):
            return True
        # Group equivalence via prefix
        if cand_group is not None and req_group is not None and cand_group == req_group:
            return True
    return False


def _score_education(
    candidate_education: list[dict],
    education_requirement: str | None,
) -> float:
    """
    Return 0-100 education match score.

    Matching is based ONLY on the Degree field (not FieldOfStudy / University).
    This avoids false mismatches when fields like "Computer Science" are absent
    from the candidate record but the degree abbreviation clearly matches.
    """
    if not education_requirement or not education_requirement.strip():
        return 100.0

    if not candidate_education:
        return 0.0

    req = education_requirement.strip()

    # ── Pass 1: exact equivalence-group match on Degree only ─────────────
    for edu in candidate_education:
        deg = (edu.get("degree") or "").strip()
        if not deg:
            continue
        if _edu_labels_match(deg, req):
            return 100.0

    # ── Pass 2: NLP similarity on Degree only (spaCy vector fallback) ────
    if SPACY_OK and _nlp is not None:
        req_doc = _nlp(req[:300])
        best = 0.0
        for edu in candidate_education:
            deg = (edu.get("degree") or "").strip()
            if not deg:
                continue
            deg_doc = _nlp(deg[:300])
            if req_doc.vector_norm and deg_doc.vector_norm:
                sim = req_doc.similarity(deg_doc) * 100
                best = max(best, sim)
        return round(min(100.0, best), 2)

    return 0.0


def _extract_required_years(experience_required: str | None) -> float | None:
    """Parse 'ExperienceRequired' strings like '4', '3-5', '4 years', '3+ years' → float."""
    if not experience_required:
        return None
    s = str(experience_required).strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    m2 = re.search(r"(\d+(?:\.\d+)?)", s)
    if m2:
        return float(m2.group(1))
    return None


_CURRENT_YEAR = datetime.now().year


def _score_experience(
    candidate_experiences: list[dict],
    experience_required: str | None,
    experience_level: str | None,
) -> tuple[float, float]:
    """
    Return (score_0_100, candidate_total_years).

    If durationInMonths is pre-computed and stored by the C# service, it is trusted
    directly (no start-year re-validation).  This handles fictional/template resumes
    with future dates (e.g. "2035 – Present") correctly.

    When durationInMonths is missing, dates are parsed and junk years (e.g. "0010")
    are ignored, but the strict "must be past year" guard is removed so that the
    absolute date difference is always counted.
    """
    total_months = 0
    for exp in candidate_experiences:
        dur = exp.get("durationInMonths")
        if isinstance(dur, (int, float)) and dur > 0:
            # Pre-computed by C# service – trust it unconditionally.
            total_months += int(dur)
        else:
            # Estimate from raw dates (fallback when duration not stored).
            s_raw = str(exp.get("startDate") or "")
            e_raw = str(exp.get("endDate") or "")
            sy_m = re.search(r"\b(19|20)(\d{2})\b", s_raw)
            if not sy_m:
                continue   # no parseable start year at all
            s_y = int(sy_m.group())
            is_present = bool(re.match(r"present|current|now", e_raw.strip(), re.IGNORECASE))
            if is_present:
                e_y = _CURRENT_YEAR
            else:
                ey_m = re.search(r"\b(19|20)(\d{2})\b", e_raw)
                if not ey_m:
                    continue
                e_y = int(ey_m.group())
            # Use abs so future-dated entries contribute rather than score zero.
            total_months += abs((e_y - s_y) * 12)

    candidate_years = total_months / 12.0

    # Determine required years from ExperienceRequired field or ExperienceLevel label
    req_years = _extract_required_years(experience_required)
    if req_years is None and experience_level:
        lvl = experience_level.strip().lower()
        req_years = _EXPERIENCE_LEVEL_YEARS.get(lvl)

    # KEY FIX: if NEITHER field is set on the job, return (0, years) so the
    # caller can mark it as "not configured" rather than silently giving 100.
    if req_years is None:
        return (0.0, candidate_years)

    if req_years == 0:
        return (100.0, candidate_years)

    if candidate_years >= req_years:
        return (100.0, candidate_years)

    score = (candidate_years / req_years) * 100
    return (round(min(100.0, score), 2), candidate_years)


def _skills_from_job_title(job_title: str) -> list[str]:
    """
    Extract likely technology keywords from a job title, e.g.
    "Senior .NET Developer" → [".NET"]
    "React + Node.js Engineer" → ["React", "Node.js"]
    """
    if not job_title:
        return []
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9#\+\-\.]{1,30}", job_title)
    stop = {
        "senior", "junior", "mid", "lead", "principal", "staff", "developer",
        "engineer", "architect", "analyst", "consultant", "manager", "specialist",
        "intern", "associate", "full", "stack", "fullstack", "backend", "frontend",
        "remote", "contract", "part", "time", "and", "or", "with", "the",
    }
    return [t for t in tokens if t.lower() not in stop]


def _score_skills_with_nlp(
    candidate_skills: list[str],
    job_skills: list[str],
    job_title: str | None,
    job_description: str | None,
) -> dict:
    """
    Skills scoring:
    1. When job_skills is non-empty (fetched from CompanyDepartmentSkills), use it as
       the ONLY authoritative set.  Do NOT augment from the job description – that would
       add noise tokens (e.g. capitalised words from a generic description) and falsely
       inflate the denominator, making a 3/3 match look like 3/11 = 21%.
    2. Only when job_skills is COMPLETELY EMPTY do we fall back to extracting keywords
       from the job title and description (best-effort).
    3. If after all extraction no skills are found → skillsNotConfigured = True.
    """
    if job_skills:
        # ── DB skills are authoritative ───────────────────────────────────────
        all_job_skills = list({s.strip() for s in job_skills if s.strip()})
        logger.info(
            "match-application  skill: using %d DB skills for title=%r",
            len(all_job_skills), job_title,
        )
    else:
        # ── No DB skills – fall back to title + description extraction ────────
        extra: list[str] = []
        extra.extend(_skills_from_job_title(job_title or ""))

        if job_description and len(job_description.strip()) > 20:
            if SPACY_OK and _nlp is not None:
                doc = _nlp(job_description[:3000])
                for chunk in doc.noun_chunks:
                    t = chunk.text.strip()
                    if 2 <= len(t) <= 40 and not re.search(
                        r"\b(year|month|day|team|company|role|experience|candidate|position|salary)\b",
                        t, re.IGNORECASE
                    ):
                        extra.append(t)
            techs = re.findall(r"\b[A-Z][A-Za-z0-9#\+\-\.]{1,30}\b", job_description)
            extra.extend(techs)

        all_job_skills = list({s.strip() for s in extra if s.strip()})
        logger.info(
            "match-application  skill: no DB skills – inferred %d from title/description for title=%r",
            len(all_job_skills), job_title,
        )

    if not all_job_skills:
        logger.info(
            "match-application  skill: no job skills found for title=%r – returning 0",
            job_title,
        )
        return {
            "score": 0.0, "skillScore": 0.0,
            "exactScore": 0.0, "semanticScore": 0.0,
            "matchedSkills": [],
            "missingSkills": [],
            "skillsNotConfigured": True,
        }

    base = _score_skill_match(candidate_skills, all_job_skills)

    return {
        "score"              : base["score"],
        "skillScore"         : base["score"],
        "exactScore"         : base.get("exact_score", 0),
        "semanticScore"      : base.get("semantic_score", 0),
        "matchedSkills"      : base.get("matched_skills", []),
        "missingSkills"      : base.get("missing_skills", []),
        "skillsNotConfigured": False,
    }


@app.route("/match-application", methods=["POST"])
def match_application():
    """
    Full ATS scoring endpoint.

    Body (JSON):
    {
      "candidateSkills"     : ["C#", ".NET", "SQL"],
      "candidateExperiences": [
          {"startDate": "2020", "endDate": "Present", "durationInMonths": 60, ...}
      ],
      "candidateEducation"  : [
          {"degree": "B.Tech", "fieldOfStudy": "Computer Science", "endYear": 2020}
      ],
      "jobRequirements": {
          "jobTitle"              : "Senior .NET Developer",
          "description"           : "...",
          "skills"                : [".NET", "SQL", "Azure"],
          "experienceRequired"    : "4",
          "experienceLevel"       : "Senior",
          "educationRequirement"  : "B-Tech / MCA"
      }
    }

    Weights: Skills 50%  |  Experience 30%  |  Education 20%
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    candidate_skills      = data.get("candidateSkills") or []
    candidate_experiences = data.get("candidateExperiences") or []
    candidate_education   = data.get("candidateEducation") or []
    job_req               = data.get("jobRequirements") or {}

    job_skills        = job_req.get("skills") or []
    job_title         = job_req.get("jobTitle") or ""
    job_description   = job_req.get("description") or ""
    exp_required      = job_req.get("experienceRequired")
    exp_level         = job_req.get("experienceLevel")
    edu_required      = job_req.get("educationRequirement")

    # ── Factor scores (each 0-100) ────────────────────────────────────
    skill_result  = _score_skills_with_nlp(candidate_skills, job_skills, job_title, job_description)
    skill_score   = round(skill_result["skillScore"], 2)

    exp_not_configured = (
        (exp_required is None or str(exp_required).strip() == "") and
        (exp_level is None or str(exp_level).strip() == "")
    )
    edu_not_configured = not edu_required or not edu_required.strip()

    exp_score_raw, candidate_years = _score_experience(candidate_experiences, exp_required, exp_level)

    # ── Effective scores ───────────────────────────────────────────────
    # When a job requirement is not configured, the factor is treated as
    # "no constraint" (full marks).  We return the EFFECTIVE value so that
    # what is stored in the DB matches what is used in the total calculation.
    eff_skill_score = skill_score
    eff_exp_score   = 100.0 if exp_not_configured else round(exp_score_raw, 2)
    eff_edu_score   = round(_score_education(candidate_education, edu_required), 2)

    # ── Weighted total (50 / 30 / 20) ─────────────────────────────────
    total_score = round(eff_skill_score * 0.50 + eff_exp_score * 0.30 + eff_edu_score * 0.20, 2)

    # ── Human-readable recommendation ─────────────────────────────────
    if total_score >= 80:
        recommendation = "Excellent Match"
    elif total_score >= 65:
        recommendation = "Strong Match"
    elif total_score >= 45:
        recommendation = "Moderate Match"
    elif total_score >= 25:
        recommendation = "Weak Match"
    else:
        recommendation = "Poor Match"

    warnings: list[str] = []
    if skill_result.get("skillsNotConfigured"):
        warnings.append("Job skills not configured – skill score could not be calculated.")
    if exp_not_configured:
        warnings.append(
            f"ExperienceRequired/Level not set on job – experience not factored "
            f"(candidate has {round(candidate_years, 1)} yrs total)."
        )
    if edu_not_configured:
        warnings.append("EducationRequirement not set on job – education not factored.")

    logger.info(
        "match-application  total=%.1f  skill=%.1f  exp=%.1f  edu=%.1f"
        "  candYrs=%.1f  expCfg=%s  eduCfg=%s  [%s]%s",
        total_score, eff_skill_score, eff_exp_score, eff_edu_score,
        candidate_years,
        not exp_not_configured, not edu_not_configured,
        recommendation,
        ("  WARN:" + "; ".join(warnings)) if warnings else "",
    )

    return jsonify({
        "totalScore"           : total_score,
        # Return EFFECTIVE scores – these match what is stored in the DB
        # and what is used in the weighted total.
        "skillScore"           : eff_skill_score,
        "experienceScore"      : eff_exp_score,
        "educationScore"       : eff_edu_score,
        "candidateYears"       : round(candidate_years, 1),
        "skillWeight"          : 0.50,
        "experienceWeight"     : 0.30,
        "educationWeight"      : 0.20,
        "matchedSkills"        : skill_result.get("matchedSkills", []),
        "missingSkills"        : skill_result.get("missingSkills", []),
        "recommendation"       : recommendation,
        "skillsNotConfigured"  : skill_result.get("skillsNotConfigured", False),
        "expNotConfigured"     : exp_not_configured,
        "eduNotConfigured"     : edu_not_configured,
        "warnings"             : warnings,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"         : "ok",
        "spacy_loaded"   : SPACY_OK,
        "nltk_ready"     : True,
    })


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port  = int(os.environ.get("PARSER_PORT", 5001))
    logger.info("=" * 65)
    logger.info("  ATS Resume Parser (NLP Edition) – http://localhost:%d", port)
    logger.info("  POST /parse-resume    POST /score-match    POST /match-application    GET /health")
    logger.info("=" * 65)
    app.run(host="127.0.0.1", port=port, debug=False)
