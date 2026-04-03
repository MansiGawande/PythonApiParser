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
# Comprehensive skill keyword list
# ---------------------------------------------------------------------------
TECH_SKILLS: set[str] = {
    # Languages
    "python", "java", "javascript", "typescript", "c#", "c++", "c", "go", "golang",
    "rust", "kotlin", "swift", "ruby", "php", "scala", "r", "matlab", "perl", "bash",
    "powershell", "vba", "cobol", "fortran", "dart", "elixir", "haskell", "lua",
    # Web
    "html", "css", "sass", "less", "react", "angular", "vue", "next.js", "nuxt",
    "node.js", "express", "fastapi", "django", "flask", "spring", "asp.net", ".net",
    "laravel", "symfony", "rails", "svelte", "jquery", "bootstrap", "tailwind",
    "graphql", "rest api", "soap", "grpc", "websocket",
    # Mobile
    "android", "ios", "flutter", "react native", "xamarin", "ionic", "swift ui",
    # Data / ML / AI
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra",
    "oracle", "sqlite", "mssql", "sql server", "db2",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "scikit-learn", "tensorflow", "pytorch", "keras", "hugging face", "transformers",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "reinforcement learning", "neural network",
    "data science", "data analysis", "data engineering", "data pipeline",
    "power bi", "tableau", "looker", "qlikview", "excel", "google analytics",
    "apache spark", "hadoop", "kafka", "airflow", "dbt", "flink",
    "etl", "data warehouse", "snowflake", "bigquery", "redshift", "databricks",
    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "heroku", "digitalocean",
    "docker", "kubernetes", "terraform", "ansible", "puppet", "chef",
    "jenkins", "git", "github", "gitlab", "bitbucket", "ci/cd", "devops",
    "linux", "unix", "nginx", "apache", "microservices", "serverless",
    # Testing
    "selenium", "cypress", "jest", "pytest", "junit", "mocha", "tdd", "bdd",
    "unit testing", "integration testing", "automation testing", "postman",
    # Methodologies
    "agile", "scrum", "kanban", "jira", "confluence", "trello", "figma",
    "uml", "design patterns", "solid", "oop", "functional programming",
    # Domain
    "sap", "salesforce", "microsoft dynamics", "erp", "crm",
    "blockchain", "ethereum", "solidity", "web3",
    "cybersecurity", "penetration testing", "owasp",
    # Web-development disciplines (common in resumes)
    "front-end development", "back-end development", "full-stack development",
    "frontend development", "backend development", "fullstack development",
    "web development", "web design", "responsive design",
    "web performance", "seo", "accessibility", "web security",
    "software development", "software engineering", "software architecture",
    "api development", "api integration", "microservice",
    "code review", "debugging", "version control", "technical documentation",
    "agile development", "sprint planning",
}

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
        r"|^work\s+(?:history|experience)",
        re.IGNORECASE,
    ),
    "education": re.compile(
        r"^education(?:al\s+(?:background|qualification)s?)?"
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

# Date patterns: handles "Month Year", "DD Month, Year", "Year" formats
_MONTH_NAMES = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
# Matches "04 February, 2029", "February 2029", "Feb 2029", or just "2029"
_DATE_PART = (
    rf"(?:\d{{1,2}}\s+)?(?:{_MONTH_NAMES})[.,]?\s*\d{{4}}"
    r"|\d{4}"
)
DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_DATE_PART})"
    r"\s*[-–—|]+\s*"
    rf"(?P<end>{_DATE_PART}|present|current|now)",
    re.IGNORECASE,
)

# Also match "Title | Date" single-entry lines like "Web Developer | 04 Feb, 2029 - Present"
TITLE_DATE_RE = re.compile(
    r"^(?P<title>[^|]+?)\s*\|\s*"
    rf"(?P<start>{_DATE_PART})\s*[-–—]+\s*(?P<end>{_DATE_PART}|present|current|now)",
    re.IGNORECASE,
)

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
        matched = None
        if stripped and len(stripped) < 60:
            for sec_name, pattern in SECTION_PATTERNS.items():
                if pattern.match(stripped):
                    matched = sec_name
                    break
        if matched:
            current = matched
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)

    return {k: "\n".join(v) for k, v in sections.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Skills extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract_skills(text: str, sections: dict[str, str]) -> list[str]:
    """
    Keyword match + spaCy ORG/PRODUCT NER.

    Always scans the full resume text so that multi-column PDFs where the
    section splitter lost the skills section are still handled correctly.
    """
    # Combine skills section (preferred) with full text
    skills_section = sections.get("skills", "").strip()
    # Search both: skills section first, then full text (deduplication handled below)
    targets = []
    if skills_section:
        targets.append(skills_section)
    targets.append(text)
    combined = "\n".join(targets).lower()

    found: set[str] = set()

    # Keyword matching across combined text
    for skill in TECH_SKILLS:
        escaped = re.escape(skill)
        # Word-boundary aware; handles c++, c#, .net, react.js, etc.
        pattern = r"(?<![a-zA-Z0-9.\-])" + escaped + r"(?![a-zA-Z0-9.\-])"
        if re.search(pattern, combined):
            # Preserve original casing for multi-word skills; title-case single words
            found.add(skill if " " in skill or any(c in skill for c in ".#+") else skill.title())

    # spaCy NER pass – catches product names not in the keyword list
    if SPACY_OK and _nlp is not None:
        doc = _nlp(text[:100_000])
        for ent in doc.ents:
            clean = ent.text.strip()
            if ent.label_ in ("ORG", "PRODUCT") and 2 <= len(clean) <= 40:
                if clean.lower() in TECH_SKILLS:
                    found.add(clean)

    return sorted(found)


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
    # Search the full text in addition to the experience section so we never
    # miss entries because the section splitter failed on multi-column PDFs.
    search_texts: list[str] = []
    exp_section = sections.get("experience", "").strip()
    if exp_section:
        search_texts.append(exp_section)
    # Always also scan the full resume text
    search_texts.append(text)

    seen_starts: set[str] = set()
    experiences: list[dict] = []

    for search_text in search_texts:
        lines = search_text.split("\n")

        # ── Strategy A: "Title | Date - Date" on one line ─────────────
        for line in lines:
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
            # Title may be "Company\nJob" or "Job  Company" or just "Job"
            # split on two or more spaces or pipe
            parts = re.split(r"\s{2,}|\s*\|\s*", raw_title)
            job_title    = parts[0].strip() if parts else raw_title
            company_name = parts[1].strip() if len(parts) > 1 else None
            duration = _duration_months(start_date, end_date)
            experiences.append({
                "companyName"     : company_name,
                "jobTitle"        : job_title,
                "startDate"       : start_date,
                "endDate"         : end_date,
                "durationInMonths": duration,
                "description"     : None,
            })

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

            # Lines BEFORE the date line → potential company / title
            before = [l.strip() for l in lines[max(0, start_i - 4): start_i] if l.strip()]
            job_title    = before[-1] if before else None
            company_name = before[-2] if len(before) >= 2 else None

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
    edu_section = sections.get("education", "").strip()
    # Always also scan full text so multi-column PDFs don't lose entries
    search_texts = []
    if edu_section:
        search_texts.append(edu_section)
    search_texts.append(text)

    education: list[dict] = []
    seen: set[str] = set()

    for search_text in search_texts:
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

    return education[:5]  # cap at 5 entries


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
    logger.info("  POST /parse-resume    POST /score-match    GET /health")
    logger.info("=" * 65)
    app.run(host="127.0.0.1", port=port, debug=False)
