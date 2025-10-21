# Ukrainian CV Parser — Colab Prototype

A lightweight prototype (single Jupyter/Colab notebook) for parsing **Ukrainian CVs** from **PDF/DOCX/TXT** into **structured JSON**. The notebook reads files from Google Drive, extracts key resume fields (name, contacts, experience, education, skills), and writes aggregated results to disk.

> This repository currently contains the notebook `CHI_NLP_CVs_Vakhovska.ipynb`. The code is optimized for **Google Colab** execution with Google Drive mounted. See the **Audit & Modernization Plan (Preview)** section for how to turn this into a robust, production‑ready toolchain.

---

## What’s Implemented

### Supported Inputs
- **PDF** via `pdfplumber`
- **DOCX** via `python-docx`
- **TXT** via standard file I/O

Files are read from a Google Drive folder, e.g.:
```
/content/drive/MyDrive/Colab Notebooks/files/CVs
```

### Extraction Pipeline
- **File readers**
  - `read_pdf(path)`: extracts text from all pages and concatenates
  - `read_docx(path)`: reads and joins paragraphs
  - `read_txt(path)`: reads UTF‑8 text
- **Field extractors**
  - **Name (Ukrainian)** — spaCy `uk_core_news_lg` NER (PERSON) on early lines with a regex/pattern fallback
  - **Contacts** — phones (UA formats, optional `+38`), emails (regex)
  - **Experience** — section keyword search (e.g., “Досвід/Проєкти”), regex for **position/company/date range**, bullets for responsibilities
  - **Education** — finds “Освіта”; extracts **institution**, **specialization**, **graduation year (YYYY)**
  - **Skills** — finds “Навички/Технології”; parses bullets/dashes into a unique list
- **Assembler**
  - `parse_resume(text)` returns a dict with `name`, `contact_info`, `experience`, `education[]`, `skills[]`
- **Batch processing**
  - `process_files_in_folder(folder_path)` loops over supported files and aggregates results
  - `write_to_json_file(data, output_file)` writes **UTF‑8** pretty JSON

### Output Schema (example)
```json
{
  "resumes": [
    {
      "name": "Ваховська Віра",
      "contact_info": {
        "phones": ["+111111111111"],
        "emails": ["example@email.com"]
      },
      "experience": "…raw text or structured blocks, depending on matches…",
      "education": [
        {
          "institution": "Хмельницький національний університет",
          "specialization": "Інженерія ПЗ",
          "graduation_year": "2026"
        }
      ],
      "skills": ["Python", "spaCy", "NLP"]
    }
  ]
}
```

---

## Quick Start

### Option A — Google Colab (recommended for the current notebook)
1. Open `CHI_NLP_CVs_Vakhovska.ipynb` in **Google Colab**.
2. **Mount Google Drive** in the first cell.
3. Ensure your CVs are in a folder like:
   ```
   /content/drive/MyDrive/Colab Notebooks/files/CVs
   ```
4. Run the **single-file demo** or the **batch processing** cell.
5. Find `output.json` (or `resume_output.json`) in the working directory or Drive path used in the notebook.

### Option B — Local (advanced; prototype is Colab‑centric)
You can adapt the notebook to run locally:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install spacy pdfplumber python-docx pandas
python -m spacy download uk_core_news_lg
```
Then convert notebook cells into a Python script and replace Drive paths with your local folders.

---

## Configuration

- **Input folder**: set the `folder_path` variable in the batch cell (defaults to a Drive path).
- **Single file demo**: update `single_file_path` to point to a specific PDF/DOCX/TXT.
- **Output path**: change the filename in `write_to_json_file(data, "output.json")`.

---

## Usage Examples

**Single CV demo (pseudocode from the notebook):**
```python
text = read_pdf(single_file_path)  # or read_docx / read_txt
parsed = parse_resume(text)
write_to_json_file({"resumes": [parsed]}, "resume_output.json")
```

**Batch processing a folder:**
```python
results = process_files_in_folder(folder_path)
write_to_json_file({"resumes": results}, "output.json")
```

---

## Assumptions & Notes

- If a field is not found, it’s set to `null`/`None` in JSON.
- Name detection prioritizes **spaCy NER** with a Ukrainian model; regex/patterns are used as fallback.
- Dates in experiences may be formatted differently (e.g., “Дотепер”, “Теперішній час”). The current regex covers typical forms.
- Skills and responsibilities are parsed from **bullets (`•`)** or **dashes**; consistency of source files affects quality.
- The notebook focuses on **Ukrainian** CVs; mixed‑language resumes might require additional patterns/models.

---

## Troubleshooting

- **`OSError: [E050] Can't find model 'uk_core_news_lg'`**  
  Run: `python -m spacy download uk_core_news_lg` (or in Colab, add a cell to download before use).
- **Empty text from PDF**  
  The file may be **scanned**. Use OCR (not implemented in the prototype) or provide a text‑based version.
- **Weird symbols / encoding**  
  Ensure UTF‑8 when saving TXT files. `python-docx` handles DOCX text reliably; PDFs depend on their structure.
- **No results for sections**  
  Adjust section keywords (e.g., add synonyms) or verify headings in the source files.

---

## Audit & Modernization Plan (Preview)

**What to improve (high level):**
- **Colab coupling**: hard‑coded Drive paths and notebook‑only workflow.
- **Parsing robustness**: limited handling for **scanned PDFs**, inconsistent headings, and varied date formats.
- **Schema guarantees**: no strict schema validation; fields may vary across CVs.
- **Testing & quality**: no unit tests, type hints only partially used, limited error handling and logging.
- **Distribution**: no CLI/API, packaging, or Docker; difficult to reuse at scale.
- **Language coverage**: tuned for **Ukrainian** only; no mixed‑language detection or fallback.
- **Determinism**: spaCy model versions not pinned; results can drift.

**Proposed direction (condensed):**
- **Refactor into a package** (`src/`), keep notebook for demos only.
- **CLI with Typer**, **REST API with FastAPI**, **Pydantic models** for strict JSON schema.
- **Robust PDF pipeline**: `pdfplumber` → fallback to **OCR** (Tesseract+`pytesseract`) for scanned files.
- **Section detection**: configurable keyword lists; **dateparser** for natural date parsing.
- **NLP**: pin `spaCy` & `uk_core_news_*`; add **rule‑based matchers** for names; consider **language identification** for mixed CVs.
- **Quality**: `pytest`, fixtures with sample CVs, `ruff`/`black`, structured logging.
- **Ops**: Dockerfile, `requirements.txt`/`pyproject.toml`, GitHub Actions CI, reproducible environments.

> A detailed audit with prioritized tasks and a step‑by‑step re‑implementation guide can be provided next.

---

## Roadmap (Suggested)
1. Extract notebook logic to a `src/` package and add a CLI.
2. Add Pydantic schemas + validation + JSON schema export.
3. Implement OCR fallback + better date & section parsing.
4. Build unit tests with a small anonymized CV fixture set.
5. Ship a FastAPI service + Docker images and examples.
6. Add language detection & bilingual parsing support.

---

## License
Add a license (e.g., MIT) appropriate for your use case.

---

## Acknowledgments
- **spaCy** (`uk_core_news_lg`) for Ukrainian NER and tokenization.
- **pdfplumber** and **python-docx** for document parsing.

---

## Notes from me now — what I should have done differently (approach to get the best results)

**Intent:** Turn a Colab prototype into a **reliable CV-to-JSON extractor** for Ukrainian (and mixed-language) resumes with measurable quality, easy reuse, and predictable outputs.

### 1) Start with the **contract**, not the code
- Define a **strict JSON schema** (required/optional fields, allowed enums, formats for dates/phones/emails).
- Write **acceptance criteria** per field (e.g., *“Full name must include last and first; patronymic optional; ≥95% exact match on gold set”*).
- Publish the schema (JSON Schema/Pydantic) and keep the extractor **schema-first** (fail fast on invalid outputs).

### 2) Build a **gold dataset** & evaluation harness
- Create a small but diverse **gold set** (20–50 CVs) covering: PDF (text), PDF (scanned), DOCX, TXT, Europass, LinkedIn style, bilingual UA/EN.
- Store **ground‑truth labels** for name, contacts, experience blocks, education entries, skills.
- Implement a **scorer**: precision/recall/F1 per field, plus document‑level pass rate. Let metrics drive changes.

### 3) Make parsing **layout‑aware**
- Don’t flatten to plain text first. Parse **layout blocks** (headings, paragraphs, tables, bullet lists).
- Keep **positions and structure** (section titles, bullet markers, indentation) so rules/ML can use them.
- Use a pipeline like: **PDF/DOCX → layout blocks → normalized text spans** (preserve bullet/heading metadata).

### 4) Use a **hybrid extraction** strategy (rules + ML + LLM, with guardrails)
- **Deterministic rules** for high‑precision items (emails, phones, dates with `dateparser`, UA phone validation).
- **NER/sequence models** for names/organizations/locations (fine‑tune or adapt a Ukrainian‑friendly model; add rule‑based matchers for patronymics and uppercase surnames).
- **LLM fallback** (only for hard cases): ask for **structured JSON** constrained by your schema; validate and retry if invalid. Keep this as an optional tier to control cost/latency.
- Combine signals with a **confidence score per field** and define **fallbacks** (e.g., if NER < 0.6, prefer rule match).

### 5) Normalize & validate early
- **Normalize Unicode**, fix hyphens/quotes, strip multiple spaces.
- **Validate phones** (region=UA), **emails**, and **dates** (ranges must make sense; end ≥ start).
- **Skill normalization**: map synonyms to a **canonical tech dictionary** (e.g., “JS” → “JavaScript”, “Postgres” → “PostgreSQL”).

### 6) Externalize **configuration**
- Maintain **section keyword lists** (e.g., “Досвід”, “Проєкти”, “Work Experience”, “Experience”) in YAML/TOML.
- Keep **regexes and thresholds** configurable; no literals hard‑coded in notebooks.
- Allow **templates** for common formats (Europass, LinkedIn export) with pluggable extractors.

### 7) Plan for **scanned PDFs** from day one
- Add **OCR** fallback (Tesseract/DocTR) with language hints (ukr+eng).
- Heuristics: if text extraction yields too few characters or too many spaces → trigger OCR.
- Post‑OCR cleanup (spell rules, common ligature fixes) before extraction.

### 8) Ship **as a package + CLI + (optional) API**
- Structure as `src/` package, expose a **CLI** (`cvparse folder/ --out out.json --ocr --lang auto`).
- Optional **FastAPI** endpoint for batch jobs; Dockerize for reproducibility.
- Pin versions, lock dependencies; provide a `Makefile` or simple `tox`/`nox` tasks.

### 9) Add **tests, fixtures, and monitoring**
- **Unit tests** for each extractor, **integration tests** for end‑to‑end runs.
- Store anonymized **fixtures** in `tests/fixtures/` (with paired ground truth).
- Produce **error reports**: which fields fail, examples, and a **confusion log** to guide rule/NER refinements.

### 10) Privacy & governance
- Never commit raw CVs; keep them encrypted or anonymized.
- Add a **redaction step** for PII in logs. Provide **consent**/usage notes in README.

---

### Best‑practice blueprint (minimal‑code view)

**Ingestion**
- Try text extraction (PDF/DOCX/TXT). If likely scanned or too sparse → run OCR.
- Convert to **blocks** (title/heading/paragraph/bullet/table). Keep order and hierarchy.

**Sectioning**
- Use a **heading classifier** (keyword + simple ML) to segment sections (Experience, Education, Skills, Contacts).

**Extraction**
- Apply **rules first** (emails/phones/dates).  
- Run **NER** for PERSON/ORG/EDU entities.  
- If critical fields are missing/low‑confidence → **LLM fallback** with schema‑constrained JSON.

**Normalization & validation**
- Canonicalize skills and organizations; validate formats; compute **confidence per field**.

**Output**
- Emit schema‑valid JSON; attach **provenance** (which method produced each field) and **confidence**.

**Quality loop**
- Evaluate on the gold set after each change; track F1/accuracy. Only merge if **metrics improve**.
