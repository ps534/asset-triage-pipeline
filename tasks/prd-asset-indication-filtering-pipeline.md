# Asset‑Indication Filtering Pipeline – Product Requirements Document (PRD)

## 1. Introduction / Overview

Business‑development (BD) analysts routinely receive spreadsheets containing thousands of clinical‑stage assets. Manually triaging these lists is slow and error‑prone. This feature delivers an **AI‑assisted pipeline** that ingests a simple two‑column asset list (asset name, company name), runs three chained LLM prompts plus live web search, and produces a new worksheet that:

* Expands each asset into its primary indication **and** plausible repurposing indications.
* Adds machine‑readable filter columns (`pursue`, `fail_reasons`, `degree_of_unmet_need`, etc.).
* Stores a one‑line rationale and any error details.

The goal is to surface a **\~10 % keep‑rate** with negligible (< 1 %) false‑negative discard errors, enabling analysts to focus diligence on the highest‑potential assets.

---

## 2. Goals

1. **Signal‑to‑Noise:** Reduce candidate list by \~90 %, retaining ≤ 10 % of assets for deeper review.
2. **Accuracy:** Achieve < 1 % false‑negative rate when spot‑checking discarded items.
3. **Throughput:** Process ≥ 7 000 assets (≈ 20 000 asset‑indication pairs) in ≤ 4 hours on a local development environment.
4. **Usability:** Output a human‑readable worksheet: one row per *asset–indication* pair with clear Yes/No flags and concise rationale.
5. **Prototype Simplicity:** Run via a CLI script (executed manually) without external DB dependencies.

---

## 3. User Stories

| # | User Story                                                                                                                            |
| - | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | **As a BD analyst,** I want the sheet to flag which assets are worth a closer look so that I can focus my diligence time effectively. |
| 2 | **As a BD analyst,** I want to see a short explanation for each decision so that I can understand the model’s reasoning at a glance.  |
| 3 | **As a prototype operator,** I want to run the script on demand with a local Excel file and get a new tab added to that file.         |
| 4 | **As a BD lead,** I want confidence that discarded assets are almost never false negatives so that I can trust the automation.        |

---

## 4. Functional Requirements

1. **Input Parsing**
   1.1 The system **must** accept an `.xlsx` or .csv file containing at least `Asset Name`, `Company Name` columns.
   1.2 The system **must** deduplicate identical asset/company rows.
2. **Prompt A – Repurposing Enumerator**
   2.1 For each asset, the system **shall** call an LLM (GPT‑o3 via LiteLLM) with a prompt that returns up to 5 plausible repurposing indications, each labelled `High | Medium | Low` plausibility.
3. **Prompt B – Unmet‑Need Rank Lookup**
   3.1 For every unique indication (primary + repurposed), the system **shall** look up its unmet‑need score locally (CSV lookup) and bucket it relative to the mean and standard deviation.
   3.2 If an indication string is not an exact match, the system **shall** perform deterministic fuzzy matching (≥ 97 token‑sort ratio) before failing.
4. **Prompt C – Asset Screen**
   4.1 For each *asset–indication* pair, the system **shall** run the pursue/not‑pursue prompt (with web search enabled) and obtain a JSON blob with `pursue` boolean, `fail_reasons` list, and `info_confidence %`.
   4.2 The system **shall** retry once on model errors/timeouts.
5. **Spreadsheet Generation**
   5.1 Output a new worksheet named `Filtered` in the same workbook.
   5.2 **Columns:** `Asset Name`, `Company`, `Indication`, `Pursue` (Yes/No), `Fail Reasons`, `Degree of Unmet Need`, `Repurposing (if from Prompt A)`, `Rationale`, `Error`.
6. **Error Handling**
   6.1 On unrecoverable errors for a row, the system **must** set `Pursue = "Error"` and populate the `Error` column with details.
7. **CLI Usage**
   7.1 Running `python filter_assets.py input.xlsx` **shall** process the file and save it in‑place (with backup copy).
   7.2 A `--dry‑run` flag **shall** output a CSV preview to stdout instead of modifying the file.

---

## 5. Non‑Goals (Out of Scope)

* Real‑time web dashboard or front‑end UI.
* Continuous integration into data‑warehouse pipelines.
* Fine‑tuning or custom training of LLMs.
* Guaranteeing deterministic identical output across model versions.

---

## 6. Design Considerations (UI / UX)

* Keep new worksheet tab visually similar to the input sheet (font, row height) for analyst comfort.
* Use conditional formatting (green = pursue, red = don’t pursue, yellow = error) — optional stretch.

---

## 7. Technical Considerations

* **Language & Runtime:** Python 3.11, `pandas`, `openpyxl`, `litellm` wrapper.
* **Prompt‑Caching:** Enabled via LiteLLM cache to avoid duplicate calls for identical prompts within a run.
* **Search:** Use OpenAI web‑search plugin (API key via env var `OPENAI_API_KEY`).
* **Fuzzy Disease Mapping:** RapidFuzz ≥ 97‑score; unresolved strings bubble to error column.
* **Parallelism:** `asyncio.gather` with rate‑limit controls; target ≤ 20 concurrent calls.

---

## 8. Success Metrics

| Metric                      | Target                                                 |
| --------------------------- | ------------------------------------------------------ |
| **Keep‑rate**               | 8–12 % of asset‑indication rows marked `Pursue = Yes`. |
| **False‑negatives (audit)** | ≤ 1 % when manually checking 100 discarded rows.       |
| **Runtime (7 000 assets)**  | ≤ 4 h end‑to‑end on a local development env            |
| **Error rows**              | ≤ 2 % rows flagged `Error`.                            |

---

## 9. Open Questions

1. **Canonical Indication Table:** Will the unmet‑need CSV be provided by BD ops, and how often will it change? Yes - it could change every few weeks. 
2. **Repurposing Depth:** Should Prompt A ever suggest > 5 indications if the science strongly supports it? Yes
3. **Cost Guard‑Rails:** Do we need usage alerts (>\$‑threshold) as we scale beyond prototype? No
4. **Analyst Feedback Loop:** How will manual corrections feed back into synonym lists or prompt tweaks? Out of scope for now.

---

*Document version:* v0.1 – 2025‑05‑29
