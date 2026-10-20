# Question 1: Data Cleaning, from Exploration to the Final Modelling Dataset

> **Question 1.** *Critically detail the data cleaning process taken from exploration to the final dataset
> for modelling. As part of the EDA, derive at least 4 observations from the dataset.*

**Project:** TechPath, an ML + mobile app that gives students a personalised tech career roadmap.
**ML approach:** Retrieval-Augmented Generation (RAG).
1. A sentence-embedding model (e.g. `all-MiniLM-L6-v2` or `bge-small`) turns every skill node into a vector.
2. The student's free-text goal is matched to the closest career paths with cosine similarity / k-nearest neighbours.
3. That path's nodes are retrieved, and skills the student already has are filtered out by a similarity threshold.
4. An LLM puts the remaining nodes into stages (fundamentals → intermediate → projects) and explains each one.

There are no labels in the data (no "ready / not ready" or score column), so this is not a classification or
regression task. The cleaned dataset is the **knowledge base** that gets embedded and searched.

**Source data:** `roadmap_dataset.csv`, a scrape of roadmap.sh.
**Code:** `preprocess_roadmap.py` (deterministic, seed 42). Row counts per step are in `out/preprocessing_log.txt`.

---

## 1.1 The raw dataset

| Property | Value |
|---|---|
| Rows | 10,634 (one row = one skill node on one career roadmap) |
| Columns | 10: `career_path`, `topic_slug`, `node_id`, `title`, `description`, `courses`, `articles`, `official_docs`, `videos`, `opensource` |
| Career paths | 93 |
| Data types | All text. The five resource columns hold `Title (url) \| Title (url) \| …` strings |

**Missing values in the raw file**

| Column | Missing | % of rows | What it means |
|---|---|---|---|
| `title` | 2 | 0.02% | Node cannot be named or embedded |
| `node_id` | 9 | 0.08% | Node cannot be joined to its resources |
| `topic_slug` | 19 | 0.18% | Recoverable from the title |
| `description` | 29 | 0.27% | Nothing to embed or show the student |
| `courses` | 10,335 | 97.2% | Most nodes have no course link. This is expected, not an error |
| `opensource` | 10,187 | 95.8% | Expected |
| `videos` | 6,337 | 59.6% | Expected |
| `official_docs` | 5,987 | 56.3% | Expected |
| `articles` | 3,634 | 34.2% | Expected |

Missing resource cells are **structural**: they mean "no link of this type", so they were not imputed or dropped.
Missing identity fields (`title`, `node_id`) and missing descriptions are real defects.

---

## 1.2 Exploration: what problems the EDA found

Each check below was run on the raw file before any cleaning. Each problem found sets up one cleaning step in 1.3.

| # | Check | Finding | Cleaning step it led to |
|---|---|---|---|
| E1 | Nulls in key fields | 10 rows missing `title` or `node_id` | Step 1 |
| E2 | Duplicates on `(career_path, node_id)` | 66 extra rows, 36 of them in `leetcode` (scrape repeats) | Step 2 |
| E3 | Path-level overlap | `r-programming` and `r` share **all 129 of 129** `node_id`s, so one is an alias of the other | Step 3 |
| E4 | Markup inside descriptions | HTML tags in 161 rows, markdown bold in 338, markdown links in 104 | Text cleaning (Step 4) |
| E5 | Description length | Min 0, median 400, max 3,647 characters. Some are empty or a few words long | Step 4 |
| E6 | Missing `topic_slug` | 19 rows | Step 5 |
| E7 | Same skill in many paths | e.g. *Git* appears in 16 paths and *Python* in 15, often with the same text | Step 6 |
| E8 | Broken URLs | 6 resource cells have a cut-off scheme (`ttps://…`) | Step 7 |
| E9 | Row order within a path | Rows are sorted by `topic_slug` (98% of neighbouring rows are in alphabetical order), **not** learning order | No cleaning step. The LLM handles ordering (Observation 7) |

---

## 1.3 The cleaning pipeline, step by step

| Step | Action | Rows in → out | Justification |
|---|---|---|---|
| 0 | Load raw CSV as strings; strip whitespace from every cell | 10,634 → 10,634 | Reading everything as strings stops pandas guessing types, and stripping stops `" git"` and `"git"` being treated as different values |
| 1 | Drop rows missing `title` or `node_id` | 10,634 → **10,624** | Without a title there is nothing to embed or display. Without a node_id the row cannot be linked to resources. Neither can be reliably imputed |
| 2 | Drop duplicate `(career_path, node_id)`, keep first | 10,624 → **10,563** | Duplicate vectors would show up twice in search results and give some skills extra weight |
| 3 | Drop the alias path `r-programming` | 10,563 → **10,434** | It duplicates `r` exactly (129/129 nodes). Keeping it would offer the student two identical careers and split the career-matching scores between them |
| 4 | Clean description text, then drop descriptions under 40 characters | 10,434 → **10,405** | Cleaning: Unicode NFC normalisation; remove HTML tags, inline-code ticks, bold/italic markers and scrape residue ("Visit the following resources…"); turn markdown links into plain text; collapse whitespace. Markup and URLs add noise to embeddings and pull unrelated nodes closer together. Under 40 characters, a description has too little meaning to give a useful vector |
| 5 | Impute missing `topic_slug` from the title (`slugify`) | 18 imputed | The slug is fully determined by the title, so imputing it is lossless |
| 6 | Add path metadata and a canonical `skill_id` (SHA-1 of title + description) | 336 rows share a `skill_id` | `parent_path` maps `*-beginner` paths to their main path. `path_type` labels each path `role` or `skill`, which is used for the career menu and matching. `skill_id` marks the same skill appearing in several paths, which is used to remove duplicate search results and to filter skill gaps across paths |
| 7 | Parse the five resource columns into a long table, one row per link | → **23,725** link rows | Repair `ttps://` to `https://`, drop duplicate URLs within a node, and extract the domain. Links are kept **out** of the embedded text and are never generated by the LLM. The app shows them straight from this table, so URLs cannot be invented |
| 8 | Build a text field per node (`prompt` / `target`) | 10,405 | `target` = cleaned description. For RAG, the text to embed is `title + ". " + description` (Observation 6 explains why the title is included) |
| 9 | Grouped 80/10/10 split by normalised title (seed 42) | 8,317 / 1,006 / 1,082 | RAG does not need a training split, because the whole cleaned table is the knowledge base. The split is kept for **evaluation**: `val` is used to tune `k` and the skill-gap threshold, and `test` gives the final retrieval score. Grouping by title means the same skill name is never in both the tuning set and the test set. Checked: every title group sits in exactly one split |

**Final dataset for modelling**

| File | Rows | Role in the RAG system |
|---|---|---|
| `out/nodes_clean.csv` | 10,405 | **The knowledge base.** Each row is embedded once (title + description). `career_path`, `path_type`, `skill_id` and `node_id` are stored as metadata alongside the vector |
| `out/path_taxonomy.csv` | 91 | Career-path list and menu (34 role, 57 skill), and the set of matching targets |
| `out/resources_clean.csv` | 23,725 | "Learn more" links, looked up by `node_id` / `skill_id` after retrieval |
| `out/val.jsonl`, `out/test.jsonl` | 1,006 / 1,082 | Evaluation queries: tune on val, report on test |
| `out/train.jsonl` | 8,317 | Not needed for RAG. Kept in case the LLM is fine-tuned later |

Overall: 229 of 10,634 rows removed (2.2%). 91 paths remain (34 role, 57 skill) with 10,228 unique skills.

---

## 1.4 EDA observations

**Observation 1: Resource coverage is uneven, and video links are concentrated on one site.**
Articles appear in 66% of raw rows, but courses appear in only 2.8% and open-source links in 4.2%. After
parsing, the 23,725 links are 48.2% articles, 26.8% official docs, 21.6% videos, 1.9% open source and 1.5% courses.
YouTube alone accounts for **21.9%** of all links. **982 nodes (9.4%) have no link at all.**
*Implication:* the roadmap will show no "Learn more" links for about 1 in 10 skills. The fallback is to borrow
links from the same `skill_id` in another path.

**Observation 2: Career paths vary a lot in size.**
Nodes per path range from **5** (`git-github-beginner`) to **301** (`cyber-security`), with a median of 112.
Skill/technology paths provide 6,331 nodes (61%) and role paths 4,074 (39%).
*Implication for career matching:* if a path is chosen by counting how many of the student's top-k nearest
nodes belong to it, large paths will win just because they have more nodes. Matching should compare the
student's goal with **one vector per path** (e.g. the mean of its node vectors), or normalise the votes by path size.

**Observation 3: Many skills repeat across paths.**
The most common titles are *Introduction* (24 paths), *Testing* (20), *Git* and *GitHub* (16 each), *JavaScript*
and *Python* (15). 336 rows have the same title **and** description as a row in another path. Another 1,034
title groups use the same name with different wording.
*Implication:* a search without de-duplication returns "Git" several times. Results should be de-duplicated by
`skill_id`. Repeats are also useful: if a student already knows Git, that one `skill_id` can be marked as known
in every path at once.

**Observation 4: Descriptions are short, fairly uniform and right-skewed.**
After cleaning, descriptions have a median of **400 characters / 61 words** (IQR 48–77 words). The longest is
3,572 characters, and only 9 exceed 1,500.
*Implication:* title + description is about 68 words on average. Only about 32 nodes go past the 256-token
input limit of `all-MiniLM-L6-v2`, so truncation hardly affects embedding quality. No chunking is needed:
one node = one vector.

**Observation 5: The raw scrape is noisy, and the noise is concentrated.**
66 duplicate rows (36 in `leetcode`), a whole duplicated path (`r-programming`), markup in hundreds of
descriptions, and 6 truncated URLs. The noise sits in specific paths rather than being spread randomly, which
points to scraper behaviour on particular pages rather than bad source content.

**Observation 6: Some titles are generic.**
Titles such as *Introduction*, *More Exercises* and *index* mean different things in each path.
*Implication:* embedding the title alone would make these nodes look identical. The vector must include the
description, and optionally the career path, so that "Introduction" in `docker` and in `python` end up far apart.

**Observation 7: The dataset holds no learning order or prerequisites.**
Within each path, rows are sorted by `topic_slug` (98% of neighbouring rows are in alphabetical order). For
example, `backend` starts with *ACID, AI Agents, AI-Assisted Coding, …, Apache*. There is no column for
order, level or prerequisite.
*Implication:* the data alone cannot tell that version control comes before APIs. This is why the LLM is
needed to arrange the retrieved nodes into stages. Row position must **not** be used as a learning order.

---

## 1.5 Critical evaluation of the cleaning

What works well:

- **Every step can be reproduced and audited.** Each step logs rows in and out, and the seed is fixed.
- **The knowledge base has no duplicates** at the row, path and skill level, so search results are not repeated.
- **Links are never embedded or generated.** They come only from the cleaned table, so the LLM cannot invent URLs.
- **Little data was lost** (2.2% of rows), and each removal has a documented reason.

Limitations, and how they could be fixed:

| Limitation | Evidence | Possible fix |
|---|---|---|
| Some scrape residue survives cleaning, and it would end up inside the embeddings | 17 cleaned descriptions still contain `http`, and 6 contain "Learn more from the following resources: - @article@…" | Extend the residue regex to cover `Learn more from…` and `@type@` markers before embedding |
| A whole path was lost | All 6 `code-review` rows have no `node_id`, so Step 1 removed the path, and students cannot be matched to it | Build a synthetic id from `career_path + topic_slug` instead of dropping these rows |
| The 40-character cut-off is a judgement call | Chosen by inspecting short rows, not tuned | Check retrieval scores with cut-offs of 20, 40 and 80 characters |
| There is no learning-order signal | Observation 7 | Rely on the LLM for ordering. Optionally add a hand-labelled `stage` for a few paths to check the LLM's ordering |
| Path sizes are imbalanced | Path sizes range from 5 to 301 | Match against one vector per path, or normalise votes (Observation 2) |
| There are no labelled student queries | The dataset has nodes only, with no examples of real student inputs | Write a small labelled set (e.g. 50 goals → expected career paths) to evaluate career matching |
| No charts in the deliverable | The statistics above are tables only | Add histograms (description length, nodes per path) and bar charts (resource types) in the notebook |
