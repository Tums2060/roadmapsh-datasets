# Question 1: Data Cleaning, from Exploration to the Final Modelling Dataset

> **Question 1.** *Critically detail the data cleaning process taken from exploration to the final dataset
> for modelling. As part of the EDA, derive at least 4 observations from the dataset.*

**Project:** TechPath, an ML + mobile app that explains the skills on a tech career roadmap to students.
**ML task:** supervised fine-tuning of a small language model. Input: a skill and a career path.
Output: a plain-language explanation of that skill.
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
| `title` | 2 | 0.02% | Node cannot be named, so it is unusable |
| `node_id` | 9 | 0.08% | Node cannot be joined to its resources |
| `topic_slug` | 19 | 0.18% | Recoverable from the title |
| `description` | 29 | 0.27% | No training target |
| `courses` | 10,335 | 97.2% | Most nodes have no course link. This is expected, not an error |
| `opensource` | 10,187 | 95.8% | Expected |
| `videos` | 6,337 | 59.6% | Expected |
| `official_docs` | 5,987 | 56.3% | Expected |
| `articles` | 3,634 | 34.2% | Expected |

Missing resource cells are **structural**: they mean "no link of this type", so they were not imputed or dropped.
Missing identity fields (`title`, `node_id`) and missing targets (`description`) are real defects.

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
| E7 | Same skill in many paths | e.g. *Git* appears in 16 paths and *Python* in 15, often with the same text | Steps 6 and 9 |
| E8 | Broken URLs | 6 resource cells have a cut-off scheme (`ttps://…`) | Step 7 |

---

## 1.3 The cleaning pipeline, step by step

| Step | Action | Rows in → out | Justification |
|---|---|---|---|
| 0 | Load raw CSV as strings; strip whitespace from every cell | 10,634 → 10,634 | Reading everything as strings stops pandas guessing types, and stripping stops `" git"` and `"git"` being treated as different values |
| 1 | Drop rows missing `title` or `node_id` | 10,634 → **10,624** | Without a title there is nothing to put in the prompt. Without a node_id the row cannot be linked to resources. Neither can be reliably imputed |
| 2 | Drop duplicate `(career_path, node_id)`, keep first | 10,624 → **10,563** | Duplicates over-weight some examples and can leak between splits |
| 3 | Drop the alias path `r-programming` | 10,563 → **10,434** | It duplicates `r` exactly (129/129 nodes). Keeping it would double-count R skills |
| 4 | Clean description text, then drop descriptions under 40 characters | 10,434 → **10,405** | Cleaning: Unicode NFC normalisation; remove HTML tags, inline-code ticks, bold/italic markers and scrape residue ("Visit the following resources…"); turn markdown links into plain text; collapse whitespace. The model should learn prose, not markup. Under 40 characters, a description is a fragment, not an explanation |
| 5 | Impute missing `topic_slug` from the title (`slugify`) | 18 imputed | The slug is fully determined by the title, so imputing it is lossless |
| 6 | Add path metadata and a canonical `skill_id` (SHA-1 of title + description) | 336 rows share a `skill_id` | `parent_path` maps `*-beginner` paths to their main path. `path_type` labels each path `role` or `skill`. `skill_id` marks the same skill appearing in several paths |
| 7 | Parse the five resource columns into a long table, one row per link | → **23,725** link rows | Repair `ttps://` to `https://`, drop duplicate URLs within a node, and extract the domain. Links are kept **out** of the training targets, so the model never learns to generate URLs |
| 8 | Build instruction pairs | 10,405 pairs | `prompt` = `Explain the skill "<title>" for a student pursuing the <path> path: what it is, what it involves and why it matters.` `target` = cleaned description |
| 9 | Grouped 80/10/10 split by normalised title (seed 42) | 8,317 / 1,006 / 1,082 | 8,206 title groups are shuffled and split, so a skill name such as "Git" is entirely in one split. A random row split would put "Git (backend)" in train and "Git (devops)" in test, which inflates test scores. Checked: every title group sits in exactly one split |

**Final dataset for modelling**

| File | Rows | Purpose |
|---|---|---|
| `out/train.jsonl` | 8,317 (79.9%) | Fine-tuning |
| `out/val.jsonl` | 1,006 (9.7%) | Model selection, early stopping, hyper-parameters |
| `out/test.jsonl` | 1,082 (10.4%) | Final score, used once |
| `out/nodes_clean.csv` | 10,405 | Master table (all fields, plus which split each row is in) |
| `out/resources_clean.csv` | 23,725 | Links shown in the app |
| `out/path_taxonomy.csv` | 91 | Career-path menu |

Overall: 229 of 10,634 rows removed (2.2%). 91 paths remain (34 role, 57 skill) with 10,228 unique skills.

The splits are well balanced. Median description length is 400 (train), 403 (val) and 391.5 (test) characters,
and the role/skill mix is similar in each split.

---

## 1.4 EDA observations

**Observation 1: Resource coverage is uneven, and video links are concentrated on one site.**
Articles appear in 66% of raw rows, but courses appear in only 2.8% and open-source links in 4.2%. After
parsing, the 23,725 links are 48.2% articles, 26.8% official docs, 21.6% videos, 1.9% open source and 1.5% courses.
YouTube alone accounts for **21.9%** of all links. **982 nodes (9.4%) have no link at all.**
*Implication:* the app's "Learn more" section will be empty for about 1 in 10 skills. The fallback is to
borrow links from the same `skill_id` in another path.

**Observation 2: Career paths vary a lot in size.**
Nodes per path range from **5** (`git-github-beginner`) to **301** (`cyber-security`), with a median of 112.
Skill/technology paths provide 6,331 nodes (61%) and role paths 4,074 (39%).
*Implication:* the training data is dominated by a few large paths. A single overall test score can hide weak
performance on small paths, so evaluation should also be broken down by `path_type` and path.

**Observation 3: Many skills repeat across paths, which creates a leakage risk.**
The most common titles are *Introduction* (24 paths), *Testing* (20), *Git* and *GitHub* (16 each), *JavaScript*
and *Python* (15). 336 rows have the same title **and** description as a row in another path. Another 1,034
title groups use the same name with different wording.
*Implication:* a random row split would let the model see near-identical answers in train and test. This finding
is the reason for the **grouped split** in Step 9.

**Observation 4: Descriptions are short, fairly uniform and right-skewed.**
After cleaning, descriptions have a median of **400 characters / 61 words** (IQR 48–77 words). The longest is
3,572 characters, and only 9 exceed 1,500.
*Implication:* almost every target fits in a 512-token training window with room to spare, so truncation affects
only a handful of examples. Generation length can be capped at about 200 new tokens.

**Observation 5: The raw scrape is noisy, and the noise is concentrated.**
66 duplicate rows (36 in `leetcode`), a whole duplicated path (`r-programming`), markup in hundreds of
descriptions, and 6 truncated URLs. The noise sits in specific paths rather than being spread randomly, which
points to scraper behaviour on particular pages rather than bad source content.

**Observation 6: Some titles are generic.**
Titles such as *Introduction*, *More Exercises* and *index* mean different things in each path. On its own, the
title is not enough for the model to know what to explain. The career path in the prompt supplies that context.

---

## 1.5 Critical evaluation of the cleaning

What works well:

- **Every step can be reproduced and audited.** Each step logs rows in and out, and the seed is fixed.
- **Leakage is controlled.** The grouped split stops the same skill appearing in both train and test.
- **The model is never trained on URLs**, so it cannot invent links. Links come only from the cleaned table.
- **Little data was lost** (2.2% of rows), and each removal has a documented reason.

Limitations, and how they could be fixed:

| Limitation | Evidence | Possible fix |
|---|---|---|
| Some scrape residue survives cleaning | 17 cleaned descriptions still contain `http`, and 6 contain "Learn more from the following resources: - @article@…" | Extend the residue regex to cover `Learn more from…` and `@type@` markers |
| A whole path was lost | All 6 `code-review` rows have no `node_id`, so Step 1 removed the path | Build a synthetic id from `career_path + topic_slug` instead of dropping these rows |
| The 40-character cut-off is a judgement call | Chosen by inspecting short rows, not tuned | Report how sensitive results are to 20, 40 and 80 characters |
| The grouped split is stricter than needed for generic titles | Every "Introduction" goes to one split even though each has different content | Group by `skill_id` for generic titles, or accept the stricter split as a conservative choice |
| `keep="first"` on duplicates is arbitrary | The duplicates were the same, so in practice no information was lost | Keep the version with the longest description |
| Imbalance between paths was not corrected | Path sizes range from 5 to 301 | Report results per path, or reweight or oversample small paths |
| No charts in the deliverable | The statistics above are tables only | Add histograms (description length, nodes per path) and bar charts (resource types) in the notebook |
