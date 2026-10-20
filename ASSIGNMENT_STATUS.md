# Progress on the ML Engine Deliverable (Questions 1–3)

**Chosen approach:** Retrieval-Augmented Generation (RAG). The pipeline is:

1. A sentence-embedding model embeds every skill node.
2. Cosine-similarity / k-NN matching links the student's free-text goal to the closest career paths.
3. The nodes for that path are retrieved.
4. A skill-gap filter uses a similarity threshold to drop skills the student already has.
5. An LLM orders the remaining skills into stages and explains each one.

There are no labels in the data, so this is not a classification, regression or CNN task.

This is a check of what is in this folder (`roadmap_dataset.csv`, `preprocess_roadmap.py`, `out/`, `README.md`)
against the three questions, as of 2026-10-20.

| Q | Asks for | Status | Evidence |
|---|---|---|---|
| 1 | Data cleaning from exploration to final dataset, critically detailed, with ≥ 4 EDA observations | **Done in the code. Written up** in `Q1_DATA_CLEANING.md`. Still needs to go into a notebook | `preprocess_roadmap.py`, `out/preprocessing_log.txt`, `Q1_DATA_CLEANING.md` |
| 2 | Modelling approach on a base model, plus validity checks | **Not done.** No embeddings, retrieval code or results yet | — |
| 3 | Other models compared with the base model, as in Figure 6 of the paper | **Not done** | — |

> **`README.md` is out of date.** Sections 4 and 5 describe *fine-tuning* Qwen2.5 on `train.jsonl`, which is the
> old plan. Under RAG the model is not trained on `train.jsonl`. Update those sections, or label them
> "alternative approach", so the report does not contradict itself.

---

## Question 1: Data cleaning and EDA

**What was already there:**

- A cleaning script with 10 logged steps, from 10,634 raw rows to 10,405 clean rows
- Row counts for each step
- Summary statistics

**What `Q1_DATA_CLEANING.md` adds:**

- **Exploration before cleaning.** A raw missing-value table and nine EDA checks (E1–E9), each linked to the
  cleaning step it led to.
- **7 observations,** each with numbers and what it means for RAG:
  - uneven resource coverage
  - path-size imbalance, which biases k-NN career matching
  - skills repeated across paths, which requires de-duplicating search results by `skill_id`
  - short descriptions, so one node fits in one vector with no chunking
  - concentrated scrape noise
  - generic titles, which means the description must be embedded along with the title
  - **no learning order in the data**, which is why the LLM is needed for ordering
- **A critical evaluation of strengths and limitations.** It includes two issues found during this check:
  17 descriptions still contain leftover link text, which would end up in the embeddings, and the whole
  `code-review` path was dropped because all of its rows are missing `node_id`.

**Still to do for Q1:** the question says *"in your notebook/development environment"*. Make a notebook
(e.g. `01_eda_cleaning.ipynb`) that loads the raw CSV, shows the EDA tables and plots, runs the cleaning,
and includes the text from `Q1_DATA_CLEANING.md` as markdown cells.

---

## Question 2: What it asks, and what it means under RAG

> *Describe the modelling approach taken on the identified base model. Ensure you have applied appropriate
> measures … to verify the validity of the results.*

Question 2 has two parts:

1. **"The modelling approach on the base model."** Explain your main pipeline and the reasons for each choice.
   Here the **base model is the embedding model plus k-NN retrieval** (e.g. `all-MiniLM-L6-v2` with cosine
   similarity). Describe:
   - what text is embedded (title + description)
   - how career matching works (e.g. one mean vector per path; see Q1 Observation 2)
   - how skill gaps are filtered (the similarity threshold)
   - how the LLM orders and explains the retrieved skills (the prompt, and the instruction to use only the
     nodes it is given)
2. **"Measures to verify the validity of the results."** Show that the system's output can be trusted, not just
   that it runs. The question's examples (elbow method for K-means, cross-validation for classification,
   regularisation for regression) are standard checks for those model types. Here are the equivalent
   checks for RAG:

| Check | What it proves | How to do it with this dataset |
|---|---|---|
| **Retrieval accuracy (Hit@k, MRR)** | The embeddings actually find the right career | Use the nodes in `test.jsonl` as queries ("does this skill's text retrieve its own `career_path` in the top-k?"). Add a small hand-written set of about 50 student goals with expected paths |
| **Choosing k and the threshold with a curve** (the "elbow" equivalent) | `k` and the skill-gap threshold were chosen with evidence, not guessed | Plot Hit@k against k, and precision/recall against the similarity threshold on `val`. Pick the knee of each curve, then report the result on `test` |
| **Tune on val, report on test** (the cross-validation equivalent) | The chosen settings are not over-fitted to the evaluation data | The grouped `val`/`test` split already exists. Optionally use 5-fold over title groups and report mean ± std |
| **Cluster sanity check** | Nodes from the same path really are close in the embedding space | Silhouette score of embeddings grouped by `career_path`, plus a t-SNE or UMAP plot |
| **Groundedness of the LLM output** | The LLM only uses retrieved skills and does not invent new ones | Count the share of roadmap items that match a retrieved node. Should be ~100% |
| **Ordering quality** | The fundamentals → intermediate → projects order makes sense | Hand-label stages for 2–3 paths and compare them with the LLM's order (Kendall's τ), or score with a short rubric |

**Status: not done.** Nothing has been embedded or evaluated yet. To answer Q2: build the pipeline, run the
checks above, and include the curves and the results table.

---

## Question 3: Other models to validate the base model

> *Develop models that follow the same process and compare performances (see Figure 6 of the paper).*

Under RAG, this means **running the same retrieval pipeline and the same metrics with different models**, then
showing them in one table and one bar chart, as in Figure 6:

| Comparison model | Why include it |
|---|---|
| **TF-IDF or BM25** (keyword search, no neural model) | A lower bound. If MiniLM can't beat keyword search, the embedding model adds nothing |
| **`all-MiniLM-L6-v2`** (base model) | Small and fast, which suits a mobile backend |
| **`BAAI/bge-small-en-v1.5`** | Similar size, newer, and often stronger on retrieval benchmarks |
| **`all-mpnet-base-v2`** or **`bge-base`** | Larger model. Shows whether the extra size is worth the slower speed |
| *Optional:* 2 LLMs for the ordering step (e.g. a small open model vs a hosted model) | Compare groundedness, ordering score and latency |

Report Hit@k, MRR, silhouette, and **latency per query / model size**, since the app runs on a phone.

**Status: not done.**

---

## Summary

- **Q1:** the cleaning is solid, and the write-up now explains it in RAG terms. Put it in a notebook with charts.
- **Q2:** build the embedding + k-NN base model, then validate it with Hit@k/MRR, the curves for k and the
  threshold, the val/test split, cluster checks and groundedness. Not started.
- **Q3:** compare TF-IDF/BM25, MiniLM, bge-small and a larger model on the same metrics. Not started.
- **Fix `README.md` §4–5**, which still describe fine-tuning.
