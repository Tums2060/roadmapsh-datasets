# Progress on the ML Engine Deliverable (Questions 1–3)

A check of what is in this folder (`roadmap_dataset.csv`, `preprocess_roadmap.py`, `out/`, `README.md`)
against the three questions, as of 2026-10-20.

| Q | Asks for | Status | Evidence |
|---|---|---|---|
| 1 | Data cleaning from exploration to final dataset, critically detailed, with ≥ 4 EDA observations | **Done in the code. Now written up** in `Q1_DATA_CLEANING.md`. Still needs to go into a notebook | `preprocess_roadmap.py`, `out/preprocessing_log.txt`, `Q1_DATA_CLEANING.md` |
| 2 | Modelling approach on a base model, plus validity checks | **Not done.** Only a code example exists | `README.md` §4 (untested code, no results) |
| 3 | Other models compared with the base model, as in Figure 6 of the paper | **Not done** | Nothing in the folder |

---

## Question 1: Data cleaning and EDA

**What was already there:**

- A cleaning script with 10 logged steps, from 10,634 raw rows to 10,405 clean rows (`preprocess_roadmap.py`)
- Row counts for each step (`out/preprocessing_log.txt`)
- A leakage-safe grouped train/val/test split (`out/*.jsonl`)
- Summary statistics, and short reasons for each step (`README.md` §2)

**What was missing, and has now been written in `Q1_DATA_CLEANING.md`:**

- **Exploration before cleaning.** A raw missing-value table and eight EDA checks (E1–E8), each linked to the
  cleaning step it led to.
- **At least 4 observations.** There are 6 (uneven resource coverage, uneven path sizes, skills repeated across
  paths, description length distribution, concentrated scrape noise, generic titles), each with numbers and an
  implication for the model.
- **The "critically" part.** A strengths and limitations table. It includes two issues found during this check:
  17 descriptions still contain leftover link text, and the whole `code-review` path was dropped because all 6
  of its rows have no `node_id`.

**Still to do for Q1:** the question says *"in your notebook/development environment"*. The cleaning lives in a
`.py` script, and there is no notebook with charts. To fully meet the question, make a notebook (e.g.
`01_eda_cleaning.ipynb`) that loads the raw CSV, shows the EDA tables and plots (description-length
histogram, nodes per path, resource-type bar chart), runs the cleaning steps, and includes the text from
`Q1_DATA_CLEANING.md` as markdown cells.

---

## Question 2: What it asks, in plain terms

> *Describe the modelling approach taken on the identified base model. Ensure you have applied appropriate
> measures … to verify the validity of the results.*

Question 2 has two parts:

1. **"The modelling approach on the base model."** Choose one main algorithm (the base model), train it on
   your cleaned data, and explain the choices: which model, why it suits the problem, how it was trained, and
   which settings were used.
2. **"Measures to verify the validity of the results."** Show that the score is real, not caused by luck,
   overfitting or leakage. The examples in the question are standard checks for each task type:
   - K-means clustering → **elbow method** (shows that the chosen number of clusters is justified)
   - Classification → **cross-validation** (shows the accuracy holds on different data subsets)
   - Regression → **regularisation** (shows the model is not overfitting)

   Your task is generative text (fine-tuning a language model), so none of those three applies directly. The
   equivalent checks for this task are:

| Validity check | What it proves | State in this project |
|---|---|---|
| Held-out validation set + early stopping (`load_best_model_at_end`) | Stops training before the model overfits | Configured in README §4.3, **not yet run** |
| Train vs validation loss curves | Visual evidence of overfitting or underfitting | **Not yet produced** |
| Grouped split by skill title | The test score is not inflated by memorised skills | **Done** (Step 9) |
| Baseline comparison: same model *without* fine-tuning | Shows that fine-tuning actually helped | Described in README §4.4, **not yet run** |
| Several metrics (ROUGE-L for word overlap, BERTScore for meaning) | The result does not depend on one metric | Code written, **not yet run** |
| Small hyper-parameter search on val (learning rate, epochs, LoRA rank) | The settings were chosen with evidence, not guessed. This is the closest equivalent to "elbow" or "cross-validation" here | **Not done** |
| *Optional:* repeat with 2–3 random seeds and report mean ± std | Stability, similar to what cross-validation gives | **Not done** |

**Status: not done.** `README.md` §4 has a sensible plan (Qwen2.5-0.5B-Instruct + LoRA, SFT, eval each
epoch, ROUGE-L and BERTScore), but nothing has been trained. There are no saved models, logs, loss curves or
scores in the folder. To answer Q2 you need to:

1. Run the fine-tuning (e.g. on Colab T4) and save the train/val loss log.
2. Plot the train vs val loss curves.
3. Score the **base model without fine-tuning** and the **fine-tuned model** on `test.jsonl`
   (ROUGE-L, BERTScore).
4. Write up the model choice, its settings, the curves, and the before/after table.

---

## Question 3: Other models to validate the base model

> *Develop models that follow the same process and compare performances (see Figure 6 of the paper).*

Figure 6 is presumably a side-by-side comparison of several models on the same metrics. For this project:
train or evaluate **2–4 other models on the same `train/val/test` files with the same metrics**, then show
them in one table and one bar chart. Possible choices:

| Comparison model | Why include it |
|---|---|
| Retrieval baseline (TF-IDF nearest neighbour: return the description of the most similar train title) | A simple non-neural lower bound. If the LLM can't beat it, fine-tuning adds little |
| `google/flan-t5-base` fine-tuned the same way | A different architecture (encoder-decoder vs decoder-only) |
| A larger decoder model, e.g. `Qwen2.5-1.5B-Instruct` or `TinyLlama-1.1B` + LoRA | Shows the effect of model size |
| Base model zero-shot (no fine-tuning) | Already required for Q2. Reuse it here |

Also report latency or model size, since the model has to run behind a mobile app.

**Status: not done.**

---

## Summary

- **Q1:** cleaning and splitting were done well in code. The write-up (`Q1_DATA_CLEANING.md`) now covers the
  EDA, 6 observations and the critical evaluation. Put it in a notebook with charts.
- **Q2:** asks you to train your main model and prove the score is trustworthy. There is a plan in `README.md`,
  but the training has not been run.
- **Q3:** asks you to compare the base model with other models on the same data and metrics. Not started.
