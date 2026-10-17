# TechPath — Processed roadmap.sh Dataset

This folder holds the raw roadmap.sh scrape, the script that cleans it, and the cleaned outputs used to
**(1) fine-tune the TechPath explanation model** and **(2) power the TechPath app** (roadmaps, career-path
menus, learning resources).

```
Dataset/
├── roadmap_dataset.csv        raw scrape (10,634 rows)
├── preprocess_roadmap.py      cleaning + splitting script
└── out/
    ├── train.jsonl            fine-tuning data        (8,317 pairs)
    ├── val.jsonl              validation data         (1,006 pairs)
    ├── test.jsonl             held-out test data      (1,082 pairs)
    ├── nodes_clean.csv        master table of skills  (10,405 rows)
    ├── resources_clean.csv    learning links          (23,725 rows)
    ├── path_taxonomy.csv      career paths            (91 rows)
    └── preprocessing_log.txt  row counts per step + summary stats
```

To regenerate everything (deterministic, seed 42):

```bash
cd "/home/tumaini/Documents/3.2 NOTES/AI/Dataset"
python preprocess_roadmap.py roadmap_dataset.csv out/
```

---

## 1. How the pieces fit together

```
                       roadmap_dataset.csv
                               │
                     preprocess_roadmap.py
                               │
        ┌──────────────────────┼─────────────────────────┐
        │                      │                         │
  TRAINING SIDE           MASTER TABLE              APP SIDE
  train / val / test  ◄── nodes_clean.csv ──►  path_taxonomy.csv
      .jsonl               (skill_id, node_id)    resources_clean.csv
        │                                               │
  fine-tuned model ──── explains a skill ────► app shows explanation
                                               + real links from resources_clean.csv
```

The model is only trained to **explain** a skill. It is never trained to produce links. Links always come from
`resources_clean.csv`, so the model can't make up URLs.

---

## 2. What the cleaning did (from `preprocessing_log.txt`)

| Step | Rows in → out | Why |
|---|---|---|
| 0 Load raw | 10,634 → 10,634 | |
| 1 Drop missing title / node_id | 10,634 → 10,624 | A node without these can't be identified |
| 2 Drop duplicate (career_path, node_id) | 10,624 → 10,563 | Scrape repeats |
| 3 Drop alias path `r-programming` | 10,563 → 10,434 | Identical to `r` (all 129 node_ids shared) |
| 4 Drop descriptions < 40 chars | 10,434 → 10,405 | Too short to explain anything |
| 5 Impute missing `topic_slug` | 18 imputed | Built from the title |
| 6 Assign canonical `skill_id` | 336 rows share one | Same skill appearing in several paths |
| 7 Parse resources to a long table | → 23,725 resource rows | One row per link |
| 8–9 Build pairs + grouped split | 8,317 / 1,006 / 1,082 | 80/10/10 over 8,206 title groups |

Text cleaning removed HTML tags, markdown bold/italic/code markers and links, and scrape residue such as
"Visit the following resources…", and collapsed extra whitespace.

Summary: 91 paths (34 role, 57 skill), 10,228 unique skills, median description 400 characters,
982 nodes with no resources, and 21.9% of links on YouTube.

---

## 3. The files in detail

### 3.1 `train.jsonl`, `val.jsonl`, `test.jsonl` — fine-tuning data

One JSON object per line, with two fields:

```json
{"prompt": "Explain the skill \"Acting / Tool Invocation\" for a student pursuing the ai agents path: what it is, what it involves and why it matters.",
 "target": "Acting, also called tool invocation, is the step where the AI chooses a tool and runs it to get real-world data or to change something. The agent looks at its current goal and the plan it just made. It then picks the best tool, such as a web search, a database query, or a calculator. ..."}
```

- `prompt` is the input the model receives: a skill name plus a career path.
- `target` is the answer the model should learn to give: the cleaned roadmap.sh description.

| File | Role | When you touch it |
|---|---|---|
| `train.jsonl` | The model learns from these | Every training epoch |
| `val.jsonl` | Checks the model during training (loss curves, early stopping, choosing learning rate / epochs / LoRA rank) | After each epoch |
| `test.jsonl` | Final, unbiased score for Chapter 4 | **Once**, after all tuning is finished |

**Why the split is grouped.** The split is done by normalised skill title, not by row. "Git" appears in
`android`, `backend`, `backend-beginner`, `blockchain`, … and *all* of those rows are in `test`. If some were
in `train`, the model could memorise the answer and the test score would be inflated. Grouping means
the test set measures how well the model explains skills it has never seen.

> `\/` and `’` in the files are standard JSON escapes for `/` and `'`. Any JSON reader decodes them.

### 3.2 `nodes_clean.csv` — the master table

One row per (career path, skill). Every other file can be traced back to it.

| Column | Meaning | Example |
|---|---|---|
| `skill_id` | 12-char hash of title + description; the same skill in different paths shares it | `2ac568203ef9` |
| `title_group` | Normalised title used for the grouped split | `acting tool invocation` |
| `split` | `train` / `val` / `test` | `train` |
| `career_path` | roadmap.sh path slug | `ai-agents` |
| `parent_path` | Beginner variants mapped to their main path | `backend-beginner` → `backend` |
| `path_type` | `role` (a job, e.g. backend) or `skill` (a technology, e.g. docker) | `skill` |
| `is_beginner_variant` | True for `*-beginner` paths | `False` |
| `topic_slug` | URL-style topic name | `acting--tool-invocation` |
| `node_id` | roadmap.sh's node id; the join key to resources | `sHYd4KsKlmw5Im3nQ19W8` |
| `title` | Skill name | `Acting / Tool Invocation` |
| `description` | Cleaned explanation | "Acting, also called tool invocation, is…" |
| `n_resources` | Number of links available for this node | `2` |
| `prompt`, `target` | The same pair that appears in the `.jsonl` files | |

**Used for:** drawing a career roadmap in the app (all nodes for one `career_path`), the descriptive
statistics in Chapter 3, and tracing any test example back to its path and resources.

### 3.3 `resources_clean.csv` — learning links

One row per link. Duplicate URLs within a node have been removed.

| Column | Example |
|---|---|
| `career_path` | `ai-agents` |
| `node_id` | `sHYd4KsKlmw5Im3nQ19W8` |
| `skill_id` | `2ac568203ef9` |
| `resource_type` | `articles` / `official_docs` / `videos` / `opensource` / `courses` |
| `title` | `What are Tools in AI Agents?` |
| `url` | `https://huggingface.co/learn/agents-course/en/unit1/tools` |
| `domain` | `huggingface.co` |

Breakdown: articles 11,437 · official docs 6,348 · videos 5,132 · open source 452 · courses 356.

**Used for:** the "Learn more" list shown under each explanation in the app. Join on `node_id`
(or on `skill_id` to collect links for the same skill from every path).

### 3.4 `path_taxonomy.csv` — career paths

| career_path | parent_path | path_type | nodes |
|---|---|---|---|
| ai-agents | ai-agents | skill | 101 |
| ai-engineer | ai-engineer | role | 190 |
| backend-beginner | backend | role | … |

**Used for:** the career-path dropdown in the app (group by `path_type`, hide or nest beginner variants
under `parent_path`), and as a summary table in the report.

### 3.5 `preprocessing_log.txt`

The row counts in section 2 above. Quote them in Chapter 3, section 3.6, so the cleaning can be reproduced.

---

## 4. Training the model — worked example

The task is **supervised fine-tuning (SFT)**: given `prompt`, generate `target`. The example below uses a
small instruction-tuned model with LoRA, which fits on a free Colab T4 GPU. Any similar small model works.

### 4.1 Install

```bash
pip install "transformers>=4.44" "trl>=0.12" peft datasets evaluate rouge_score bert_score accelerate
```

### 4.2 Load the splits

```python
from datasets import load_dataset

ds = load_dataset("json", data_files={
    "train": "out/train.jsonl",
    "validation": "out/val.jsonl",
    "test": "out/test.jsonl",
})
# TRL's prompt-completion format expects the field names "prompt" and "completion"
ds = ds.rename_column("target", "completion")
print(ds)   # train 8317 · validation 1006 · test 1082
```

### 4.3 Fine-tune with LoRA

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

BASE = "Qwen/Qwen2.5-0.5B-Instruct"          # small, fast; swap for a larger model if you have the GPU
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto")

cfg = SFTConfig(
    output_dir="techpath-explainer",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    eval_strategy="epoch",                  # evaluates on val.jsonl after every epoch
    save_strategy="epoch",
    load_best_model_at_end=True,            # keeps the epoch with the lowest validation loss
    metric_for_best_model="eval_loss",
    completion_only_loss=True,              # learn the answer only, not to repeat the prompt
    max_length=512,
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],          # test is NOT passed here
    processing_class=tok,
    peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM"),
)
trainer.train()
trainer.save_model("techpath-explainer/final")
```

What happens in each epoch:
1. The model reads each `prompt` from `train.jsonl` and tries to predict the `target` one token at a time.
2. The loss (how wrong those predictions were) updates only the small LoRA adapter weights.
3. At the end of the epoch, the loss on `val.jsonl` is computed. If it stops falling while training loss
   keeps dropping, the model is overfitting. `load_best_model_at_end` keeps the best epoch.

### 4.4 Evaluate once on the test set

```python
import evaluate, torch
from transformers import pipeline

gen = pipeline("text-generation", model="techpath-explainer/final", tokenizer=tok,
               device=0 if torch.cuda.is_available() else -1)

test = ds["test"]
preds = [gen(p, max_new_tokens=200, do_sample=False, return_full_text=False)[0]["generated_text"]
         for p in test["prompt"]]
refs = test["completion"]

rouge = evaluate.load("rouge").compute(predictions=preds, references=refs)
bert  = evaluate.load("bertscore").compute(predictions=preds, references=refs, lang="en")
print("ROUGE-L:", rouge["rougeL"])
print("BERTScore F1:", sum(bert["f1"]) / len(bert["f1"]))
```

Report the same metrics for the **base model with no fine-tuning** as a baseline. The gap between the two
is your result. ROUGE-L measures word overlap with the reference. BERTScore measures closeness in meaning,
which suits explanations that can be phrased in many ways.

### 4.5 Asking the trained model a question

```python
prompt = ('Explain the skill "Docker" for a student pursuing the devops path: '
          'what it is, what it involves and why it matters.')
print(gen(prompt, max_new_tokens=200, do_sample=False, return_full_text=False)[0]["generated_text"])
```

At inference time, always use **the same prompt template** as in training. The model learned that exact phrasing.

---

## 5. Using the outputs in the TechPath app — worked example

A student picks a career path, opens a skill, and sees an explanation from the model plus real links.

```python
import pandas as pd

nodes = pd.read_csv("out/nodes_clean.csv")
res   = pd.read_csv("out/resources_clean.csv")
tax   = pd.read_csv("out/path_taxonomy.csv")

# 1. Career-path menu (hide beginner variants, group by role/skill)
menu = tax[tax.career_path == tax.parent_path].groupby("path_type").career_path.apply(list)

# 2. Roadmap for the chosen path
path = "devops"
roadmap = nodes[nodes.career_path == path][["node_id", "skill_id", "title", "n_resources"]]

# 3. The student clicks a skill
node = roadmap[roadmap.title == "Docker"].iloc[0]

# 4. Explanation from the fine-tuned model (template must match training)
prompt = (f'Explain the skill "{node.title}" for a student pursuing the '
          f'{path.replace("-", " ")} path: what it is, what it involves and why it matters.')
explanation = gen(prompt, max_new_tokens=200, do_sample=False, return_full_text=False)[0]["generated_text"]

# 5. Real links from the dataset, never generated
links = res[res.node_id == node.node_id][["resource_type", "title", "url"]]
if links.empty:                      # 982 nodes have no links of their own; borrow from the same skill elsewhere
    links = res[res.skill_id == node.skill_id][["resource_type", "title", "url"]].drop_duplicates("url")
```

| File | Role in the app |
|---|---|
| `path_taxonomy.csv` | Builds the career-path picker |
| `nodes_clean.csv` | Lists the skills on the chosen roadmap; supplies `node_id`/`skill_id` for lookups |
| fine-tuned model | Writes the explanation for the chosen skill and career path |
| `resources_clean.csv` | Supplies the "Learn more" links |

---

## 6. Rules to keep the results valid

1. **Never train or tune on `test.jsonl`.** Use `val.jsonl` for every decision, and `test.jsonl` once at the end.
2. **Don't re-split by hand.** If the data changes, rerun the script. The seed (42) keeps the split reproducible.
3. **Keep the prompt template fixed** between training, evaluation and the app.
4. **Links come only from `resources_clean.csv`**, never from model output.
