"""TechPath — roadmap.sh dataset pre-processing (Chapter 3, section 3.6).

Usage:  python preprocess_roadmap.py roadmap_dataset.csv out_dir/
Outputs: nodes_clean.csv, train/val/test.jsonl, resources_clean.csv, path_taxonomy.csv, preprocessing_log.txt
Every step logs rows in / rows out so the counts in 3.6 can be reproduced.
"""
import re, sys, hashlib, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
RES_COLS = ["courses", "articles", "official_docs", "videos", "opensource"]
MIN_DESC_CHARS = 40          # descriptions shorter than this carry no usable explanation
ALIAS_PATHS = {"r-programming": "r"}   # verified identical (129/129 node_ids shared)
BEGINNER_PATHS = {"backend-beginner": "backend", "frontend-beginner": "frontend",
                  "devops-beginner": "devops", "git-github-beginner": "git-github"}
# roadmap.sh's own split: role-based roadmaps vs skill/technology roadmaps
ROLE_PATHS = {"frontend", "backend", "full-stack", "devops", "devsecops", "data-analyst",
    "ai-engineer", "ai-data-scientist", "data-engineer", "android", "ios", "machine-learning",
    "mlops", "postgresql-dba", "blockchain", "qa", "software-architect", "cyber-security",
    "ux-design", "technical-writer", "game-developer", "server-side-game-developer",
    "product-manager", "engineering-manager", "devrel", "bi-analyst", "network-engineer",
    "forward-deployed-engineer", "ai-red-teaming", "ai-product-builder", "product-design"}

log = []
def step(name, before, after, note=""):
    log.append(f"{name:<45} {before:>6} -> {after:>6}  {note}")

def clean_text(s):
    s = unicodedata.normalize("NFC", str(s))
    s = re.sub(r"<[^>]+>", " ", s)                 # HTML tags
    s = re.sub(r"`([^`]*)`", r"\1", s)             # inline code ticks
    s = re.sub(r"\*\*|__|(?<!\w)\*(?!\s)", "", s)  # bold / italic markers
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s) # markdown links -> text
    s = re.sub(r"(Articles|Visit the following resources.*)\s*=*\s*$", "", s)  # scrape residue
    return re.sub(r"\s+", " ", s).strip()

def slugify(t):
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", t.lower())).strip("-") or "untitled"

def parse_resources(cell):
    out = []
    if pd.isna(cell): return out
    for part in str(cell).split(" | "):
        m = re.match(r"^(.*)\((h?ttps?://[^\s)]+)\)\s*$", part.strip())
        if not m: continue
        title, url = m.group(1).strip(), m.group(2)
        if url.startswith("ttp"): url = "h" + url          # repair truncated scheme
        out.append((title, url))
    return out

def main(src, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src, dtype=str); n0 = len(df); step("0 load raw", n0, n0)

    for c in df.columns: df[c] = df[c].str.strip()
    n = len(df); df = df.dropna(subset=["title", "node_id"]); step("1 drop missing title/node_id", n, len(df))

    n = len(df); df = df.drop_duplicates(subset=["career_path", "node_id"], keep="first")
    step("2 drop duplicate (path,node_id)", n, len(df))

    n = len(df); df = df[~df.career_path.isin(ALIAS_PATHS)]
    step("3 drop alias path r-programming", n, len(df), "duplicate of 'r'")

    df["description"] = df["description"].fillna("").map(clean_text)
    n = len(df); df = df[df.description.str.len() >= MIN_DESC_CHARS]
    step(f"4 drop description < {MIN_DESC_CHARS} chars", n, len(df))

    miss = df.topic_slug.isna().sum()
    df["topic_slug"] = df["topic_slug"].fillna(df["title"].map(slugify))
    step("5 impute missing topic_slug from title", len(df), len(df), f"{miss} imputed")

    # path metadata (kept, not dropped)
    df["parent_path"] = df.career_path.map(BEGINNER_PATHS).fillna(df.career_path)
    df["path_type"] = df.parent_path.map(lambda p: "role" if p in ROLE_PATHS else "skill")
    df["is_beginner_variant"] = df.career_path.isin(BEGINNER_PATHS)

    # canonical skill id: same title + same description = same skill appearing in several paths
    key = df.title.str.lower().str.strip() + "||" + df.description.str.lower()
    df["skill_id"] = key.map(lambda k: hashlib.sha1(k.encode()).hexdigest()[:12])
    shared = df.skill_id.duplicated(keep=False).sum()
    step("6 assign canonical skill_id", len(df), len(df), f"{shared} rows share a skill_id across paths")

    # resources -> long table
    rows = []
    for _, r in df.iterrows():
        seen = set()
        for c in RES_COLS:
            for title, url in parse_resources(r[c]):
                if url in seen: continue
                seen.add(url)
                dom = re.sub(r"^www\.", "", re.findall(r"https?://([^/]+)", url)[0])
                rows.append(dict(career_path=r.career_path, node_id=r.node_id, skill_id=r.skill_id,
                                 resource_type=c, title=clean_text(title), url=url, domain=dom))
    res = pd.DataFrame(rows)
    df["n_resources"] = df.node_id.map(res.groupby("node_id").size()).fillna(0).astype(int)
    step("7 parse resources to long table", len(df), len(res), "resource rows")

    # 8. instruction pairs for fine-tuning: (career, skill) -> roadmap.sh description (links stay out of targets)
    df["prompt"] = ("Explain the skill \"" + df.title + "\" for a student pursuing the "
                    + df.career_path.str.replace("-", " ") + " path: what it is, what it involves and why it matters.")
    df["target"] = df.description

    # 9. grouped split 80/10/10 by normalised title, so one skill name never sits in two splits
    df["title_group"] = (df.title.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip())
    groups = np.array(sorted(df.title_group.unique()), dtype=object)
    np.random.default_rng(SEED).shuffle(groups)
    n = len(groups); tr = set(groups[: int(.8 * n)]); va = set(groups[int(.8 * n): int(.9 * n)])
    df["split"] = np.where(df.title_group.isin(tr), "train", np.where(df.title_group.isin(va), "val", "test"))
    step("8-9 build pairs + grouped split (seed 42)", len(df), len(df),
         f"groups={n} " + str(df.split.value_counts().to_dict()))

    keep = ["skill_id", "title_group", "split", "career_path", "parent_path", "path_type",
            "is_beginner_variant", "topic_slug", "node_id", "title", "description", "n_resources",
            "prompt", "target"]
    df[keep].to_csv(out / "nodes_clean.csv", index=False)
    for s in ["train", "val", "test"]:
        df[df.split == s][["prompt", "target"]].to_json(out / f"{s}.jsonl", orient="records", lines=True)
    res.to_csv(out / "resources_clean.csv", index=False)
    tax = df.groupby(["career_path", "parent_path", "path_type"]).size().rename("nodes").reset_index()
    tax.to_csv(out / "path_taxonomy.csv", index=False)

    summ = [f"paths={df.career_path.nunique()} role={tax[tax.path_type=='role'].career_path.nunique()} "
            f"skill={tax[tax.path_type=='skill'].career_path.nunique()}",
            f"unique skills={df.skill_id.nunique()} nodes={len(df)}",
            f"nodes with 0 resources={(df.n_resources==0).sum()}",
            f"resource types={res.resource_type.value_counts().to_dict()}",
            f"top domains={res.domain.value_counts().head(5).to_dict()}",
            f"youtube share={(res.domain.str.contains('youtube|youtu.be')).mean():.1%}",
            f"desc chars median={df.description.str.len().median():.0f}",
            f"nodes per path min/median/max={tax.nodes.min()}/{tax.nodes.median():.0f}/{tax.nodes.max()}"]
    (out / "preprocessing_log.txt").write_text("\n".join(log + [""] + summ) + "\n")
    print("\n".join(log + [""] + summ))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])