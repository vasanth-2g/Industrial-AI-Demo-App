# Push Rules

Use Git for source, notebooks, and small contracts. Do not use Git for raw
datasets, generated images, or model memory banks.

## Push

```text
restore_ignored_datasets.ipynb
build_restore_ignored_datasets_notebook.py
RECREATE_CURRENT_STATE.md
next_steps/*.md
telemetrics/*.ipynb
vision/*.ipynb
knowledge_rag.ipynb
knowledge/artifacts/rag/*.json
knowledge/artifacts/rag/*.jsonl
knowledge/artifacts/rag/*.csv
```

## Do Not Push

```text
data/VisA/
data/CMAPSSData/
data/_cmapss_extract/
data/*.zip
data/synthetic_mechanical/generated/anomaly/
data/synthetic_mechanical/generated/masks/
data/synthetic_mechanical/generated/normal/
data/synthetic_mechanical/generated/component_masks/
vision/artifacts/**/normal_patch_memory.pt
.ipynb_checkpoints/
```

## Check Before Push

Run:

```bash
git status --short --ignored
git ls-files | grep -E "\\.pt$|data/VisA|data/CMAPSSData|_cmapss_extract|\\.zip$"
```

The second command should return nothing unless you intentionally track a large
artifact.

## Safe Push Commands

```bash
git add restore_ignored_datasets.ipynb
git add build_restore_ignored_datasets_notebook.py
git add RECREATE_CURRENT_STATE.md
git add next_steps/*.md
git commit -m "Add execution order and restore instructions"
git push origin main
```

