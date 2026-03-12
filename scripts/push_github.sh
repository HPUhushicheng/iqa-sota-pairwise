#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="iqa-sota-pairwise"
OWNER="HPUhushicheng"

# 1) Login once (if needed)
gh auth status >/dev/null 2>&1 || gh auth login --web --scopes repo

# 2) Create remote repo if missing
if ! gh repo view "${OWNER}/${REPO_NAME}" >/dev/null 2>&1; then
  gh repo create "${OWNER}/${REPO_NAME}" --public --description "Small-data pairwise IQA baseline: CLIP + LR/XGB + GroupKFold" --confirm
fi

# 3) Push to remote main (from local codex branch)
git remote remove origin >/dev/null 2>&1 || true
git remote add origin "https://github.com/${OWNER}/${REPO_NAME}.git"
git push -u origin codex/iqa-pairwise-sota:main
