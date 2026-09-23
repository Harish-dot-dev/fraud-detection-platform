#!/usr/bin/env bash
#
# Download the IEEE-CIS Fraud Detection dataset into data/raw/.
#
# The dataset is competition data: you must accept the rules on Kaggle and use
# your own API token. It is NOT committed to this repository (see .gitignore)
# because its licence does not allow redistribution.
#
# Prerequisites:
#   1. A Kaggle account.
#   2. Accept the rules at
#      https://www.kaggle.com/competitions/ieee-fraud-detection/rules
#   3. An API token at https://www.kaggle.com/settings -> "Create New Token",
#      saved to ~/.kaggle/kaggle.json with mode 600.
#
# If you would rather not use the CLI, download the ZIP manually and unzip
# train_transaction.csv and train_identity.csv into data/raw/.

set -euo pipefail

COMPETITION="ieee-fraud-detection"
RAW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
REQUIRED_FILES=("train_transaction.csv" "train_identity.csv")

mkdir -p "$RAW_DIR"

# Already downloaded? Do nothing: the files are ~1.4 GB together.
all_present=true
for f in "${REQUIRED_FILES[@]}"; do
  [[ -f "$RAW_DIR/$f" ]] || all_present=false
done
if [[ "$all_present" == true ]]; then
  echo "Dataset already present in $RAW_DIR - nothing to do."
  ls -lh "$RAW_DIR"
  exit 0
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
  cat >&2 <<MSG
ERROR: ~/.kaggle/kaggle.json not found.

  1. Sign in to Kaggle and accept the competition rules:
       https://www.kaggle.com/competitions/${COMPETITION}/rules
  2. Create an API token at https://www.kaggle.com/settings
  3. Save it and lock down the permissions:
       mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
       chmod 600 ~/.kaggle/kaggle.json

Alternatively, download the ZIP by hand and unzip these files into data/raw/:
  ${REQUIRED_FILES[*]}
MSG
  exit 1
fi

# Prefer the project's own virtualenv over whatever is on PATH. `make data`
# runs without the venv activated, so requiring an activated shell here is a
# trap: the CLI gets installed into .venv and then is not found.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$REPO_ROOT/.venv/bin/kaggle" ]]; then
  KAGGLE="$REPO_ROOT/.venv/bin/kaggle"
elif command -v kaggle >/dev/null 2>&1; then
  KAGGLE="kaggle"
else
  cat >&2 <<'MSG'
ERROR: the Kaggle CLI is not installed.

It is an optional extra, because the tests and CI run on the committed
synthetic fixture and need no Kaggle account. Install it with:

    pip install -e '.[data]'

or, if you are not using the project's virtualenv:

    pip install kaggle==1.6.17
MSG
  exit 1
fi

echo "Downloading ${COMPETITION} into ${RAW_DIR} (about 1.4 GB unzipped)..."
for f in "${REQUIRED_FILES[@]}"; do
  echo "  -> $f"
  "$KAGGLE" competitions download -c "$COMPETITION" -f "$f" -p "$RAW_DIR"
done

# The CLI delivers single files zipped when they are large.
shopt -s nullglob
for zip in "$RAW_DIR"/*.zip; do
  echo "Unzipping $(basename "$zip")..."
  unzip -o -q "$zip" -d "$RAW_DIR"
  rm -f "$zip"
done
shopt -u nullglob

echo ""
echo "Done. Files in $RAW_DIR:"
ls -lh "$RAW_DIR"
