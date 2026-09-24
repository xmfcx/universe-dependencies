#!/usr/bin/env bash
set -euo pipefail

site_dir=$1
data_dir=$2

if git ls-remote --exit-code --heads origin data >/dev/null; then
  git fetch origin data
  git worktree add -B data "$data_dir" FETCH_HEAD
else
  git worktree add --detach "$data_dir" HEAD
  git -C "$data_dir" switch --orphan data
  git -C "$data_dir" rm -r -f --ignore-unmatch .
fi

cp "$site_dir/candidates.csv" "$data_dir/candidates.csv"
cp "$site_dir/metadata.json" "$data_dir/metadata.json"
git -C "$data_dir" config user.name "Mete Fatih Cırıt"
git -C "$data_dir" config user.email "mfc@autoware.org"
git -C "$data_dir" add --all
if ! git -C "$data_dir" diff --cached --quiet; then
  git -C "$data_dir" commit -m "chore(data): refresh candidate snapshot" \
    -m "Signed-off-by: Mete Fatih Cırıt <mfc@autoware.org>"
  git -C "$data_dir" push origin HEAD:refs/heads/data
fi
git worktree remove "$data_dir"
