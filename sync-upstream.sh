#!/usr/bin/env bash
#
# Sync new course material from epfml/ML_course into this fork.
#
#   ./sync-upstream.sh            # sync main, then merge it into your current branch
#   ./sync-upstream.sh --check    # only report whether upstream has new commits
#
# Layout this assumes:
#   main     -> clean mirror of upstream, you never commit here
#   my-work  -> your notes, data and solutions
#
set -euo pipefail

MIRROR_BRANCH="main"
UPSTREAM="upstream"

cd "$(git rev-parse --show-toplevel)"

if ! git remote get-url "$UPSTREAM" >/dev/null 2>&1; then
  echo "error: no '$UPSTREAM' remote. Add it with:" >&2
  echo "  git remote add $UPSTREAM https://github.com/epfml/ML_course.git" >&2
  exit 1
fi

echo "==> Fetching $UPSTREAM"
git fetch --quiet "$UPSTREAM" "$MIRROR_BRANCH"

behind=$(git rev-list --count "$MIRROR_BRANCH..$UPSTREAM/$MIRROR_BRANCH")
if [ "$behind" -eq 0 ]; then
  echo "==> Already up to date with $UPSTREAM/$MIRROR_BRANCH."
  [ "${1:-}" = "--check" ] && exit 0
else
  echo "==> $behind new upstream commit(s):"
  git log --oneline --no-decorate "$MIRROR_BRANCH..$UPSTREAM/$MIRROR_BRANCH" | sed 's/^/    /'
  echo "==> Files touched:"
  git diff --name-status "$MIRROR_BRANCH".."$UPSTREAM/$MIRROR_BRANCH" | sed 's/^/    /'
  if [ "${1:-}" = "--check" ]; then
    echo "==> Run './sync-upstream.sh' to pull these in."
    exit 0
  fi
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "error: working tree is dirty. Commit or stash first." >&2
  exit 1
fi

work_branch=$(git rev-parse --abbrev-ref HEAD)

if [ "$behind" -gt 0 ]; then
  echo "==> Fast-forwarding $MIRROR_BRANCH"
  git checkout --quiet "$MIRROR_BRANCH"
  # --ff-only on purpose: if this fails you committed your own work to the
  # mirror branch. Move those commits to your work branch instead of merging.
  git merge --ff-only "$UPSTREAM/$MIRROR_BRANCH"
  git push --quiet origin "$MIRROR_BRANCH"
  echo "==> Pushed $MIRROR_BRANCH to your fork"
fi

if [ "$work_branch" = "$MIRROR_BRANCH" ]; then
  echo "==> Done (you were on $MIRROR_BRANCH, nothing to merge)."
  exit 0
fi

echo "==> Merging $MIRROR_BRANCH into $work_branch"
git checkout --quiet "$work_branch"
if git merge --no-edit "$MIRROR_BRANCH"; then
  echo "==> Done. $work_branch now has the new course material."
else
  echo
  echo "!!! Merge conflict. Upstream changed files you also edited."
  echo "    Resolve them, then: git add <files> && git commit"
  echo "    Or back out with: git merge --abort"
  exit 1
fi
