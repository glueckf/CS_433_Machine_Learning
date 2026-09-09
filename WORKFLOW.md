# How I maintain this fork

This is a fork of [epfml/ML_course](https://github.com/epfml/ML_course) (EPFL CS-433).
Goal: keep receiving new lectures and labs from the course, while keeping my own
notes, data and solutions in the same repo.

## Layout

| Branch    | What it is                                                      |
|-----------|-----------------------------------------------------------------|
| `main`    | A clean mirror of the course repo. **I never commit here.**      |
| `my-work` | Everything of mine. This is the branch I work on day to day.     |

| Remote     | Points at                                                |
|------------|----------------------------------------------------------|
| `origin`   | `glueckf/CS_433_Machine_Learning` — my fork              |
| `upstream` | `epfml/ML_course` — the course repo (read-only)          |

## Day to day

```bash
git checkout my-work     # where I always am
# ...edit, then...
git add -A && git commit -m "lab 3 notes"
git push                 # pushes my-work to my fork
```

## Pulling in new course material

```bash
./sync-upstream.sh --check   # anything new? (just reports, changes nothing)
./sync-upstream.sh           # ff main to upstream, push it, merge main -> my-work
```

That is the whole maintenance story. Run it whenever a new lecture or lab drops.

## The one rule that keeps merges painless

**Add files; don't edit theirs in place.** Conflicts only happen when upstream
changes the same lines I changed. So:

- My own work goes in new files or folders — `my/`, `notes/`, `lab03/my_solution.ipynb`.
- To work through one of their notebooks, copy it first rather than editing it:
  `cp labs/ex03/template.ipynb labs/ex03/ex03_finn.ipynb`

Follow that and `./sync-upstream.sh` merges cleanly essentially every time.

## When it does conflict anyway

```bash
./sync-upstream.sh
# !!! Merge conflict.
git status                       # see which files
# fix the conflict markers, then
git add <files> && git commit
# or bail out entirely:
git merge --abort
```

Notebooks conflict ugly (they're big JSON blobs). Usually the fastest fix is to
decide wholesale which side wins:

```bash
git checkout --theirs labs/ex03/template.ipynb   # take upstream's version
git checkout --ours   labs/ex03/template.ipynb   # keep mine
git add labs/ex03/template.ipynb
```

## Gotcha: `./sync-upstream.sh` refuses to fast-forward `main`

That means I accidentally committed to `main`. Move the commit to `my-work`:

```bash
git checkout my-work && git cherry-pick <sha>   # bring it over
git checkout main && git reset --hard upstream/main
```

## Notes

- I never open a pull request against `epfml/ML_course` — my work is not meant to
  go back to them. The fork is just a convenient way to track their updates.
- GitHub's "Sync fork" button on the repo page does the `main` half of the sync,
  but I still need `git merge main` locally to get it into `my-work`, so the
  script is simpler.
