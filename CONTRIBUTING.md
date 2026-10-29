# Contributing

Thanks for helping! These are the team's working rules for the Meta Global AI Developer Hackathon
(submission: Nov 4, 9pm ET). Coding agents follow the same rules through [AGENTS.md](AGENTS.md).

## Workflow

- Work on a branch and open a pull request into `main`. Don't push to `main` directly (it is protected).
- Keep PRs small and focused: one idea per PR.
- Yuzuki reviews and merges. We merge with **merge commits, not squash**: the commit history shows
  what was built during the hackathon, so keep it.
- Write everything in English: code comments, docs, commit messages, PR descriptions.
  Yuzuki keeps `README.ja.md` in sync.

## Before you open a PR

```bash
python -m unittest discover -s tests -t .
python -m pyflakes topo_agent tests
```

Both must pass. Tests use a fake SAM and a fake Muse Spark, so they never call an API.

If the change affects what a method finds or how fast (prompts, tools, SAM settings, scoring),
include bench results in the PR description and add an entry to [docs/experiments.md](docs/experiments.md):

```bash
python -m topo_agent.eval --bench --methods sam_color,meta --seeds 11,12
```

Calling Meta Model API needs your own `MODEL_API_KEY` (from your hackathon credits), set as an
environment variable. Never put a key in code, commits, PR text or chat.

## Data and privacy

- Send to external AI APIs only photos you may share: drawn sample walls, your own photos, or a dataset
  whose license allows it. Don't send photos of other people's faces.
- Don't commit a dataset until its license is checked and attribution is written next to it.
- Requests to Meta use `store: false`. Keep it that way.

## Hackathon boundary

`topo_agent/baseline/` is pre-hackathon code, imported in the first commit (tag `pre-hackathon`).
Don't change it: it proves which parts were built during the hackathon. Build new code next to it.

## Record what you try

Every experiment worth remembering, including the ones that didn't work, goes into
[docs/experiments.md](docs/experiments.md): what, setup, results, takeaways. The repo is public, so
this is also how the judges see what we tried and learned.
