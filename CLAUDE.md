# CLAUDE.md

See [AGENTS.md](AGENTS.md) for everything: using soif from an agent, repo conventions,
factor-update rules, and dev commands (`pip install -e ".[dev]"`, `pytest -q`,
`ruff check .`).

Claude-specific extras:

- A ready-made skill for water-footprint questions lives in `.claude/skills/soif/`.
- The Stop-hook integration (`soif claude-hook`) is documented in
  `integrations/claude-code/README.md`.
