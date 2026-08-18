# soif × Claude Code

A [Claude Code hook](https://code.claude.com/docs/en/hooks) that reports the water
footprint of your session after every turn, computed from the **real token usage** in the
session transcript (input, output, and cache-read tokens per assistant message, per
model) — no guessing.

## Setup

1. Install soif where Claude Code can see it:

   ```bash
   pip install soif    # or: uv tool install soif / pipx install soif
   ```

2. Add a `Stop` hook to `.claude/settings.json` (project) or `~/.claude/settings.json`
   (global):

   ```json
   {
     "hooks": {
       "Stop": [
         {
           "hooks": [
             { "type": "command", "command": "soif claude-hook" }
           ]
         }
       ]
     }
   }
   ```

   Optionally pin your grid region: `"command": "soif claude-hook --region eu"`.

3. That's it. After each turn Claude Code shows something like:

   ```
   soif: this session used ~4.31 mL of water (0.9 teaspoons); range 0.42 mL - 48.1 mL across 12 model call(s).
   ```

## Notes

- The hook reads the transcript path from the hook payload on stdin, de-duplicates
  streamed message chunks by message id, and sums usage for the **whole session** so the
  number grows as the session does.
- It never blocks the agent: any error exits 0 silently.
- The same subcommand works for any tool that produces Claude-style JSONL transcripts:
  `soif claude-hook --transcript path/to/transcript.jsonl < /dev/null`.
