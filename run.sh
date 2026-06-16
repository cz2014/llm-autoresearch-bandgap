#!/usr/bin/env bash
# Persistent Claude Code launcher (parallel-agent test vs the codex run, SAME substrate).
# ONE long-lived `claude -p` session owns the loop (program.md "Loop (forever)"). If it ever exits,
# an outer loop resumes the SAME session with --continue, so in-context memory is preserved.
# Subscription-billed (Claude Max); ANTHROPIC_API_KEY is unset so it cannot fall back to API credits.
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
unset ANTHROPIC_API_KEY              # force subscription (Max) billing; never API credits
export CLAUDE_CODE_EFFORT_LEVEL=max  # highest adaptive-reasoning budget for Opus 4.8 (low<med<high<xhigh<max)
cd "$REPO"
MODEL="opus"                         # bare alias -> Claude Opus 4.8 (latest, most capable)

PROMPT="You are the autonomous researcher described in program.md. Read program.md and run its 'Loop \
(forever)' section CONTINUOUSLY, in this single session. Your FIRST action after setup is the BASELINE \
-- run the seed train.py AS IS once and record it as the baseline (status keep); only THEN start \
changing things. Skim the reference files in ref/ for ideas, and spread experiments across the \
design-space axes A1-A5 (A6 optimization is mostly mapped out). The RNG seed is fixed, so one run per \
candidate is a fair comparison -- KEEP iff mae < best - 0.002, else git reset. Tag each results.tsv row \
with its axis. After you log each result, IMMEDIATELY begin the next experiment. Do NOT stop after one \
experiment, do NOT end your turn, do NOT print a final summary, do NOT ask whether to continue. Never \
touch run.sh, tmux, this session, or prepare.py. Run experiment after experiment until a human \
interrupts you."

# Primary: one persistent headless session that owns the loop.
claude -p --verbose --dangerously-skip-permissions --model "$MODEL" "$PROMPT"

# Self-heal: if the session ever ends, resume it (memory preserved) and keep looping.
while true; do
  sleep 5
  claude -p --continue --verbose --dangerously-skip-permissions --model "$MODEL" \
    "Continue the autoresearch loop per program.md and your earlier instructions. Do NOT stop, do NOT \
ask whether to continue -- immediately run the next experiment, and keep going until interrupted."
done
