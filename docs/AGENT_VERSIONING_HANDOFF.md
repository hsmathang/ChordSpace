# Agent Versioning Handoff

This repo may be edited by different coding agents. To avoid Git metadata conflicts,
do not open the same local folder in two agent apps at the same time.

Recommended workflow:

1. Use one task branch per line of work.
   - Current defense-slide branch: `codex/defensa-diapos`

2. Use separate local clones for separate apps.
   - Codex clone example: `D:\Documents\GitHub\ChordSpace`
   - Antigravity clone example: `D:\Documents\GitHub\ChordSpace-Antigravity`

3. Transfer work through GitHub.
   - Before switching apps:
     `git status`
     `git add <changed files>`
     `git commit -m "<short message>"`
     `git push`
   - In the other app/clone:
     `git fetch`
     `git switch codex/defensa-diapos`
     `git pull`

4. Avoid committing local agent worktrees.
   - `.claude/worktrees/` is local agent workspace data and should not be committed.

5. Current slide server command:
   `python -m http.server 4173 --bind 127.0.0.1`

   Run it from:
   `D:\Documents\GitHub\ChordSpace\docs\Presentación-defensa\UNAL Design System`

   Then open:
   `http://127.0.0.1:4173/slides/index.html`

Notes:

- The repository contains gitlink/submodule-style entries without a complete
  `.gitmodules` file. This can make plain `git status` fail if Git tries to
  recurse into those entries.
- This clone is configured locally to avoid submodule status recursion.
- If another tool behaves strangely, prefer using a fresh separate clone and
  pulling `codex/defensa-diapos` from GitHub.
