# Public Security Checklist

Run this checklist before pushing public updates.

## Credentials
- No hardcoded API keys/tokens/passwords in tracked files.
- `.env` is ignored.
- `.env.example` contains placeholders only.

## Personal Data
- No personal names, emails, addresses, or local machine absolute paths in tracked files.
- No private family audio samples in tracked files.

## Large/Sensitive Artifacts
- Raw audio is ignored (`*.wav`, `*.mp3`, `*.m4a`, etc.).
- Generated datasets are ignored (`Data/**`, `*.csv`, `*.zip`).
- Model binaries are ignored unless intentionally publishing.

## Git Hygiene
- `git status` is clean before push.
- Verify `git remote -v` points to your intended repository.
- If a secret was ever committed, rotate it and rewrite git history before public release.

## Suggested Scan Commands
```bash
git grep -nI -E "api[_-]?key|secret|token|password|BEGIN (RSA|OPENSSH)"
git ls-files | rg "(\.wav$|\.mp3$|\.m4a$|\.env$|^Data/)" || true
```
