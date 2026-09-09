# Sync this palace with its peer, both directions. Run from the repo root:
#   .\sync.ps1              pull then push
#   .\sync.ps1 --dry-run    show the delta, ship nothing
#   .\sync.ps1 --pull-only
#   .\sync.ps1 --source codex  # only exported Codex transcripts
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\scripts\palace_sync.py" @args
exit $LASTEXITCODE
