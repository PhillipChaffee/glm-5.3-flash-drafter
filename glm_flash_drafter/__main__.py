"""Verify the frozen reference snapshots from the repository root."""

from pathlib import Path

from glm_flash_drafter.references import verify_references

failures = verify_references(Path.cwd())
if failures:
    raise SystemExit("\n".join(failures))
print("Pinned reference checksums verified.")
