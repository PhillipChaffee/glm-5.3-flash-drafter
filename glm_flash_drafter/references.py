"""Check frozen upstream reference files before using their target contracts."""

import hashlib
from pathlib import Path


def verify_references(root: Path) -> list[str]:
    """Return missing or modified reference paths from the committed checksum list.

    Args:
        root: Repository root containing reference/SHA256SUMS.

    Raises:
        ValueError: The checksum list is empty or a record is malformed.
        OSError: The checksum list or an existing reference cannot be read.
    """
    records = (root / "reference/SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if not records:
        message = "The reference checksum list is empty."
        raise ValueError(message)
    failures = []
    for record in records:
        expected, relative_path = record.split(maxsplit=1)
        path = root / relative_path
        if not path.is_file():
            failures.append(f"{relative_path}: missing")
            continue
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            failures.append(f"{relative_path}: checksum mismatch")
    return failures
