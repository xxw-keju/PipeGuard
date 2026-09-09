# Evaluation suites

This directory contains source-controlled corpus snapshots only. Runtime outputs, compiler caches, binaries, and `.artifacts` are intentionally excluded. `provenance.json` records the source worktree identities and corpus digests.

- `suite2/`: the complete source/specification corpus copied from the TileLang owner worktree.
- `suite3/`: the complete 964-kernel real upstream source/specification corpus.
- `suite4/`: the frozen 376-case mutation corpus used by the source-based PipeGuard black-box pass.

Each suite contains `FILES.sha256`, covering the copied source corpus before the checksum file itself was added.
