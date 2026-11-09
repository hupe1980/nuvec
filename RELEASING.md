# Releasing

1. Write the release's entries under `## [Unreleased]` in `CHANGELOG.md` and commit.
2. If the release touches kernels, the scheduler or models: `just bench-parity --tier smoke`
   (about 10 minutes, two models against mlx-lm and the previous release; fine on a busy
   machine). It exits 4 on a confirmed regression: fix it or explain it in the changelog.
3. `just release X.Y.Z`: bumps the versions, dates the changelog section, runs `just ci`,
   `cargo publish --workspace --dry-run` and `just api-check`, then commits `release: X.Y.Z`
   and tags `vX.Y.Z`.
4. `git push origin main vX.Y.Z`. The release workflow checks the tag, the version and the
   changelog, re-runs the API check, publishes to crates.io (secret `CARGO_REGISTRY_TOKEN`)
   and creates the GitHub release from the changelog section.

## Measurements

Not part of a release; run them when the machine is free and commit the records under
`site/static/benchmarks/` (`--release` writes them there and needs a committed tree):

- `just bench-parity --release` (quick tier, about 45 minutes) and
  `just bench-parity --release --tier full` (hours; `--budget SECONDS` stops between
  measurements, `--resume` continues). Then `just bench-validate` and update the
  [benchmarks page](site/content/docs/benchmarks.md) and the headline numbers in the README
  and on the landing page.
- `just soak 1` (an hour of server-like work) and `just test-large` (arrays of 2^32 elements
  and more, about 14 GB of GPU memory) after changes to memory, the allocator or indexing.
