# Releasing

1. Bump `version` in the root `Cargo.toml` (`[workspace.package]` and every
   `[workspace.dependencies]` entry of a nuvec crate); `cargo update -w`.
2. `CHANGELOG.md`: move `[Unreleased]` to `## [X.Y.Z] - <date>`, keep an empty
   `## [Unreleased]`, update the link lines at the bottom.
3. `just ci` and `cargo publish --workspace --dry-run --locked`.
4. Benchmarks on an idle Mac on AC power (models in the Hugging Face cache, [uv](https://docs.astral.sh/uv/)):
   - `just bench-parity --release` (the quick tier, about 45 minutes): nuvec against
     mlx-lm and against the previous release (the latest `v*` tag, built in a worktree and
     run in the same session). It exits 4 when a cell stays below 0.97× of the previous
     release after a re-measure: fix it or explain it in the notes.
   - On a minor release also `just bench-parity --release --tier full` (hours; continue
     with `--resume`, stop between children with `--budget SECONDS`).
   - `just bench-validate`; update the README and the benchmarks page from the records.
   - `just soak 1`: an hour of server-like work on a caller-owned thread without growth.
   - `just test-large`: arrays of 2^32 elements and more on the GPU (~14 GB, run alone).
5. `just api-check` (the pinned surface and `cargo semver-checks` on `nuvec` and `nuvec-lm`).
6. Commit (records under `site/static/benchmarks/` included), push to `main`, wait for CI.
7. Tag `vX.Y.Z` on that commit and push the tag. The release workflow checks the tag, the
   version and the changelog, re-runs the API check, publishes to crates.io (secret
   `CARGO_REGISTRY_TOKEN`) and creates the GitHub release from the changelog section.
