# Releasing

1. Bump `version` in the root `Cargo.toml` (`[workspace.package]` and every
   `[workspace.dependencies]` entry of a nuvec crate); `cargo update -w`.
2. `CHANGELOG.md`: move `[Unreleased]` to `## [X.Y.Z] - <date>`, keep an empty
   `## [Unreleased]`, update the link lines at the bottom.
3. `just ci` and `cargo publish --workspace --dry-run --locked`.
4. Benchmarks on an idle Mac on AC power (models in the Hugging Face cache, [uv](https://docs.astral.sh/uv/)):
   - `just bench-parity --nuvec-ref vPREVIOUS --out target/bench-parity/base.json` (the
     previous release, same day), then `just bench-parity --release` (all sections on a
     minor release, `--sections decode,ttft` on a patch). Long runs continue with
     `--resume`; `--budget SECONDS` stops between children.
   - `just bench-compare target/bench-parity/base.json site/static/benchmarks/X.Y.Z/<machine>.json`:
     a cell that fell by more than 2% blocks the release or is explained in the notes.
   - `just bench-validate`; update the README and the benchmarks page from the record.
5. `just api-check` (the pinned surface and `cargo semver-checks` on `nuvec` and `nuvec-lm`).
6. Commit (records under `site/static/benchmarks/` included), push to `main`, wait for CI.
7. Tag `vX.Y.Z` on that commit and push the tag. The release workflow checks the tag, the
   version and the changelog, re-runs the API check, publishes to crates.io (secret
   `CARGO_REGISTRY_TOKEN`) and creates the GitHub release from the changelog section.
