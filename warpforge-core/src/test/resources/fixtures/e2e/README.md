# E2E Test Fixtures

**Fixtures are NOT stored in this directory.**

Test fixtures are generated to `build/generated-fixtures/e2e/` which is:
- Outside the source tree
- Always gitignored (build/ is never committed)
- Generated on-demand

## Generating Fixtures

```bash
./gradlew :warpforge-core:generateE2EFixtures
```

This uses the snakegrinder native binary to trace PyTorch models and capture
tensor data for end-to-end testing.

## Why Not Store Fixtures in the Repo?

Large ML test fixtures (BERT models are 300MB+) must NEVER be committed to
git repositories. They bloat the repo permanently and make cloning slow.

See CLAUDE.md section "CRITICAL: No Data Files in Repository" for details.
