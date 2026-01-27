# Testing Philosophy

This document covers WarpForge's testing principles and practices.

## Framework

- **JUnit 5** for all tests
- Tag tests with `@Tag("cpu")`, `@Tag("nvidia")`, or `@Tag("amd")` for hardware-specific execution
- GPU tests have 60-second SSH/wake timeouts

## Hardware Tests Must Run on Hardware

**GPU kernel tests must actually execute on GPU hardware.** Code generation tests alone are insufficient.

When writing tests for GPU backends (NVIDIA or AMD), you need BOTH:

1. **Code generation tests** (no hardware required) - Verify the kernel source/PTX is generated correctly
2. **Hardware execution tests** (tagged for specific GPU) - Verify the kernel runs correctly on actual hardware

**Wrong approach:**
```java
// BAD - Only tests that HIP source code is generated, never runs on GPU
@Test
void testAddKernel() {
    String src = HipKernels.generateAddF32(SALT_NONE);
    assertTrue(src.contains("extern \"C\" __global__ void add_f32"));
}
```

**Correct approach:**
```java
// GOOD - Tests code generation (runs on NUC)
@Test
void testAddPtxGeneration() {
    String src = HipKernels.generateAddF32(SALT_NONE);
    assertTrue(src.contains("extern \"C\" __global__ void add_f32"));
}

// GOOD - Tests actual execution (runs on AMD box)
@Test
@Tag("amd")
void testAddExecution() {
    assumeTrue(HipRuntime.isAvailable(), "ROCm not available");
    float[] a = {1.0f, 2.0f, 3.0f};
    float[] b = {4.0f, 5.0f, 6.0f};
    float[] result = executeAdd(a, b);
    assertArrayEquals(new float[]{5.0f, 7.0f, 9.0f}, result, 1e-5f);
}
```

**Why this matters:**
- Code generation tests only verify string content, not correctness
- A kernel might generate valid-looking code but produce wrong results
- GPU architectures have subtle differences that only manifest at runtime
- Without hardware tests, regressions in actual computation go undetected

**Rule:** If NVIDIA has hardware execution tests for an operation, AMD must have equivalent hardware execution tests. See "Backend Parity: NVIDIA ↔ AMD" in CLAUDE.md.

## Zero Tolerance for Test Failures

**The build must fail if even a single test fails.** This is non-negotiable.

Allowing test failures to accumulate leads to:
- Silent regressions that compound over time
- Loss of confidence in the test suite
- "It's probably fine" mentality that masks real bugs
- Technical debt that becomes impossible to pay down

**Rules:**
- A failing test is a build-breaking event—treat it with the same urgency as a compilation error
- Never commit code that causes test failures
- Never disable or skip tests to make the build pass (fix the test or fix the code)
- CI enforces minimum test counts per module to detect accidentally skipped tests

**If a test is flaky:**
- Fix the flakiness immediately—flaky tests are worse than no tests
- If the test cannot be fixed quickly, delete it and file an issue
- Never leave flaky tests in the suite "for now"

**Module test minimums (enforced by CI):**
- `snakeburger-core`: 300+ tests
- `warpforge-io`: 150+ tests
- `warpforge-core`: 100+ tests

## Fix the Root Cause, Not the Symptom

**When a build fails due to missing dependencies or environment issues, fix the underlying problem—never skip or disable tests to make the build pass.**

This applies to:
- Missing tools or binaries (e.g., GraalPy, native libraries)
- Missing environment configuration
- Unavailable external dependencies

**Wrong approach:**
```bash
# BAD: Skip tests when dependency is missing
if [[ ! -f "$GRAALPY_BIN" ]]; then
  log "Skipping tests because GraalPy is missing"
fi
```

**Correct approach:**
```bash
# GOOD: Download the dependency as part of the build
./gradlew :snakegrinder-dist:downloadGraalPy
./gradlew :snakegrinder-dist:testDist
```

The build system should be self-healing: if a required dependency is missing, download or build it automatically. Tests should always run—if they can't run due to missing infrastructure, that's a build system bug to fix, not a test to skip.

## Build Pass Without Tests = Not Validated

**If the build and unit tests completed but the unit tests did not actually run (for any reason), the code is not validated.**

A "successful" build that skips tests is worse than a failing build because:
- It provides false confidence that the code works
- Regressions slip through undetected
- The problem compounds as more changes are built on unvalidated code

**This counts as a failed build:**
- Tests skipped due to missing dependencies or configuration
- Tests skipped due to early exit in test setup
- Zero tests executed (test task "passed" trivially)
- Test count significantly lower than expected minimums

**What to do:**
1. If tests didn't run, investigate immediately—do not proceed with other work
2. Check the test output for "0 tests" or "skipped" messages
3. Fix the root cause (missing dependencies, broken setup, etc.)
4. Re-run and verify the expected number of tests executed

**Prevention:**
- CI enforces minimum test counts per module
- Test tasks should fail-fast if prerequisites are missing
- Never treat "no tests found" as a passing condition

## Test Isolation: No Repo Modifications

**Tests must NEVER create, modify, or delete files in the Git repository.**

All test types (unit, integration, performance, fixture validation, etc.) must:
- Write output only to `build/` directories or system temp directories
- Never modify source files, resource files, or fixtures
- Leave `git status` clean after execution

**Fixture generators are NOT tests** - they are on-demand tools that intentionally modify fixtures:
- Must be tagged separately (e.g., `@Tag("fixture-generator")`)
- Must be excluded from all test tasks (`excludeTags 'fixture-generator'`)
- Should only run via explicit task (e.g., `./gradlew generateFixtures`)
- Generated changes must be reviewed and committed intentionally

If `git status` shows modified files after running `./gradlew test`, this is a bug that must be fixed immediately.

## E2E Fixture Versioning

**Generated test fixtures must stay synchronized with their generator's dependencies.**

E2E test fixtures (in `warpforge-core/src/test/resources/fixtures/e2e/`) are generated by running PyTorch models through the snakegrinder native binary. These fixtures capture expected outputs that WarpForge must match.

**Problem**: If PyTorch is upgraded but fixtures aren't regenerated, tests may pass/fail for wrong reasons—the expected values no longer reflect current PyTorch behavior.

**Solution**: Version-stamped fixtures with automatic validation.

- `fixtures/e2e/VERSION` contains the PyTorch version used to generate fixtures
- The `test` task depends on `checkE2EFixturesVersion` which compares against `snakegrinder-dist/versions.env`
- If versions mismatch, the build fails with clear instructions:
  ```
  E2E fixtures are stale: generated with PyTorch 2.7.0, current is 2.8.0.
  Run: ./gradlew :warpforge-core:generateE2EFixtures
  ```

**Workflow when upgrading PyTorch**:
1. Update `snakegrinder-dist/versions.env` with new version
2. Rebuild venv: `./gradlew :snakegrinder-dist:buildPytorchVenv`
3. Rebuild distribution: `./gradlew :snakegrinder-dist:assembleDist`
4. Regenerate fixtures: `./gradlew :warpforge-core:generateE2EFixtures`
5. Review and commit the updated fixtures

## Polyglot Verification Testing

**When implementing the same functionality in multiple languages, both implementations must produce identical output.**

This is a core WarpForge principle: if you write a tool in Python, and a matching tool in Java, they must be byte-for-byte identical in their output. This proves correctness across language boundaries.

### Example: Logo Generator

```bash
# Python version
python3 assets/generate-logo.py --all --svg-only --output ./test-py

# Java version
java assets/GenerateLogo.java --all --svg-only --output ./test-java

# Verify identical output
diff -r test-py test-java
```

### When to Apply This Pattern

Use polyglot verification when:
- Implementing CLI tools that could be written in either language
- Creating code generators or formatters
- Building serialization/deserialization logic
- Any functionality that crosses the Python↔Java boundary
