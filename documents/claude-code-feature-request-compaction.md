# Feature Request: Context Compaction Visibility and Control

## Summary

Provide visibility and control over context compaction to prevent specification drift during complex technical tasks.

## Problem Description

When working on complex, multi-step technical tasks with Claude Code, context compaction can occur mid-task without any indication to the model or user. This causes several serious problems:

### 1. Silent Specification Drift

The model continues working but loses detailed requirements that were read earlier in the session. For example:
- Specific API signatures from architecture documentation
- Field names and exact patterns from specifications
- Constraints and requirements discussed earlier

The model doesn't know this happened and continues implementing based on a summarized understanding, which may differ from the original specifications.

### 2. Wasted User Time

Work that appeared correct during implementation fails to meet specifications that were discussed earlier but lost during compaction. The user discovers hours later that the implementation doesn't match what was specified.

### 3. No Recovery Path

Neither the user nor the model knows compaction occurred. There's no opportunity to:
- Re-read critical documents
- Verify assumptions are still valid
- Take extra care with specifications

## Real-World Example

1. User asks Claude to implement GPU scheduling validation per architecture docs
2. Claude reads `GPU-SCHEDULING.md` which specifies using `nvmlDeviceGetUtilizationRates()` for real GPU metrics
3. Claude begins implementation correctly
4. Mid-implementation, context compaction occurs
5. The summary preserves "implementing GPU validation" but loses the specific NVML API requirement
6. Claude continues, implementing **simulated metrics** instead of real NVML calls
7. Claude outputs code with `System.out.println("PASSED")` instead of real assertions
8. User discovers the error hours later after reviewing the code
9. An entire day of work is wasted

This is not a hypothetical - this happened in a real session.

## Requested Features

### 1. Compaction Signal to Model (High Priority)

Provide a signal or marker to the model when compaction has occurred. This would enable the model to:

- Proactively notify the user: "Context was compacted. I should re-read critical documents before continuing."
- Re-read project documentation (CLAUDE.md equivalent files)
- Take extra care to verify assumptions against source files
- Ask clarifying questions rather than proceeding with potentially stale context

**Implementation suggestion:** A system message or special token indicating "conversation history was summarized at this point."

### 2. Pinned Context (Medium Priority)

Allow certain content to be marked as "pinned" and preserved intact through compaction:

- Results from Read tool calls to architecture/specification documents
- User-designated critical context (e.g., "this specification is critical, preserve it")
- Similar treatment to CLAUDE.md which survives compaction

**Use case:** When a user reads a 500-line architecture document, the critical 50 lines of API specifications should survive compaction intact, not be summarized to "user read architecture document."

### 3. User-Controlled Compaction (Lower Priority)

- CLI option to set compaction threshold (e.g., `--context-threshold 80%`)
- Warning before automatic compaction occurs: "Context is 90% full. Compact now or continue?"
- Ability to defer compaction until task completion
- Manual compaction command with confirmation

### 4. Compaction Summary Quality Settings

Option to control how aggressive summarization is:
- "Preserve code snippets" mode
- "Preserve API names and signatures" mode
- "Aggressive summarization" mode for simple tasks

## Current Workarounds (Inadequate)

Users and models currently work around this with:

1. **Placing all specifications in CLAUDE.md** - Doesn't scale for large projects with multiple architecture documents

2. **Manual `/compact` between tasks** - Requires user to anticipate compaction timing, which is difficult

3. **Re-stating requirements in every prompt** - Tedious, error-prone, and defeats the purpose of having documentation

4. **Post-task summaries** - Model can produce structured summaries after completing work, but this only helps if the model remembers to do it and the summary is detailed enough

## Impact

This issue particularly affects:

- **Long technical sessions** - Multi-hour implementation work
- **Projects with architecture documentation** - Where specifications are read early and must be followed throughout
- **Complex multi-file changes** - Where context about what was already done is critical
- **Users with limited time** - Wasted work due to specification drift is costly

## Environment

- Claude Code CLI
- Long-running sessions with technical documentation
- Multi-file codebases with architecture specifications

## Suggested Labels

- `enhancement`
- `context-management`
- `user-experience`

---

*Submitted by a WarpForge developer after experiencing significant time loss due to undetected context compaction during GPU backend implementation.*
