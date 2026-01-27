# Virtual Thread Debugging: Loom-dev Mailing List Research

This document summarizes findings from a comprehensive review of the OpenJDK loom-dev mailing list (2018-2026) regarding virtual thread debugging challenges, JDWP limitations, and observability gaps.

## Executive Summary

Virtual thread debugging in Java remains an evolving area with significant limitations. The core challenges stem from a fundamental architecture mismatch: debugging tools (JDWP, JDI, JDB) were designed for a 1:1 mapping of Java threads to OS threads. Virtual threads break this assumption with N:M multiplexing onto carrier threads.

**Key findings:**
- JDWP lacks commands for virtual thread enumeration and dumps
- Breakpoints in anonymous classes on virtual thread stacks may be skipped
- JFR events show empty thread fields for virtual threads
- IDEs (IntelliJ, Eclipse) inherit these underlying JDWP limitations
- Thread visualization doesn't scale to hundreds of thousands of virtual threads

---

## Documented Challenges

### 1. JDWP Virtual Thread Dump Limitations

**Source:** Egor Ushakov (JetBrains), Alan Bateman (Oracle) - June 2023

**Problem:** There is no JDWP command to create a virtual thread dump. The standard `VirtualMachine/AllThreads` command is inadequate for virtual threads.

**Current workaround:**
> "invoke HotSpotDiagnosticMXBean dumpThreads to generate a thread dump to a file"
> -- Alan Bateman

This works when debugger and target VM share filesystem access but fails in distributed debugging scenarios.

**Future direction:** JDWP + JDI require updates to define new commands/methods to find threads.

### 2. Breakpoints in Anonymous Classes

**Source:** Ben Berman - June 2023, tracked as IDEA-324002

**Problem:** Breakpoints placed inside anonymous class methods executing on virtual thread call stacks are unexpectedly skipped. The code executes (console output appears) but the debugger doesn't stop.

```java
// Breakpoint here is skipped on virtual threads
new Object() {
    void test() {
        System.out.println("breakpoint here does not work");
    }
}.test();
```

**Root cause:** "a buggy interaction between JDWP and virtual threads"

**Affected:** Both IntelliJ and Eclipse IDEs (inheriting from JDWP behavior)

### 3. JFR Event Thread Identification

**Source:** Jonathan Ross, Alan Bateman - March 2023

**Problem:** Events recorded on virtual threads have empty 'Event Thread' fields. Carrier threads lack default names. This causes Java Mission Control to display all virtual thread events as a single unnamed thread.

**Impact:**
- Impossible to trace which virtual thread executed specific events
- Cannot track carrier thread assignments
- Concurrent behavior patterns obscured
- Program analysis significantly hampered compared to traditional ThreadPoolExecutor/ForkJoinPool

**Suggested workaround:**
> "Frameworks/libraries that create virtual threads can set names where it makes sense"
> -- Alan Bateman

**WarpForge implication:** We MUST name our virtual threads meaningfully. See `CPU-GPU-VISIBILITY.md` for our thread naming conventions.

### 4. jstack Mounted Virtual Thread Stacks

**Source:** Inigo Mediavilla, Alan Bateman - June 2024 (JDK-8330846)

**Problem:** Adding mounted virtual thread stacks to jstack is challenging:

1. Cannot walk heap to find virtual thread objects (increases STW time)
2. Text format doesn't scale to hundreds of thousands of threads
3. ThreadContainers not accessible from jstack

**Design decision:** Only show mounted virtual threads, not all virtual threads.

**Alternative:** Use `Thread.dump_to_file` which uses the thread grouping tree.

### 5. JVMTI GetThreadState Incorrect

**Source:** Serguei Spitsyn - March 2022 (JDK-8282579)

**Problem:** JVMTI GetThreadState returned incorrect state for virtual threads waiting on a monitor.

**Fixed in:** Changeset 326c5c36 (March 5, 2022)

### 6. OPAQUE_FRAME Errors

**Source:** Chris Plummer - March 2022

**Problem:** Virtual thread debugging encounters OPAQUE_FRAME errors from JVMTI when executing:
- `PopFrame` - removing topmost stack frame
- `ForceEarlyReturn` - causing method to return prematurely
- `SetLocalXXX` - modifying local variable values

These are common debugger operations that may fail silently or throw unexpected errors on virtual threads.

### 7. Thread Event Flooding

**Source:** Chris Plummer - March 2022

**Problem:** JDB was overwhelmed by virtual thread lifecycle events (ThreadStart/ThreadDeath).

**Solution implemented:**
1. `PlatformThreadOnlyFilters` to suppress virtual thread lifecycle notifications
2. Individual `ThreadDeathRequest` objects for threads discovered during event processing
3. `-trackallvthreads` flag to revert to comprehensive tracking when needed

### 8. Virtual Thread Enumeration Gap

**Source:** Chris Plummer - March 2022

**Problem:** Virtual threads don't appear in traditional `ThreadGroup` hierarchies, creating visibility gaps for debuggers.

**Solution implemented:**
1. Dedicated virtual thread tracking list
2. Modified `ThreadIterator.next()` to transition between ThreadGroup traversal and virtual thread enumeration
3. New `getThreadIdsByName()` method for thread location by naming convention

### 9. JVMTI Suspend/Resume Independence

**Source:** Serguei Spitsyn - September 2024

**Key clarification:** JVMTI suspend/resume operations act ONLY on the target thread. There is no automatic cascade between virtual threads and carrier threads.

| Operation | Effect |
|-----------|--------|
| SuspendThread(virtualThread) | Carrier blocked waiting for yield, NOT suspended |
| SuspendThread(carrierThread) | Virtual thread continues at unmount point |
| ResumeThread(virtualThread) | Only resumes targeted virtual thread |
| ResumeThread(carrierThread) | No effect on mounted virtual thread's suspension |

**WarpForge implication:** When debugging GPU operations, suspending a virtual thread waiting on GPU completion won't suspend the underlying carrier - which may still be blocked on FFM calls.

---

## Timeline of Debugging Support Development

| Date | Milestone |
|------|-----------|
| Dec 2018 | Initial debugger support for fibers (Chris Plummer) |
| Jun 2019 | Continuation events and debugging helpers added |
| Dec 2019 | JVMTI GetThreadInfo virtual thread support |
| Dec 2020 | Single-step event fixes, frame pop handling |
| Mar 2021 | JDB tests updated for vthreads, JVMTI event renaming |
| Sep 2021 | JDWP/JDI/JDB test failures systematically resolved |
| Mar 2022 | Major JFR refresh, JDB vthread support, JVMTI fixes |
| Jun 2023 | JDWP vthread dump limitations documented |
| Jun 2024 | jstack mounted vthread stacks (JDK-8330846) |

---

## Implications for WarpForge

### Naming Conventions

Given JFR's empty thread field problem, WarpForge MUST name virtual threads meaningfully:

```java
// Good: Descriptive name with scope context
Thread.ofVirtual()
    .name("warpforge-gpu-scope-" + scopeId + "-task-" + taskId)
    .start(task);

// Bad: Anonymous or generic names
Thread.ofVirtual().start(task);
```

### GPU Operation Debugging

When a virtual thread blocks on GPU operations (via FFM):

1. The carrier thread is blocked in native code
2. Suspending the virtual thread doesn't help - work is on GPU
3. Need GPU-side visibility (see `CPU-GPU-VISIBILITY.md`) for actual debugging
4. JFR events from GPU backends must include virtual thread context

### Recommended Debugging Approach

1. **Use JFR over JDWP** for virtual thread profiling - JFR handles scale better
2. **Name all virtual threads** with scope/task identifiers
3. **Log GPU operations** with matching virtual thread IDs
4. **Use Thread.dump_to_file** instead of jstack for full thread dumps
5. **Avoid anonymous classes** in performance-critical virtual thread paths

### Future Integration Points

| WarpForge Component | Debugging Integration |
|---------------------|----------------------|
| GpuTaskScope | Emit JFR events with scope ID, link to virtual thread |
| GpuLease | Track stream handle → virtual thread mapping |
| TimeSlicedKernel | Log chunk execution with thread context |
| DeadlineContext | Record deadline violations with thread ID |

---

## Known IDE Limitations

### IntelliJ IDEA

- Breakpoints in anonymous classes may be skipped on virtual threads (IDEA-324002)
- Virtual thread enumeration can be slow/incomplete
- Thread view may not show all virtual threads

### Eclipse

- Same JDWP-inherited limitations as IntelliJ
- Virtual thread debugging support varies by version

### VisualVM / JMC

- JFR events may show empty thread fields
- Thread dumps may not include all virtual threads
- Visualization doesn't scale to high thread counts

---

## Recommendations for OpenJDK

Based on this research, WarpForge would benefit from these JDWP/JDI enhancements:

1. **New JDWP command:** `VirtualMachine/AllVirtualThreads` - enumerate all virtual threads
2. **New JDWP command:** `VirtualMachine/DumpVirtualThreads` - generate virtual thread dump
3. **Fix:** Anonymous class breakpoints on virtual thread stacks
4. **JFR enhancement:** Always populate Event Thread for virtual threads
5. **JMC enhancement:** Group events by virtual thread, show carrier relationships

---

## References

- loom-dev mailing list: https://mail.openjdk.org/pipermail/loom-dev/
- JDK-8282579: JVMTI GetThreadState incorrect for vthread on monitor
- JDK-8330846: jstack add stacks of mounted virtual threads
- IDEA-324002: Breakpoints in anonymous classes on virtual thread stacks
- WarpForge CPU-GPU Visibility: `architecture/CPU-GPU-VISIBILITY.md`

---

## Research Conducted

- **Date:** 2026-01-27
- **Scope:** All loom-dev archives from January 2018 through January 2026
- **Focus:** Debugging, JDWP, JDI, JVMTI, JFR, visualization, IDE integration
- **Key contributors cited:** Alan Bateman, Chris Plummer, Serguei Spitsyn, Egor Ushakov (JetBrains), Jonathan Ross, Ben Berman, Inigo Mediavilla
