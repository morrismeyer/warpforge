package io.surfworks.warpforge.launch.job;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the job model classes.
 */
class JobModelTest {

    // ===== GpuType tests =====

    @Test
    void gpuTypeValues() {
        assertEquals(4, GpuType.values().length);
        assertNotNull(GpuType.valueOf("NVIDIA"));
        assertNotNull(GpuType.valueOf("AMD"));
        assertNotNull(GpuType.valueOf("ANY"));
        assertNotNull(GpuType.valueOf("NONE"));
    }

    // ===== JobType tests =====

    @Test
    void jobTypeValues() {
        assertEquals(3, JobType.values().length);
        assertNotNull(JobType.valueOf("FULL_PIPELINE"));
        assertNotNull(JobType.valueOf("STABLEHLO_ONLY"));
        assertNotNull(JobType.valueOf("PYTORCH_NATIVE"));
    }

    // ===== InputSpec tests =====

    @Test
    void inputSpecCreation() {
        InputSpec spec = new InputSpec(new int[]{2, 3, 4}, "f32");
        assertArrayEquals(new int[]{2, 3, 4}, spec.shape());
        assertEquals("f32", spec.dtype());
    }

    @Test
    void inputSpecElementCount() {
        InputSpec spec = new InputSpec(new int[]{2, 3, 4}, "f32");
        assertEquals(24, spec.elementCount());
    }

    @Test
    void inputSpecNullShapeThrows() {
        assertThrows(NullPointerException.class, () ->
                new InputSpec(null, "f32"));
    }

    @Test
    void inputSpecNullDtypeThrows() {
        assertThrows(NullPointerException.class, () ->
                new InputSpec(new int[]{1}, null));
    }

    @Test
    void inputSpecEmptyShapeThrows() {
        assertThrows(IllegalArgumentException.class, () ->
                new InputSpec(new int[]{}, "f32"));
    }

    @Test
    void inputSpecFactoryMethods() {
        InputSpec f32 = InputSpec.f32(2, 3, 4);
        assertEquals("f32", f32.dtype());
        assertArrayEquals(new int[]{2, 3, 4}, f32.shape());

        InputSpec f64 = InputSpec.f64(1, 8);
        assertEquals("f64", f64.dtype());

        InputSpec i32 = InputSpec.i32(4, 4);
        assertEquals("i32", i32.dtype());

        InputSpec i64 = InputSpec.i64(8);
        assertEquals("i64", i64.dtype());
    }

    @Test
    void inputSpecToSpecString() {
        InputSpec spec = InputSpec.f32(1, 3, 224, 224);
        assertEquals("(1,3,224,224):f32", spec.toSpecString());
    }

    // ===== ResourceRequirements tests =====

    @Test
    void resourceRequirementsCpuOnly() {
        ResourceRequirements res = ResourceRequirements.cpuOnly(4, 8192);

        assertEquals(4, res.cpuCores());
        assertEquals(8192, res.memoryMb());
        assertEquals(GpuType.NONE, res.gpuType());
        assertEquals(0, res.gpuCount());
    }

    @Test
    void resourceRequirementsNvidia() {
        ResourceRequirements res = ResourceRequirements.nvidia(2, 16000, 8);

        assertEquals(GpuType.NVIDIA, res.gpuType());
        assertEquals(2, res.gpuCount());
        assertEquals(16000, res.memoryMb());
        assertEquals(8, res.cpuCores());
    }

    @Test
    void resourceRequirementsAmd() {
        ResourceRequirements res = ResourceRequirements.amd(1);

        assertEquals(GpuType.AMD, res.gpuType());
        assertEquals(1, res.gpuCount());
    }

    @Test
    void resourceRequirementsAnyGpu() {
        ResourceRequirements res = ResourceRequirements.anyGpu(4);

        assertEquals(GpuType.ANY, res.gpuType());
        assertEquals(4, res.gpuCount());
    }

    @Test
    void resourceRequirementsRequiresGpu() {
        ResourceRequirements noGpu = ResourceRequirements.cpuOnly(1, 1024);
        assertFalse(noGpu.requiresGpu());

        ResourceRequirements withGpu = ResourceRequirements.nvidia(1);
        assertTrue(withGpu.requiresGpu());
    }

    @Test
    void resourceRequirementsWithQueue() {
        ResourceRequirements res = ResourceRequirements.nvidia(1).withQueue("gpu-queue");
        assertEquals("gpu-queue", res.queue());
    }

    @Test
    void resourceRequirementsWithPriority() {
        ResourceRequirements res = ResourceRequirements.nvidia(1).withPriority(90);
        assertEquals(90, res.priority());
    }

    @Test
    void resourceRequirementsWithNodeAffinity() {
        ResourceRequirements res = ResourceRequirements.nvidia(1)
                .withNodeAffinity(Set.of("node-1", "node-2"));
        assertTrue(res.nodeAffinity().contains("node-1"));
        assertTrue(res.nodeAffinity().contains("node-2"));
    }

    // ===== JobDefinition tests =====

    @Test
    void jobDefinitionBuilder() {
        JobDefinition def = JobDefinition.builder()
                .name("test-job")
                .modelSource("/models/test.py")
                .modelClass("TestModel")
                .inputSpecs(InputSpec.f32(1, 8))
                .seed(42)
                .timeout(Duration.ofMinutes(5))
                .build();

        assertEquals("test-job", def.name());
        assertEquals("TestModel", def.modelClass());
        assertEquals(42, def.seed());
        assertEquals(Duration.ofMinutes(5), def.timeout());
    }

    @Test
    void jobDefinitionFormatInputSpecs() {
        JobDefinition def = JobDefinition.builder()
                .name("test")
                .modelSource("/test.py")
                .modelClass("Test")
                .inputSpecs(
                        InputSpec.f32(1, 8),
                        InputSpec.i64(2, 4)
                )
                .build();

        String formatted = def.formatInputSpecs();
        assertTrue(formatted.contains("(1,8):f32"));
        assertTrue(formatted.contains("(2,4):i64"));
    }

    // ===== JobSubmission tests =====

    @Test
    void jobSubmissionCreation() {
        JobDefinition def = JobDefinition.builder()
                .name("test")
                .modelSource("/test.py")
                .modelClass("Test")
                .inputSpecs(InputSpec.f32(1, 8))
                .build();

        JobSubmission submission = JobSubmission.submit(def, "user@test.com");

        assertEquals(def, submission.definition());
        assertEquals("user@test.com", submission.submittedBy());
        assertNotNull(submission.correlationId());
        assertNotNull(submission.submittedAt());
    }

    @Test
    void jobSubmissionShortCorrelationId() {
        JobDefinition def = JobDefinition.builder()
                .name("test")
                .modelSource("/test.py")
                .modelClass("Test")
                .inputSpecs(InputSpec.f32(1, 8))
                .build();

        JobSubmission submission = JobSubmission.submit(def, "user");
        String shortId = submission.shortCorrelationId();

        assertTrue(shortId.length() <= 8);
    }

    @Test
    void jobSubmissionDefaultUser() {
        JobDefinition def = JobDefinition.builder()
                .name("test")
                .modelSource("/test.py")
                .modelClass("Test")
                .inputSpecs(InputSpec.f32(1, 8))
                .build();

        JobSubmission submission = JobSubmission.submit(def);
        assertNotNull(submission.submittedBy());
        assertFalse(submission.submittedBy().isBlank());
    }

    // ===== JobState tests =====

    @Test
    void jobStateTerminalStates() {
        assertFalse(JobState.PENDING.isTerminal());
        assertFalse(JobState.RUNNING.isTerminal());
        assertTrue(JobState.COMPLETED.isTerminal());
        assertTrue(JobState.FAILED.isTerminal());
        assertTrue(JobState.CANCELLED.isTerminal());
        assertTrue(JobState.TIMEOUT.isTerminal());
    }

    // ===== JobResult tests =====

    @Test
    void jobResultSuccess() {
        JobResult result = JobResult.success("job-1", "corr-1", List.of(), Duration.ofSeconds(10));

        assertTrue(result.success());
        assertEquals("job-1", result.jobId());
        assertEquals("corr-1", result.correlationId());
        assertNull(result.errorMessage());
    }

    @Test
    void jobResultFailure() {
        JobResult result = JobResult.failure("job-1", "corr-1", "Something went wrong", Duration.ofSeconds(5));

        assertFalse(result.success());
        assertEquals("Something went wrong", result.errorMessage());
    }
}
