package io.surfworks.warpforge.launch.testing;

import io.surfworks.warpforge.launch.job.InputSpec;
import io.surfworks.warpforge.launch.job.JobDefinition;
import io.surfworks.warpforge.launch.job.JobResult;
import io.surfworks.warpforge.launch.job.JobState;
import io.surfworks.warpforge.launch.job.JobStatus;
import io.surfworks.warpforge.launch.job.JobSubmission;
import io.surfworks.warpforge.launch.scheduler.JobQuery;
import io.surfworks.warpforge.launch.scheduler.SchedulerException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for MockScheduler.
 */
class MockSchedulerTest {

    private MockScheduler scheduler;
    private JobSubmission testSubmission;

    @BeforeEach
    void setUp() {
        scheduler = new MockScheduler();
        JobDefinition def = JobDefinition.builder()
                .name("test-job")
                .modelSource("/test.py")
                .modelClass("TestModel")
                .inputSpecs(InputSpec.f32(1, 8))
                .build();
        testSubmission = JobSubmission.submit(def, "test-user");
    }

    @Test
    void submitReturnsJobId() throws SchedulerException {
        String jobId = scheduler.submit(testSubmission);
        assertNotNull(jobId);
        assertTrue(jobId.startsWith("mock-job-"));
    }

    @Test
    void statusReturnsSubmittedJob() throws SchedulerException {
        String jobId = scheduler.submit(testSubmission);
        JobStatus status = scheduler.status(jobId);

        assertEquals(jobId, status.jobId());
        assertEquals(testSubmission.correlationId(), status.correlationId());
    }

    @Test
    void statusOfNonexistentJobThrows() {
        assertThrows(SchedulerException.class, () ->
                scheduler.status("nonexistent-job"));
    }

    @Test
    void defaultOutcomeIsCompleted() throws SchedulerException {
        String jobId = scheduler.submit(testSubmission);
        JobStatus status = scheduler.status(jobId);

        assertEquals(JobState.COMPLETED, status.state());
    }

    @Test
    void setDefaultOutcome() throws SchedulerException {
        scheduler.setDefaultOutcome(JobState.FAILED);
        String jobId = scheduler.submit(testSubmission);
        JobStatus status = scheduler.status(jobId);

        assertEquals(JobState.FAILED, status.state());
    }

    @Test
    void resultForCompletedJob() throws SchedulerException {
        String jobId = scheduler.submit(testSubmission);
        JobResult result = scheduler.result(jobId);

        assertTrue(result.success());
        assertEquals(jobId, result.jobId());
    }

    @Test
    void resultForFailedJob() throws SchedulerException {
        scheduler.setDefaultOutcome(JobState.FAILED);
        String jobId = scheduler.submit(testSubmission);
        JobResult result = scheduler.result(jobId);

        assertFalse(result.success());
    }

    @Test
    void cancelJob() throws SchedulerException {
        scheduler.setDefaultDelay(Duration.ofHours(1)); // Won't complete immediately
        String jobId = scheduler.submit(testSubmission);

        boolean cancelled = scheduler.cancel(jobId);
        assertTrue(cancelled);

        JobStatus status = scheduler.status(jobId);
        assertEquals(JobState.CANCELLED, status.state());
    }

    @Test
    void cancelNonexistentJobReturnsFalse() throws SchedulerException {
        boolean cancelled = scheduler.cancel("nonexistent");
        assertFalse(cancelled);
    }

    @Test
    void listJobs() throws SchedulerException {
        scheduler.submit(testSubmission);
        scheduler.submit(testSubmission);
        scheduler.submit(testSubmission);

        List<JobStatus> jobs = scheduler.list(JobQuery.all());
        assertEquals(3, jobs.size());
    }

    @Test
    void listJobsWithStateFilter() throws SchedulerException {
        scheduler.setDefaultOutcome(JobState.COMPLETED);
        scheduler.submit(testSubmission);

        scheduler.setDefaultOutcome(JobState.FAILED);
        scheduler.submit(testSubmission);

        // Use the completed() factory which includes COMPLETED, FAILED, etc.
        List<JobStatus> terminal = scheduler.list(JobQuery.completed());
        assertEquals(2, terminal.size());

        // Use running() factory
        List<JobStatus> running = scheduler.list(JobQuery.running());
        assertEquals(0, running.size()); // All are terminal
    }

    @Test
    void listJobsWithLimit() throws SchedulerException {
        for (int i = 0; i < 10; i++) {
            scheduler.submit(testSubmission);
        }

        List<JobStatus> limited = scheduler.list(JobQuery.all().withLimit(5));
        assertEquals(5, limited.size());
    }

    @Test
    void isConnectedDefault() {
        assertTrue(scheduler.isConnected());
    }

    @Test
    void setConnected() {
        scheduler.setConnected(false);
        assertFalse(scheduler.isConnected());
    }

    @Test
    void submitExceptionThrows() {
        scheduler.setSubmitException(new SchedulerException("Test error"));
        assertThrows(SchedulerException.class, () ->
                scheduler.submit(testSubmission));
    }

    @Test
    void getSubmissionsTracksAllSubmissions() throws SchedulerException {
        scheduler.submit(testSubmission);
        scheduler.submit(testSubmission);

        assertEquals(2, scheduler.getSubmissionCount());
        assertEquals(2, scheduler.getSubmissions().size());
    }

    @Test
    void clearSubmissionsResetsState() throws SchedulerException {
        scheduler.submit(testSubmission);
        scheduler.clearSubmissions();

        assertEquals(0, scheduler.getSubmissionCount());
    }

    @Test
    void hasSubmissionByCorrelationId() throws SchedulerException {
        scheduler.submit(testSubmission);

        assertTrue(scheduler.hasSubmission(testSubmission.correlationId()));
        assertFalse(scheduler.hasSubmission("other-correlation-id"));
    }

    @Test
    void clusterInfoReturnsValidInfo() throws SchedulerException {
        var info = scheduler.clusterInfo();

        assertEquals("mock", info.schedulerName());
        assertEquals(1, info.totalNodes());
        assertFalse(info.nodes().isEmpty());
    }

    @Test
    void completeJobManually() throws SchedulerException {
        scheduler.setDefaultDelay(Duration.ofHours(1)); // Won't complete automatically
        String jobId = scheduler.submit(testSubmission);

        // Initially pending/running
        JobStatus before = scheduler.status(jobId);
        assertFalse(before.state().isTerminal());

        // Complete manually
        scheduler.completeJob(jobId, JobState.COMPLETED);

        JobStatus after = scheduler.status(jobId);
        assertEquals(JobState.COMPLETED, after.state());
    }
}
