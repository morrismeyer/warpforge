package io.surfworks.warpforge.launch.scheduler.local;

import io.surfworks.warpforge.launch.job.JobResult;
import io.surfworks.warpforge.launch.job.JobState;
import io.surfworks.warpforge.launch.job.JobStatus;
import io.surfworks.warpforge.launch.job.JobSubmission;
import io.surfworks.warpforge.launch.scheduler.ClusterInfo;
import io.surfworks.warpforge.launch.scheduler.JobQuery;
import io.surfworks.warpforge.launch.scheduler.Scheduler;
import io.surfworks.warpforge.launch.scheduler.SchedulerCapabilities;
import io.surfworks.warpforge.launch.scheduler.SchedulerException;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Local scheduler that runs jobs in-process.
 * Useful for development and single-node testing.
 *
 * <p>Jobs are executed using a thread pool. Each job runs the WarpForge
 * pipeline (or a placeholder for now) and returns results.
 */
public final class LocalScheduler implements Scheduler {

    private static final int DEFAULT_MAX_CONCURRENT = 4;

    private final ExecutorService executor;
    private final Map<String, LocalJob> jobs;
    private final int maxConcurrentJobs;
    private final AtomicInteger jobCounter;
    private volatile boolean closed;

    /**
     * Creates a local scheduler with default concurrency.
     */
    public LocalScheduler() {
        this(DEFAULT_MAX_CONCURRENT);
    }

    /**
     * Creates a local scheduler with the specified concurrency.
     *
     * @param maxConcurrentJobs Maximum number of jobs to run concurrently
     */
    public LocalScheduler(int maxConcurrentJobs) {
        this.maxConcurrentJobs = maxConcurrentJobs;
        this.jobs = new ConcurrentHashMap<>();
        this.jobCounter = new AtomicInteger(0);
        this.closed = false;
        this.executor = Executors.newFixedThreadPool(maxConcurrentJobs, r -> {
            Thread t = new Thread(r, "warpforge-local-" + jobCounter.incrementAndGet());
            t.setDaemon(true);
            return t;
        });
    }

    @Override
    public String name() {
        return "local";
    }

    @Override
    public SchedulerCapabilities capabilities() {
        return SchedulerCapabilities.local(maxConcurrentJobs);
    }

    @Override
    public String submit(JobSubmission submission) throws SchedulerException {
        ensureOpen();

        String jobId = "local-" + UUID.randomUUID().toString().substring(0, 8);
        LocalJob job = new LocalJob(jobId, submission);
        jobs.put(jobId, job);

        executor.submit(() -> executeJob(job));

        return jobId;
    }

    @Override
    public JobStatus status(String jobId) throws SchedulerException {
        ensureOpen();
        LocalJob job = getJob(jobId);
        return job.toStatus();
    }

    @Override
    public JobResult result(String jobId) throws SchedulerException {
        ensureOpen();
        LocalJob job = getJob(jobId);

        if (!job.state().isTerminal()) {
            throw new IllegalStateException(
                    "Job " + jobId + " is not complete (state: " + job.state() + ")");
        }

        JobResult r = job.result();
        if (r == null) {
            throw new SchedulerException("Job " + jobId + " completed but has no result");
        }
        return r;
    }

    @Override
    public boolean cancel(String jobId) throws SchedulerException {
        ensureOpen();
        LocalJob job = jobs.get(jobId);
        if (job == null) {
            return false;
        }
        job.cancel();
        return job.state() == JobState.CANCELLED;
    }

    @Override
    public List<JobStatus> list(JobQuery query) throws SchedulerException {
        ensureOpen();

        return jobs.values().stream()
                .map(LocalJob::toStatus)
                .filter(status -> matchesQuery(status, query))
                .limit(query.limit() > 0 ? query.limit() : Long.MAX_VALUE)
                .toList();
    }

    @Override
    public boolean isConnected() {
        return !closed;
    }

    @Override
    public ClusterInfo clusterInfo() {
        return ClusterInfo.local();
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        executor.shutdown();
        try {
            if (!executor.awaitTermination(10, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    private void executeJob(LocalJob job) {
        job.markRunning();

        try {
            // TODO: Integrate with PipelineExecutor when implemented
            // For now, simulate job execution
            Duration timeout = job.submission().definition().timeout();
            long sleepMs = Math.min(100, timeout.toMillis());
            Thread.sleep(sleepMs);

            // Create a placeholder successful result
            // In the real implementation, this will invoke PipelineExecutor
            JobResult result = JobResult.success(
                    job.jobId(),
                    job.submission().correlationId(),
                    List.of(), // Empty outputs for now
                    job.elapsed()
            );

            job.complete(result);

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            job.cancel();
        } catch (Exception e) {
            job.fail(e);
        }
    }

    private LocalJob getJob(String jobId) throws SchedulerException {
        LocalJob job = jobs.get(jobId);
        if (job == null) {
            throw new SchedulerException("Job not found: " + jobId);
        }
        return job;
    }

    private void ensureOpen() throws SchedulerException {
        if (closed) {
            throw new SchedulerException("Scheduler is closed");
        }
    }

    private boolean matchesQuery(JobStatus status, JobQuery query) {
        if (!query.states().isEmpty() && !query.states().contains(status.state())) {
            return false;
        }
        if (query.submittedAfter() != null &&
                status.stateChangedAt().isBefore(query.submittedAfter())) {
            return false;
        }
        if (query.submittedBefore() != null &&
                status.stateChangedAt().isAfter(query.submittedBefore())) {
            return false;
        }
        // submittedBy filter would require storing submitter in LocalJob
        return true;
    }
}
