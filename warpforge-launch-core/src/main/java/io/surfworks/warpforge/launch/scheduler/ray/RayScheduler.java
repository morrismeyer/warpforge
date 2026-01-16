package io.surfworks.warpforge.launch.scheduler.ray;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.surfworks.warpforge.launch.job.GpuType;
import io.surfworks.warpforge.launch.job.InputSpec;
import io.surfworks.warpforge.launch.job.JobResult;
import io.surfworks.warpforge.launch.job.JobState;
import io.surfworks.warpforge.launch.job.JobStatus;
import io.surfworks.warpforge.launch.job.JobSubmission;
import io.surfworks.warpforge.launch.scheduler.ClusterInfo;
import io.surfworks.warpforge.launch.scheduler.JobQuery;
import io.surfworks.warpforge.launch.scheduler.NodeInfo;
import io.surfworks.warpforge.launch.scheduler.Scheduler;
import io.surfworks.warpforge.launch.scheduler.SchedulerCapabilities;
import io.surfworks.warpforge.launch.scheduler.SchedulerException;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Ray cluster scheduler using the Ray Jobs REST API.
 *
 * <p>Ray Jobs API documentation:
 * https://docs.ray.io/en/latest/cluster/running-applications/job-submission/api.html
 */
public final class RayScheduler implements Scheduler {

    private final RayConfig config;
    private final HttpClient httpClient;
    private final ObjectMapper jsonMapper;
    private volatile boolean closed;

    /**
     * Creates a Ray scheduler with the given configuration.
     */
    public RayScheduler(RayConfig config) {
        this.config = config;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(config.connectionTimeout())
                .build();
        this.jsonMapper = new ObjectMapper();
        this.closed = false;
    }

    /**
     * Creates a Ray scheduler with default local configuration.
     */
    public RayScheduler() {
        this(RayConfig.local());
    }

    @Override
    public String name() {
        return "ray";
    }

    @Override
    public SchedulerCapabilities capabilities() {
        return SchedulerCapabilities.fullFeatured(Set.of(GpuType.NVIDIA, GpuType.AMD));
    }

    @Override
    public String submit(JobSubmission submission) throws SchedulerException {
        ensureOpen();

        try {
            ObjectNode requestBody = jsonMapper.createObjectNode();

            // Generate entrypoint command
            String entrypoint = generateEntrypoint(submission);
            requestBody.put("entrypoint", entrypoint);

            // Add runtime environment with job metadata
            ObjectNode runtimeEnv = requestBody.putObject("runtime_env");
            ObjectNode envVars = runtimeEnv.putObject("env_vars");
            envVars.put("WARPFORGE_JOB_ID", submission.correlationId());
            envVars.put("WARPFORGE_JOB_NAME", submission.definition().name());

            // Add resource requirements
            ObjectNode entrypointResources = requestBody.putObject("entrypoint_resources");
            var resources = submission.definition().resources();
            entrypointResources.put("CPU", resources.cpuCores());
            if (resources.requiresGpu()) {
                entrypointResources.put("GPU", resources.gpuCount());
            }

            // Add submission ID as metadata
            ObjectNode metadata = requestBody.putObject("metadata");
            metadata.put("correlation_id", submission.correlationId());
            metadata.put("submitted_by", submission.submittedBy());

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(config.jobsApiUrl()))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonMapper.writeValueAsString(requestBody)))
                    .timeout(config.requestTimeout())
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() != 200) {
                throw new SchedulerException("Ray job submission failed (HTTP " +
                        response.statusCode() + "): " + response.body());
            }

            JsonNode result = jsonMapper.readTree(response.body());
            return result.get("submission_id").asText();

        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new SchedulerException("Failed to submit job to Ray", e);
        }
    }

    @Override
    public JobStatus status(String jobId) throws SchedulerException {
        ensureOpen();

        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(config.jobsApiUrl() + jobId))
                    .GET()
                    .timeout(config.requestTimeout())
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() == 404) {
                throw new SchedulerException("Job not found: " + jobId);
            }
            if (response.statusCode() != 200) {
                throw new SchedulerException("Failed to get job status (HTTP " +
                        response.statusCode() + "): " + response.body());
            }

            JsonNode result = jsonMapper.readTree(response.body());
            return parseJobStatus(jobId, result);

        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new SchedulerException("Failed to get job status from Ray", e);
        }
    }

    @Override
    public JobResult result(String jobId) throws SchedulerException {
        JobStatus status = status(jobId);

        if (!status.state().isTerminal()) {
            throw new IllegalStateException(
                    "Job " + jobId + " is not complete (state: " + status.state() + ")");
        }

        // Ray Jobs API logs are accessible via /logs endpoint
        // For now, return a basic result based on status
        if (status.state() == JobState.COMPLETED) {
            return JobResult.success(
                    jobId,
                    status.correlationId(),
                    List.of(), // Outputs would need to be retrieved separately
                    status.elapsed()
            );
        } else {
            return JobResult.failure(
                    jobId,
                    status.correlationId(),
                    status.message(),
                    status.elapsed()
            );
        }
    }

    @Override
    public boolean cancel(String jobId) throws SchedulerException {
        ensureOpen();

        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(config.jobsApiUrl() + jobId))
                    .DELETE()
                    .timeout(config.requestTimeout())
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

            return response.statusCode() == 200;

        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new SchedulerException("Failed to cancel job", e);
        }
    }

    @Override
    public List<JobStatus> list(JobQuery query) throws SchedulerException {
        ensureOpen();

        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(config.jobsApiUrl()))
                    .GET()
                    .timeout(config.requestTimeout())
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() != 200) {
                throw new SchedulerException("Failed to list jobs (HTTP " +
                        response.statusCode() + "): " + response.body());
            }

            JsonNode result = jsonMapper.readTree(response.body());
            List<JobStatus> statuses = new ArrayList<>();

            for (JsonNode jobNode : result) {
                String jobId = jobNode.get("submission_id").asText();
                JobStatus status = parseJobStatus(jobId, jobNode);
                if (matchesQuery(status, query)) {
                    statuses.add(status);
                }
            }

            if (query.limit() > 0 && statuses.size() > query.limit()) {
                return statuses.subList(0, query.limit());
            }
            return statuses;

        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new SchedulerException("Failed to list jobs from Ray", e);
        }
    }

    @Override
    public boolean isConnected() {
        if (closed) return false;

        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(config.dashboardUrl() + "/api/version"))
                    .GET()
                    .timeout(Duration.ofSeconds(5))
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return response.statusCode() == 200;

        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public ClusterInfo clusterInfo() throws SchedulerException {
        ensureOpen();

        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(config.dashboardUrl() + "/api/cluster_status"))
                    .GET()
                    .timeout(config.requestTimeout())
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() != 200) {
                throw new SchedulerException("Failed to get cluster info (HTTP " +
                        response.statusCode() + "): " + response.body());
            }

            JsonNode result = jsonMapper.readTree(response.body());
            return parseClusterInfo(result);

        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new SchedulerException("Failed to get cluster info from Ray", e);
        }
    }

    @Override
    public void close() {
        closed = true;
        // HttpClient doesn't need explicit close in Java 21+
    }

    private String generateEntrypoint(JobSubmission submission) {
        var def = submission.definition();

        // Build command to invoke snakegrinder
        StringBuilder cmd = new StringBuilder("snakegrinder");
        cmd.append(" --trace-with-values");
        cmd.append(" --source ").append(def.modelSource());
        cmd.append(" --class ").append(def.modelClass());
        cmd.append(" --inputs '").append(def.formatInputSpecs()).append("'");
        cmd.append(" --seed ").append(def.seed());
        cmd.append(" --out /tmp/warpforge-output");

        return cmd.toString();
    }

    private JobStatus parseJobStatus(String jobId, JsonNode node) {
        String rayStatus = node.has("status") ? node.get("status").asText() : "PENDING";
        JobState state = mapRayState(rayStatus);

        String correlationId = null;
        if (node.has("metadata") && node.get("metadata").has("correlation_id")) {
            correlationId = node.get("metadata").get("correlation_id").asText();
        }

        String message = node.has("message") ? node.get("message").asText() : rayStatus;

        long runtimeMs = node.has("runtime_ms") ? node.get("runtime_ms").asLong() : 0;
        Duration elapsed = Duration.ofMillis(runtimeMs);

        return new JobStatus(
                jobId,
                correlationId,
                state,
                Instant.now(),
                null,
                elapsed,
                message,
                Map.of("ray_status", rayStatus)
        );
    }

    private JobState mapRayState(String rayState) {
        return switch (rayState.toUpperCase()) {
            case "PENDING" -> JobState.PENDING;
            case "RUNNING" -> JobState.RUNNING;
            case "SUCCEEDED" -> JobState.COMPLETED;
            case "FAILED" -> JobState.FAILED;
            case "STOPPED" -> JobState.CANCELLED;
            default -> JobState.PENDING;
        };
    }

    private ClusterInfo parseClusterInfo(JsonNode node) {
        // Parse Ray cluster status response
        int totalNodes = 1;
        int availableNodes = 1;

        if (node.has("nodes")) {
            totalNodes = node.get("nodes").size();
            availableNodes = totalNodes; // Simplified
        }

        // Get version from separate call or default
        String version = "unknown";

        return new ClusterInfo(
                "ray",
                version,
                totalNodes,
                availableNodes,
                Map.of(GpuType.NVIDIA, 0, GpuType.AMD, 0),
                List.of(NodeInfo.cpuNode("ray-head", "ready", 16000, 8))
        );
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
        return true;
    }
}
