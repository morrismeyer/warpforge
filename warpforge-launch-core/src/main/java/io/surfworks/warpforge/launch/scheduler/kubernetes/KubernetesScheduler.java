package io.surfworks.warpforge.launch.scheduler.kubernetes;

import io.kubernetes.client.openapi.ApiClient;
import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.Configuration;
import io.kubernetes.client.openapi.apis.BatchV1Api;
import io.kubernetes.client.openapi.apis.CoreV1Api;
import io.kubernetes.client.openapi.models.V1Container;
import io.kubernetes.client.openapi.models.V1EnvVar;
import io.kubernetes.client.openapi.models.V1Job;
import io.kubernetes.client.openapi.models.V1JobList;
import io.kubernetes.client.openapi.models.V1JobSpec;
import io.kubernetes.client.openapi.models.V1JobStatus;
import io.kubernetes.client.openapi.models.V1Node;
import io.kubernetes.client.openapi.models.V1NodeList;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import io.kubernetes.client.openapi.models.V1PodSpec;
import io.kubernetes.client.openapi.models.V1PodTemplateSpec;
import io.kubernetes.client.openapi.models.V1ResourceRequirements;
import io.kubernetes.client.util.Config;
import io.surfworks.warpforge.launch.job.GpuType;
import io.surfworks.warpforge.launch.job.JobResult;
import io.surfworks.warpforge.launch.job.JobState;
import io.surfworks.warpforge.launch.job.JobStatus;
import io.surfworks.warpforge.launch.job.JobSubmission;
import io.surfworks.warpforge.launch.job.ResourceRequirements;
import io.surfworks.warpforge.launch.scheduler.ClusterInfo;
import io.surfworks.warpforge.launch.scheduler.JobQuery;
import io.surfworks.warpforge.launch.scheduler.NodeInfo;
import io.surfworks.warpforge.launch.scheduler.Scheduler;
import io.surfworks.warpforge.launch.scheduler.SchedulerCapabilities;
import io.surfworks.warpforge.launch.scheduler.SchedulerException;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Kubernetes scheduler using the Jobs API.
 *
 * <p>Creates Kubernetes Jobs for WarpForge workloads with GPU support
 * via nvidia.com/gpu and amd.com/gpu resource requests.
 */
public final class KubernetesScheduler implements Scheduler {

    private static final String LABEL_JOB_ID = "warpforge.io/job-id";
    private static final String LABEL_CORRELATION_ID = "warpforge.io/correlation-id";

    private final KubernetesConfig config;
    private final ApiClient apiClient;
    private final BatchV1Api batchApi;
    private final CoreV1Api coreApi;
    private volatile boolean closed;

    /**
     * Creates a Kubernetes scheduler with the given configuration.
     */
    public KubernetesScheduler(KubernetesConfig config) throws SchedulerException {
        this.config = config;
        try {
            if (config.kubeConfigPath() != null) {
                this.apiClient = Config.fromConfig(config.kubeConfigPath().toString());
            } else {
                this.apiClient = Config.defaultClient();
            }
            Configuration.setDefaultApiClient(apiClient);
            this.batchApi = new BatchV1Api();
            this.coreApi = new CoreV1Api();
            this.closed = false;
        } catch (IOException e) {
            throw new SchedulerException("Failed to initialize Kubernetes client", e);
        }
    }

    /**
     * Creates a Kubernetes scheduler with default configuration.
     */
    public KubernetesScheduler() throws SchedulerException {
        this(KubernetesConfig.defaults());
    }

    @Override
    public String name() {
        return "kubernetes";
    }

    @Override
    public SchedulerCapabilities capabilities() {
        return SchedulerCapabilities.fullFeatured(Set.of(GpuType.NVIDIA, GpuType.AMD));
    }

    @Override
    public String submit(JobSubmission submission) throws SchedulerException {
        ensureOpen();

        String jobName = "warpforge-" + submission.shortCorrelationId();

        try {
            V1Job job = buildJob(jobName, submission);
            V1Job created = batchApi.createNamespacedJob(
                    config.namespace(), job, null, null, null, null);
            return created.getMetadata().getName();

        } catch (ApiException e) {
            throw new SchedulerException("Failed to create Kubernetes job: " + e.getResponseBody(), e);
        }
    }

    @Override
    public JobStatus status(String jobId) throws SchedulerException {
        ensureOpen();

        try {
            V1Job job = batchApi.readNamespacedJob(jobId, config.namespace(), null);
            return mapK8sJobStatus(job);

        } catch (ApiException e) {
            if (e.getCode() == 404) {
                throw new SchedulerException("Job not found: " + jobId);
            }
            throw new SchedulerException("Failed to get job status: " + e.getResponseBody(), e);
        }
    }

    @Override
    public JobResult result(String jobId) throws SchedulerException {
        JobStatus status = status(jobId);

        if (!status.state().isTerminal()) {
            throw new IllegalStateException(
                    "Job " + jobId + " is not complete (state: " + status.state() + ")");
        }

        if (status.state() == JobState.COMPLETED) {
            return JobResult.success(
                    jobId,
                    status.correlationId(),
                    List.of(),
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
            batchApi.deleteNamespacedJob(
                    jobId, config.namespace(), null, null, null, null, null, null);
            return true;

        } catch (ApiException e) {
            if (e.getCode() == 404) {
                return false;
            }
            throw new SchedulerException("Failed to cancel job: " + e.getResponseBody(), e);
        }
    }

    @Override
    public List<JobStatus> list(JobQuery query) throws SchedulerException {
        ensureOpen();

        try {
            V1JobList jobList = batchApi.listNamespacedJob(
                    config.namespace(), null, null, null, null,
                    "warpforge.io/job-id", // Label selector
                    query.limit() > 0 ? query.limit() : null,
                    null, null, null, null);

            List<JobStatus> statuses = new ArrayList<>();
            for (V1Job job : jobList.getItems()) {
                JobStatus status = mapK8sJobStatus(job);
                if (matchesQuery(status, query)) {
                    statuses.add(status);
                }
            }
            return statuses;

        } catch (ApiException e) {
            throw new SchedulerException("Failed to list jobs: " + e.getResponseBody(), e);
        }
    }

    @Override
    public boolean isConnected() {
        if (closed) return false;

        try {
            // Simple connectivity check - list namespaces with limit 1
            coreApi.listNamespace(null, null, null, null, null, 1, null, null, null, null);
            return true;
        } catch (ApiException e) {
            return false;
        }
    }

    @Override
    public ClusterInfo clusterInfo() throws SchedulerException {
        ensureOpen();

        try {
            V1NodeList nodeList = coreApi.listNode(null, null, null, null, null, null, null, null, null, null);

            List<NodeInfo> nodes = new ArrayList<>();
            Map<GpuType, Integer> gpuCounts = new HashMap<>();
            gpuCounts.put(GpuType.NVIDIA, 0);
            gpuCounts.put(GpuType.AMD, 0);

            for (V1Node node : nodeList.getItems()) {
                String nodeName = node.getMetadata().getName();
                String status = "ready"; // Simplified

                // Check for GPU resources
                var capacity = node.getStatus().getCapacity();
                GpuType gpuType = GpuType.NONE;
                int gpuCount = 0;

                if (capacity != null) {
                    if (capacity.containsKey("nvidia.com/gpu")) {
                        gpuType = GpuType.NVIDIA;
                        gpuCount = capacity.get("nvidia.com/gpu").getNumber().intValue();
                        gpuCounts.merge(GpuType.NVIDIA, gpuCount, Integer::sum);
                    } else if (capacity.containsKey("amd.com/gpu")) {
                        gpuType = GpuType.AMD;
                        gpuCount = capacity.get("amd.com/gpu").getNumber().intValue();
                        gpuCounts.merge(GpuType.AMD, gpuCount, Integer::sum);
                    }
                }

                long memoryMb = 0;
                int cpuCores = 0;
                if (capacity != null) {
                    if (capacity.containsKey("memory")) {
                        // Parse memory (e.g., "16Gi")
                        memoryMb = parseMemory(capacity.get("memory").toSuffixedString());
                    }
                    if (capacity.containsKey("cpu")) {
                        cpuCores = capacity.get("cpu").getNumber().intValue();
                    }
                }

                nodes.add(new NodeInfo(nodeName, status, gpuType, gpuCount, memoryMb, cpuCores));
            }

            return new ClusterInfo(
                    "kubernetes",
                    "v1",
                    nodes.size(),
                    nodes.size(),
                    gpuCounts,
                    nodes
            );

        } catch (ApiException e) {
            throw new SchedulerException("Failed to get cluster info: " + e.getResponseBody(), e);
        }
    }

    @Override
    public void close() {
        closed = true;
    }

    private V1Job buildJob(String jobName, JobSubmission submission) {
        var def = submission.definition();
        var res = def.resources();

        // Build resource requirements
        Map<String, io.kubernetes.client.custom.Quantity> requests = new HashMap<>();
        Map<String, io.kubernetes.client.custom.Quantity> limits = new HashMap<>();

        requests.put("memory", new io.kubernetes.client.custom.Quantity(res.memoryMb() + "Mi"));
        requests.put("cpu", new io.kubernetes.client.custom.Quantity(String.valueOf(res.cpuCores())));
        limits.put("memory", new io.kubernetes.client.custom.Quantity(res.memoryMb() + "Mi"));
        limits.put("cpu", new io.kubernetes.client.custom.Quantity(String.valueOf(res.cpuCores())));

        if (res.requiresGpu()) {
            String gpuResource = res.gpuType() == GpuType.AMD ? "amd.com/gpu" : "nvidia.com/gpu";
            requests.put(gpuResource, new io.kubernetes.client.custom.Quantity(String.valueOf(res.gpuCount())));
            limits.put(gpuResource, new io.kubernetes.client.custom.Quantity(String.valueOf(res.gpuCount())));
        }

        // Build node selector
        Map<String, String> nodeSelector = new HashMap<>();
        if (res.gpuType() == GpuType.NVIDIA) {
            nodeSelector.put("accelerator", "nvidia");
        } else if (res.gpuType() == GpuType.AMD) {
            nodeSelector.put("accelerator", "amd");
        }

        // Build command
        List<String> command = buildCommand(submission);

        // Build environment variables
        List<V1EnvVar> envVars = new ArrayList<>();
        envVars.add(new V1EnvVar().name("WARPFORGE_JOB_ID").value(submission.correlationId()));
        def.environment().forEach((k, v) -> envVars.add(new V1EnvVar().name(k).value(v)));

        return new V1Job()
                .metadata(new V1ObjectMeta()
                        .name(jobName)
                        .namespace(config.namespace())
                        .labels(Map.of(
                                LABEL_JOB_ID, jobName,
                                LABEL_CORRELATION_ID, submission.correlationId()
                        )))
                .spec(new V1JobSpec()
                        .backoffLimit(0)
                        .template(new V1PodTemplateSpec()
                                .spec(new V1PodSpec()
                                        .restartPolicy("Never")
                                        .nodeSelector(nodeSelector.isEmpty() ? null : nodeSelector)
                                        .containers(List.of(new V1Container()
                                                .name("warpforge")
                                                .image(config.warpforgeImage())
                                                .command(command)
                                                .env(envVars)
                                                .resources(new V1ResourceRequirements()
                                                        .requests(requests)
                                                        .limits(limits)))))));
    }

    private List<String> buildCommand(JobSubmission submission) {
        var def = submission.definition();
        return List.of(
                "/opt/warpforge/bin/snakegrinder",
                "--trace-with-values",
                "--source", def.modelSource().toString(),
                "--class", def.modelClass(),
                "--inputs", def.formatInputSpecs(),
                "--seed", String.valueOf(def.seed()),
                "--out", "/data/output"
        );
    }

    private JobStatus mapK8sJobStatus(V1Job job) {
        V1JobStatus k8sStatus = job.getStatus();
        String jobId = job.getMetadata().getName();
        String correlationId = job.getMetadata().getLabels().get(LABEL_CORRELATION_ID);

        JobState state = JobState.PENDING;
        if (k8sStatus != null) {
            if (k8sStatus.getSucceeded() != null && k8sStatus.getSucceeded() > 0) {
                state = JobState.COMPLETED;
            } else if (k8sStatus.getFailed() != null && k8sStatus.getFailed() > 0) {
                state = JobState.FAILED;
            } else if (k8sStatus.getActive() != null && k8sStatus.getActive() > 0) {
                state = JobState.RUNNING;
            }
        }

        Duration elapsed = Duration.ZERO;
        if (k8sStatus != null && k8sStatus.getStartTime() != null) {
            Instant start = k8sStatus.getStartTime().toInstant();
            Instant end = k8sStatus.getCompletionTime() != null ?
                    k8sStatus.getCompletionTime().toInstant() : Instant.now();
            elapsed = Duration.between(start, end);
        }

        return new JobStatus(
                jobId,
                correlationId,
                state,
                Instant.now(),
                null,
                elapsed,
                state.name(),
                Map.of()
        );
    }

    private long parseMemory(String memory) {
        // Parse Kubernetes memory format (e.g., "16Gi", "4096Mi")
        try {
            if (memory.endsWith("Gi")) {
                return Long.parseLong(memory.replace("Gi", "")) * 1024;
            } else if (memory.endsWith("Mi")) {
                return Long.parseLong(memory.replace("Mi", ""));
            } else if (memory.endsWith("Ki")) {
                return Long.parseLong(memory.replace("Ki", "")) / 1024;
            }
            return Long.parseLong(memory) / (1024 * 1024);
        } catch (NumberFormatException e) {
            return 0;
        }
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
