package io.surfworks.warpforge.launch.cli;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import io.surfworks.warpforge.core.backend.GpuDetector;
import io.surfworks.warpforge.launch.artifact.ArtifactException;
import io.surfworks.warpforge.launch.artifact.ArtifactRegistry;
import io.surfworks.warpforge.launch.artifact.ArtifactStore;
import io.surfworks.warpforge.launch.config.LaunchConfig;
import io.surfworks.warpforge.launch.config.LaunchConfigLoader;
import io.surfworks.warpforge.launch.job.GpuType;
import io.surfworks.warpforge.launch.job.InputSpec;
import io.surfworks.warpforge.launch.job.JobDefinition;
import io.surfworks.warpforge.launch.job.JobResult;
import io.surfworks.warpforge.launch.job.JobState;
import io.surfworks.warpforge.launch.job.JobStatus;
import io.surfworks.warpforge.launch.job.JobSubmission;
import io.surfworks.warpforge.launch.job.ResourceRequirements;
import io.surfworks.warpforge.launch.pipeline.ParityResult;
import io.surfworks.warpforge.launch.pipeline.ParityTester;
import io.surfworks.warpforge.launch.pipeline.PipelineException;
import io.surfworks.warpforge.launch.scheduler.ClusterInfo;
import io.surfworks.warpforge.launch.scheduler.JobQuery;
import io.surfworks.warpforge.launch.scheduler.NodeInfo;
import io.surfworks.warpforge.launch.scheduler.Scheduler;
import io.surfworks.warpforge.launch.scheduler.SchedulerException;
import io.surfworks.warpforge.launch.scheduler.SchedulerRegistry;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * WarpForge Launch CLI - Distributed job submission tool.
 *
 * <p>Commands:
 * <ul>
 *   <li>submit - Submit a job to the scheduler</li>
 *   <li>status - Get job status</li>
 *   <li>result - Get job result</li>
 *   <li>cancel - Cancel a job</li>
 *   <li>list - List jobs</li>
 *   <li>cluster-info - Show cluster topology</li>
 *   <li>parity-test - Run PyTorch vs WarpForge comparison</li>
 *   <li>config - Show/set configuration</li>
 * </ul>
 */
public class WarpforgeLaunchCli {

    private static final String VERSION = "0.1.0";
    private static final ObjectMapper JSON = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);
    private static final DateTimeFormatter TIME_FORMAT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss").withZone(ZoneId.systemDefault());

    public static void main(String[] args) {
        if (args.length == 0) {
            printHelp();
            return;
        }

        String command = args[0];

        // Handle global flags (only when they're the command itself)
        if (command.equals("--help") || command.equals("-h")) {
            printHelp();
            return;
        }
        if (command.equals("--version") || command.equals("-v")) {
            System.out.println("warpforge-launch " + VERSION);
            return;
        }
        String[] commandArgs = Arrays.copyOfRange(args, 1, args.length);

        try {
            switch (command) {
                case "submit" -> handleSubmit(commandArgs);
                case "status" -> handleStatus(commandArgs);
                case "result" -> handleResult(commandArgs);
                case "cancel" -> handleCancel(commandArgs);
                case "list" -> handleList(commandArgs);
                case "cluster-info" -> handleClusterInfo(commandArgs);
                case "parity-test" -> handleParityTest(commandArgs);
                case "config" -> handleConfig(commandArgs);
                case "gpu-info" -> handleGpuInfo(commandArgs);
                default -> {
                    System.err.println("Unknown command: " + command);
                    System.err.println("Run 'warpforge-launch --help' for usage.");
                    System.exit(1);
                }
            }
        } catch (SchedulerException e) {
            System.err.println("Scheduler error: " + e.getMessage());
            System.exit(1);
        } catch (ArtifactException e) {
            System.err.println("Artifact error: " + e.getMessage());
            System.exit(1);
        } catch (PipelineException e) {
            System.err.println("Pipeline error: " + e.getMessage());
            System.exit(1);
        } catch (IOException e) {
            System.err.println("I/O error: " + e.getMessage());
            System.exit(1);
        } catch (IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    private static void handleSubmit(String[] args) throws SchedulerException, IOException {
        if (hasFlag(args, "--help")) {
            printSubmitHelp();
            return;
        }

        String source = getFlagValue(args, "--source");
        String modelClass = getFlagValue(args, "--class");
        String inputs = getFlagValue(args, "--inputs");
        String schedulerName = getFlagValue(args, "--scheduler");
        String gpuStr = getFlagValue(args, "--gpu");
        String gpuCountStr = getFlagValue(args, "--gpu-count");
        String timeoutStr = getFlagValue(args, "--timeout");
        boolean wait = hasFlag(args, "--wait");
        boolean json = hasFlag(args, "--json");

        if (source == null) {
            System.err.println("Error: --source is required");
            printSubmitHelp();
            System.exit(1);
        }

        // Parse inputs
        List<InputSpec> inputSpecs = parseInputSpecs(inputs);

        // Parse GPU requirements
        GpuType gpuType = gpuStr != null ? GpuType.valueOf(gpuStr.toUpperCase()) : GpuType.ANY;
        int gpuCount = gpuCountStr != null ? Integer.parseInt(gpuCountStr) : 1;

        // Parse timeout
        Duration timeout = timeoutStr != null ?
                Duration.ofMinutes(Long.parseLong(timeoutStr)) : Duration.ofMinutes(30);

        // Build job definition
        JobDefinition.Builder defBuilder = JobDefinition.builder()
                .name(Path.of(source).getFileName().toString().replace(".py", ""))
                .modelSource(source)
                .timeout(timeout);

        if (modelClass != null) {
            defBuilder.modelClass(modelClass);
        }

        if (!inputSpecs.isEmpty()) {
            defBuilder.inputSpecs(inputSpecs.toArray(new InputSpec[0]));
        }

        if (gpuType != GpuType.NONE && gpuCount > 0) {
            ResourceRequirements resources = switch (gpuType) {
                case NVIDIA -> ResourceRequirements.nvidia(gpuCount);
                case AMD -> ResourceRequirements.amd(gpuCount);
                case ANY -> ResourceRequirements.anyGpu(gpuCount);
                default -> ResourceRequirements.cpuOnly(4, 8192);
            };
            defBuilder.resources(resources);
        }

        JobDefinition definition = defBuilder.build();
        JobSubmission submission = JobSubmission.submit(definition);

        // Get scheduler
        LaunchConfig config = LaunchConfigLoader.load();
        Scheduler scheduler = getScheduler(schedulerName, config);

        // Submit job
        String jobId = scheduler.submit(submission);

        if (json) {
            System.out.println(JSON.writeValueAsString(new SubmitResponse(jobId, submission.correlationId())));
        } else {
            System.out.println("Job submitted: " + jobId);
            System.out.println("Correlation ID: " + submission.correlationId());
        }

        // Wait for completion if requested
        if (wait) {
            System.out.println("Waiting for job to complete...");
            JobResult result = scheduler.awaitCompletion(jobId, timeout);
            printJobResult(result, json);
        }
    }

    private static void handleStatus(String[] args) throws SchedulerException, IOException {
        if (args.length == 0 || hasFlag(args, "--help")) {
            System.out.println("Usage: warpforge-launch status <job-id> [--scheduler <name>] [--json]");
            if (args.length == 0) System.exit(1);
            return;
        }

        String jobId = args[0];
        String schedulerName = getFlagValue(args, "--scheduler");
        boolean json = hasFlag(args, "--json");

        LaunchConfig config = LaunchConfigLoader.load();
        Scheduler scheduler = getScheduler(schedulerName, config);

        JobStatus status = scheduler.status(jobId);
        printJobStatus(status, json);
    }

    private static void handleResult(String[] args) throws SchedulerException, IOException {
        if (args.length == 0 || hasFlag(args, "--help")) {
            System.out.println("Usage: warpforge-launch result <job-id> [--scheduler <name>] [--json]");
            if (args.length == 0) System.exit(1);
            return;
        }

        String jobId = args[0];
        String schedulerName = getFlagValue(args, "--scheduler");
        boolean json = hasFlag(args, "--json");

        LaunchConfig config = LaunchConfigLoader.load();
        Scheduler scheduler = getScheduler(schedulerName, config);

        JobResult result = scheduler.result(jobId);
        printJobResult(result, json);
    }

    private static void handleCancel(String[] args) throws SchedulerException, IOException {
        if (args.length == 0 || hasFlag(args, "--help")) {
            System.out.println("Usage: warpforge-launch cancel <job-id> [--scheduler <name>]");
            if (args.length == 0) System.exit(1);
            return;
        }

        String jobId = args[0];
        String schedulerName = getFlagValue(args, "--scheduler");

        LaunchConfig config = LaunchConfigLoader.load();
        Scheduler scheduler = getScheduler(schedulerName, config);

        boolean cancelled = scheduler.cancel(jobId);
        if (cancelled) {
            System.out.println("Job " + jobId + " cancelled.");
        } else {
            System.out.println("Job " + jobId + " could not be cancelled (may have already completed).");
        }
    }

    private static void handleList(String[] args) throws SchedulerException, IOException {
        if (hasFlag(args, "--help")) {
            System.out.println("Usage: warpforge-launch list [--state <state>] [--limit <n>] [--scheduler <name>] [--json]");
            System.out.println("States: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT");
            return;
        }

        String stateStr = getFlagValue(args, "--state");
        String limitStr = getFlagValue(args, "--limit");
        String schedulerName = getFlagValue(args, "--scheduler");
        boolean json = hasFlag(args, "--json");

        LaunchConfig config = LaunchConfigLoader.load();
        Scheduler scheduler = getScheduler(schedulerName, config);

        JobQuery query = JobQuery.all();
        if (stateStr != null) {
            JobState state = JobState.valueOf(stateStr.toUpperCase());
            query = new JobQuery(java.util.Set.of(state), null, null, null, 0);
        }
        if (limitStr != null) {
            query = query.withLimit(Integer.parseInt(limitStr));
        }

        List<JobStatus> jobs = scheduler.list(query);

        if (json) {
            System.out.println(JSON.writeValueAsString(jobs));
        } else {
            if (jobs.isEmpty()) {
                System.out.println("No jobs found.");
            } else {
                System.out.println("Jobs (" + jobs.size() + "):");
                System.out.println("-".repeat(80));
                for (JobStatus status : jobs) {
                    printJobStatusCompact(status);
                }
            }
        }
    }

    private static void handleClusterInfo(String[] args) throws SchedulerException, IOException {
        if (hasFlag(args, "--help")) {
            System.out.println("Usage: warpforge-launch cluster-info [--scheduler <name>] [--json]");
            return;
        }

        String schedulerName = getFlagValue(args, "--scheduler");
        boolean json = hasFlag(args, "--json");

        LaunchConfig config = LaunchConfigLoader.load();
        Scheduler scheduler = getScheduler(schedulerName, config);

        ClusterInfo info = scheduler.clusterInfo();

        if (json) {
            System.out.println(JSON.writeValueAsString(info));
        } else {
            System.out.println("Cluster: " + info.schedulerName());
            System.out.println("Nodes: " + info.totalNodes() + " total, " + info.availableNodes() + " available");
            System.out.println("GPU Counts: " + info.gpuCounts());
            System.out.println();
            System.out.println("Nodes:");
            System.out.println("-".repeat(60));
            for (NodeInfo node : info.nodes()) {
                System.out.printf("  %-20s  %-10s  %s GPUs  %s%n",
                        node.name(),
                        node.status(),
                        node.gpuCount(),
                        node.gpuType());
            }
        }
    }

    private static void handleParityTest(String[] args)
            throws SchedulerException, ArtifactException, PipelineException, IOException {
        if (hasFlag(args, "--help")) {
            printParityTestHelp();
            return;
        }

        String source = getFlagValue(args, "--source");
        String modelClass = getFlagValue(args, "--class");
        String inputs = getFlagValue(args, "--inputs");
        String toleranceStr = getFlagValue(args, "--tolerance");
        boolean json = hasFlag(args, "--json");

        if (source == null) {
            System.err.println("Error: --source is required");
            printParityTestHelp();
            System.exit(1);
        }

        List<InputSpec> inputSpecs = parseInputSpecs(inputs);

        JobDefinition.Builder defBuilder = JobDefinition.builder()
                .name(Path.of(source).getFileName().toString().replace(".py", ""))
                .modelSource(source);

        if (modelClass != null) {
            defBuilder.modelClass(modelClass);
        }
        if (!inputSpecs.isEmpty()) {
            defBuilder.inputSpecs(inputSpecs.toArray(new InputSpec[0]));
        }

        JobDefinition definition = defBuilder.build();

        LaunchConfig config = LaunchConfigLoader.load();
        double tolerance = toleranceStr != null ? Double.parseDouble(toleranceStr) : ParityTester.DEFAULT_TOLERANCE;

        ParityTester tester = new ParityTester(config, tolerance);
        ParityResult result = tester.test(definition);

        if (json) {
            System.out.println(JSON.writeValueAsString(result));
        } else {
            System.out.println("Parity Test Result");
            System.out.println("-".repeat(40));
            System.out.println("Match: " + (result.matches() ? "YES" : "NO"));
            System.out.println("Tolerance: " + result.tolerance());
            double maxDiff = result.differences().stream()
                    .mapToDouble(ParityResult.OutputDifference::maxDifference)
                    .max().orElse(0.0);
            System.out.println("Max Difference: " + maxDiff);
            System.out.println("PyTorch Time: " + result.pytorchTime().toMillis() + "ms");
            System.out.println("WarpForge Time: " + result.warpforgeTime().toMillis() + "ms");

            if (!result.matches() && result.errorMessage() != null) {
                System.out.println("Error: " + result.errorMessage());
            }
        }
    }

    private static void handleGpuInfo(String[] args) throws IOException {
        if (hasFlag(args, "--help")) {
            System.out.println("Usage: warpforge-launch gpu-info [--json] [--install]");
            System.out.println();
            System.out.println("Display detected GPUs and backend installation status.");
            System.out.println();
            System.out.println("Options:");
            System.out.println("  --json       Output as JSON");
            System.out.println("  --install    Install backend for detected GPU");
            return;
        }

        boolean json = hasFlag(args, "--json");
        boolean install = hasFlag(args, "--install");

        var gpus = GpuDetector.detectAll();

        if (json) {
            System.out.println(JSON.writeValueAsString(gpus));
        } else {
            if (gpus.isEmpty()) {
                System.out.println("No GPUs detected.");
                System.out.println();
                System.out.println("Supported GPUs:");
                System.out.println("  - NVIDIA GPUs (CUDA)");
                System.out.println("  - AMD GPUs (ROCm)");
                System.out.println();
                System.out.println("If you have a GPU that's not detected:");
                System.out.println("  - NVIDIA: Ensure nvidia-smi is available");
                System.out.println("  - AMD: Ensure rocm-smi is available");
            } else {
                System.out.println("Detected GPUs:");
                System.out.println("=".repeat(50));

                for (var gpu : gpus) {
                    System.out.printf("%s GPU %d: %s%n", gpu.vendor(), gpu.deviceIndex(), gpu.name());
                    System.out.printf("  Memory:  %d MB%n", gpu.totalMemoryBytes() / (1024 * 1024));
                    System.out.printf("  Driver:  %s%n", gpu.driverVersion());

                    boolean installed = gpu.isBackendInstalled();
                    System.out.printf("  Backend: %s (%s)%n",
                            gpu.vendor().backendName(),
                            installed ? "installed" : "not installed");

                    if (!installed && install) {
                        System.out.println();
                        System.out.println("Installing " + gpu.vendor().backendName() + " backend...");
                        try {
                            GpuDetector.ensureBackendInstalled(gpu.vendor());
                            System.out.println("Backend installed successfully.");
                        } catch (GpuDetector.BackendNotAvailableException e) {
                            System.err.println("Failed to install backend: " + e.getMessage());
                        }
                    }
                    System.out.println();
                }

                // Show summary
                boolean anyNotInstalled = gpus.stream().anyMatch(g -> !g.isBackendInstalled());
                if (anyNotInstalled && !install) {
                    System.out.println("To install missing backends, run:");
                    System.out.println("  warpforge gpu-info --install");
                    System.out.println("Or use the download script:");
                    System.out.println("  download-backend.sh auto");
                }
            }
        }
    }

    private static void handleConfig(String[] args) throws IOException {
        if (hasFlag(args, "--help")) {
            System.out.println("Usage: warpforge-launch config [--show] [--set <key>=<value>] [--json]");
            System.out.println();
            System.out.println("Options:");
            System.out.println("  --show         Show current configuration");
            System.out.println("  --set          Set a configuration value");
            System.out.println("  --json         Output as JSON");
            System.out.println();
            System.out.println("Config keys:");
            System.out.println("  defaultScheduler    - Default scheduler (local, ray, kubernetes, slurm)");
            System.out.println("  defaultArtifactStore - Default artifact store (local, shared)");
            System.out.println("  ray.dashboardUrl    - Ray dashboard URL");
            System.out.println("  kubernetes.namespace - Kubernetes namespace");
            System.out.println("  slurm.host          - Slurm head node host");
            return;
        }

        boolean json = hasFlag(args, "--json");
        String setValue = getFlagValue(args, "--set");

        LaunchConfig config = LaunchConfigLoader.load();

        if (setValue != null) {
            String[] parts = setValue.split("=", 2);
            if (parts.length != 2) {
                System.err.println("Invalid format. Use --set key=value");
                System.exit(1);
            }
            String key = parts[0];
            String value = parts[1];

            config = switch (key) {
                case "defaultScheduler" -> config.withScheduler(value);
                case "defaultArtifactStore" -> config.withArtifactStore(value);
                default -> {
                    System.err.println("Unknown config key: " + key);
                    System.exit(1);
                    yield config;
                }
            };

            LaunchConfigLoader.save(config);
            System.out.println("Configuration updated.");
        }

        // Show current config
        if (json) {
            System.out.println(JSON.writeValueAsString(config));
        } else {
            System.out.println("Configuration:");
            System.out.println("-".repeat(40));
            System.out.println("Default Scheduler: " + config.defaultScheduler());
            System.out.println("Default Artifact Store: " + config.defaultArtifactStore());
            System.out.println("Shared Directory: " + (config.sharedDir() != null ? config.sharedDir() : "(not set)"));
            System.out.println("SnakeGrinder Path: " + (config.snakegrinderPath() != null ? config.snakegrinderPath() : "(auto-detect)"));
            if (config.rayConfig() != null) {
                System.out.println("Ray Dashboard: " + config.rayConfig().dashboardUrl());
            }
            if (config.kubernetesConfig() != null) {
                System.out.println("Kubernetes Namespace: " + config.kubernetesConfig().namespace());
            }
            if (config.slurmConfig() != null) {
                System.out.println("Slurm Host: " + config.slurmConfig().sshHost());
            }
        }
    }

    // ===== Helper methods =====

    private static Scheduler getScheduler(String name, LaunchConfig config) {
        String schedulerName = name != null ? name : config.defaultScheduler();
        if (!SchedulerRegistry.isRegistered(schedulerName)) {
            throw new IllegalArgumentException("Unknown scheduler: " + schedulerName +
                    ". Available: " + SchedulerRegistry.available());
        }
        return SchedulerRegistry.get(schedulerName);
    }

    private static ArtifactStore getArtifactStore(LaunchConfig config) {
        String storeName = config.defaultArtifactStore();
        return ArtifactRegistry.instance().get(storeName)
                .orElseThrow(() -> new IllegalArgumentException(
                        "Unknown artifact store: " + storeName +
                                ". Available: " + ArtifactRegistry.instance().available()));
    }

    private static List<InputSpec> parseInputSpecs(String inputs) {
        List<InputSpec> specs = new ArrayList<>();
        if (inputs == null || inputs.isBlank()) {
            return specs;
        }

        // Parse format: [(1,3,224,224):f32, (1,8):i64]
        String trimmed = inputs.trim();
        if (trimmed.startsWith("[")) {
            trimmed = trimmed.substring(1);
        }
        if (trimmed.endsWith("]")) {
            trimmed = trimmed.substring(0, trimmed.length() - 1);
        }

        for (String part : trimmed.split(",\\s*(?=\\()")) {
            part = part.trim();
            if (part.isEmpty()) continue;

            // Parse (shape):dtype or just (shape)
            String shapeStr;
            String dtype = "f32";

            int colonIdx = part.lastIndexOf(':');
            if (colonIdx > 0 && !part.substring(colonIdx).contains(")")) {
                dtype = part.substring(colonIdx + 1).trim();
                shapeStr = part.substring(0, colonIdx).trim();
            } else {
                shapeStr = part;
            }

            // Remove parentheses from shape
            if (shapeStr.startsWith("(")) {
                shapeStr = shapeStr.substring(1);
            }
            if (shapeStr.endsWith(")")) {
                shapeStr = shapeStr.substring(0, shapeStr.length() - 1);
            }

            // Parse dimensions
            String[] dims = shapeStr.split(",");
            int[] shape = new int[dims.length];
            for (int i = 0; i < dims.length; i++) {
                shape[i] = Integer.parseInt(dims[i].trim());
            }

            specs.add(new InputSpec(shape, dtype));
        }

        return specs;
    }

    private static void printJobStatus(JobStatus status, boolean json) throws IOException {
        if (json) {
            System.out.println(JSON.writeValueAsString(status));
        } else {
            System.out.println("Job Status");
            System.out.println("-".repeat(40));
            System.out.println("Job ID: " + status.jobId());
            System.out.println("Correlation ID: " + status.correlationId());
            System.out.println("State: " + status.state());
            System.out.println("Node: " + (status.nodeName() != null ? status.nodeName() : "(not assigned)"));
            if (status.stateChangedAt() != null) {
                System.out.println("State Changed: " + TIME_FORMAT.format(status.stateChangedAt()));
            }
            if (status.elapsed() != null) {
                System.out.println("Elapsed: " + formatDuration(status.elapsed()));
            }
            if (status.message() != null) {
                System.out.println("Message: " + status.message());
            }
        }
    }

    private static void printJobStatusCompact(JobStatus status) {
        String time = status.stateChangedAt() != null ?
                TIME_FORMAT.format(status.stateChangedAt()) : "";
        System.out.printf("%-20s  %-12s  %-15s  %s%n",
                status.jobId(),
                status.state(),
                status.nodeName() != null ? status.nodeName() : "-",
                time);
    }

    private static void printJobResult(JobResult result, boolean json) throws IOException {
        if (json) {
            System.out.println(JSON.writeValueAsString(result));
        } else {
            System.out.println("Job Result");
            System.out.println("-".repeat(40));
            System.out.println("Job ID: " + result.jobId());
            System.out.println("Correlation ID: " + result.correlationId());
            System.out.println("Success: " + result.success());
            System.out.println("Execution Time: " + formatDuration(result.executionTime()));
            if (result.errorMessage() != null) {
                System.out.println("Error: " + result.errorMessage());
            }
            if (result.mlirOutput() != null) {
                System.out.println("MLIR Output: " + result.mlirOutput());
            }
            if (!result.outputs().isEmpty()) {
                System.out.println("Output Tensors: " + result.outputs().size());
                for (var tensor : result.outputs()) {
                    System.out.println("  - " + tensor);
                }
            }
        }
    }

    private static String formatDuration(Duration duration) {
        if (duration == null) return "-";
        long seconds = duration.getSeconds();
        if (seconds < 60) {
            return seconds + "s";
        } else if (seconds < 3600) {
            return (seconds / 60) + "m " + (seconds % 60) + "s";
        } else {
            return (seconds / 3600) + "h " + ((seconds % 3600) / 60) + "m";
        }
    }

    private static boolean hasFlag(String[] args, String flag) {
        return Arrays.asList(args).contains(flag);
    }

    private static String getFlagValue(String[] args, String flag) {
        List<String> argList = Arrays.asList(args);
        int index = argList.indexOf(flag);
        if (index >= 0 && index < args.length - 1) {
            return args[index + 1];
        }
        return null;
    }

    // ===== Help output =====

    private static void printHelp() {
        System.out.println("WarpForge Launch CLI - Distributed job submission tool");
        System.out.println();
        System.out.println("Usage: warpforge-launch <command> [options]");
        System.out.println();
        System.out.println("Commands:");
        System.out.println("  submit        Submit a job to the scheduler");
        System.out.println("  status        Get job status");
        System.out.println("  result        Get job result");
        System.out.println("  cancel        Cancel a job");
        System.out.println("  list          List jobs");
        System.out.println("  cluster-info  Show cluster topology");
        System.out.println("  gpu-info      Show GPU info and backend status");
        System.out.println("  parity-test   Run PyTorch vs WarpForge comparison");
        System.out.println("  config        Show/set configuration");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  -h, --help    Show help for a command");
        System.out.println("  -v, --version Show version");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  warpforge-launch submit --source model.py --class MyModel --inputs '[(1,8):f32]'");
        System.out.println("  warpforge-launch status job-12345");
        System.out.println("  warpforge-launch list --state RUNNING");
        System.out.println("  warpforge-launch cluster-info");
    }

    private static void printSubmitHelp() {
        System.out.println("Usage: warpforge-launch submit [options]");
        System.out.println();
        System.out.println("Submit a job to the scheduler.");
        System.out.println();
        System.out.println("Required:");
        System.out.println("  --source <path>       Path to Python model file");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  --class <name>        Model class name (default: auto-detect)");
        System.out.println("  --inputs <spec>       Input specifications, e.g., '[(1,3,224,224):f32]'");
        System.out.println("  --scheduler <name>    Scheduler to use (local, ray, kubernetes, slurm)");
        System.out.println("  --gpu <type>          GPU type: NVIDIA, AMD, ANY, NONE (default: ANY)");
        System.out.println("  --gpu-count <n>       Number of GPUs (default: 1)");
        System.out.println("  --timeout <minutes>   Job timeout in minutes (default: 30)");
        System.out.println("  --wait                Wait for job to complete");
        System.out.println("  --json                Output as JSON");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  warpforge-launch submit --source models/resnet.py --class ResNet18");
        System.out.println("  warpforge-launch submit --source mlp.py --inputs '[(1,8):f32]' --wait");
        System.out.println("  warpforge-launch submit --source model.py --gpu NVIDIA --gpu-count 2");
    }

    private static void printParityTestHelp() {
        System.out.println("Usage: warpforge-launch parity-test [options]");
        System.out.println();
        System.out.println("Run PyTorch vs WarpForge comparison.");
        System.out.println();
        System.out.println("Required:");
        System.out.println("  --source <path>       Path to Python model file");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  --class <name>        Model class name (default: auto-detect)");
        System.out.println("  --inputs <spec>       Input specifications");
        System.out.println("  --tolerance <value>   Comparison tolerance (default: 1e-5)");
        System.out.println("  --json                Output as JSON");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  warpforge-launch parity-test --source models/mlp.py");
        System.out.println("  warpforge-launch parity-test --source model.py --tolerance 1e-4");
    }

    // ===== Response records for JSON output =====

    record SubmitResponse(String jobId, String correlationId) {}
}
