package io.surfworks.warpforge.launch.scheduler.slurm;

import io.surfworks.warpforge.launch.job.GpuType;
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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Slurm scheduler using SSH + sbatch/squeue commands.
 *
 * <p>Follows patterns from the existing Holmes lab CI scripts.
 */
public final class SlurmScheduler implements Scheduler {

    private static final Pattern JOB_ID_PATTERN = Pattern.compile("Submitted batch job (\\d+)");

    private final SlurmConfig config;
    private volatile boolean closed;

    /**
     * Creates a Slurm scheduler with the given configuration.
     */
    public SlurmScheduler(SlurmConfig config) {
        this.config = config;
        this.closed = false;
    }

    @Override
    public String name() {
        return "slurm";
    }

    @Override
    public SchedulerCapabilities capabilities() {
        return SchedulerCapabilities.fullFeatured(Set.of(GpuType.NVIDIA, GpuType.AMD));
    }

    @Override
    public String submit(JobSubmission submission) throws SchedulerException {
        ensureOpen();

        // Generate batch script
        String batchScript = generateBatchScript(submission);
        String scriptName = "warpforge-" + submission.shortCorrelationId() + ".slurm";
        String remotePath = config.remoteWorkDir() + "/" + scriptName;

        // Upload script via SSH
        sshExec("mkdir -p " + config.remoteWorkDir());
        sshExecWithInput("cat > " + remotePath, batchScript);

        // Submit with sbatch
        String output = sshExec("sbatch " + remotePath);

        // Parse job ID from output
        Matcher matcher = JOB_ID_PATTERN.matcher(output);
        if (matcher.find()) {
            return matcher.group(1);
        }
        throw new SchedulerException("Failed to parse Slurm job ID from: " + output);
    }

    @Override
    public JobStatus status(String jobId) throws SchedulerException {
        ensureOpen();

        // Try squeue first (for running/pending jobs)
        String output = sshExec("squeue -j " + jobId + " -o '%T|%M|%N' -h 2>/dev/null || true");

        if (!output.isBlank()) {
            return parseSqueueOutput(jobId, output);
        }

        // Job not in queue - check sacct for completed job
        output = sshExec("sacct -j " + jobId + " -o 'State,Elapsed,NodeList' -n -P 2>/dev/null || echo 'UNKNOWN||'");
        return parseSacctOutput(jobId, output);
    }

    @Override
    public JobResult result(String jobId) throws SchedulerException {
        JobStatus status = status(jobId);

        if (!status.state().isTerminal()) {
            throw new IllegalStateException(
                    "Job " + jobId + " is not complete (state: " + status.state() + ")");
        }

        if (status.state() == JobState.COMPLETED) {
            return JobResult.success(jobId, status.correlationId(), List.of(), status.elapsed());
        } else {
            return JobResult.failure(jobId, status.correlationId(), status.message(), status.elapsed());
        }
    }

    @Override
    public boolean cancel(String jobId) throws SchedulerException {
        ensureOpen();

        try {
            sshExec("scancel " + jobId);
            return true;
        } catch (SchedulerException e) {
            return false;
        }
    }

    @Override
    public List<JobStatus> list(JobQuery query) throws SchedulerException {
        ensureOpen();

        // List all jobs with squeue
        String output = sshExec("squeue -u " + config.sshUser() + " -o '%j|%i|%T|%M|%N' -h");

        List<JobStatus> statuses = new ArrayList<>();
        for (String line : output.split("\n")) {
            if (line.isBlank()) continue;

            String[] parts = line.split("\\|");
            if (parts.length >= 5) {
                String jobName = parts[0];
                String jobId = parts[1];
                String stateStr = parts[2];
                String elapsed = parts[3];
                String nodeName = parts[4];

                JobState state = mapSlurmState(stateStr);
                if (!query.states().isEmpty() && !query.states().contains(state)) {
                    continue;
                }

                statuses.add(new JobStatus(
                        jobId, null, state, Instant.now(),
                        nodeName.isBlank() ? null : nodeName,
                        parseElapsed(elapsed),
                        stateStr, Map.of("slurm_state", stateStr)
                ));
            }
        }

        if (query.limit() > 0 && statuses.size() > query.limit()) {
            return statuses.subList(0, query.limit());
        }
        return statuses;
    }

    @Override
    public boolean isConnected() {
        if (closed) return false;

        try {
            String output = sshExec("echo ok");
            return "ok".equals(output.trim());
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public ClusterInfo clusterInfo() throws SchedulerException {
        ensureOpen();

        // Get node info with sinfo
        String output = sshExec("sinfo -N -o '%N|%T|%c|%m|%G' -h");

        List<NodeInfo> nodes = new ArrayList<>();
        int nvidiaCount = 0;
        int amdCount = 0;

        for (String line : output.split("\n")) {
            if (line.isBlank()) continue;

            String[] parts = line.split("\\|");
            if (parts.length >= 5) {
                String nodeName = parts[0].trim();
                String status = parts[1].trim();
                int cpuCores = parseInt(parts[2].trim(), 1);
                long memoryMb = parseLong(parts[3].trim(), 1024);
                String gres = parts[4].trim();

                GpuType gpuType = GpuType.NONE;
                int gpuCount = 0;

                if (gres.contains("gpu:nvidia")) {
                    gpuType = GpuType.NVIDIA;
                    gpuCount = parseGpuCount(gres);
                    nvidiaCount += gpuCount;
                } else if (gres.contains("gpu:amd")) {
                    gpuType = GpuType.AMD;
                    gpuCount = parseGpuCount(gres);
                    amdCount += gpuCount;
                }

                String readyStatus = status.contains("idle") || status.contains("mix") ? "ready" : status;
                nodes.add(new NodeInfo(nodeName, readyStatus, gpuType, gpuCount, memoryMb, cpuCores));
            }
        }

        int availableNodes = (int) nodes.stream().filter(NodeInfo::isReady).count();

        return new ClusterInfo(
                "slurm",
                getSlurmVersion(),
                nodes.size(),
                availableNodes,
                Map.of(GpuType.NVIDIA, nvidiaCount, GpuType.AMD, amdCount),
                nodes
        );
    }

    @Override
    public void close() {
        closed = true;
    }

    private String generateBatchScript(JobSubmission submission) {
        var def = submission.definition();
        var res = def.resources();

        StringBuilder sb = new StringBuilder();
        sb.append("#!/bin/bash\n");
        sb.append("#SBATCH --job-name=warpforge-").append(submission.shortCorrelationId()).append("\n");
        sb.append("#SBATCH --output=").append(config.remoteWorkDir()).append("/warpforge-%j.out\n");
        sb.append("#SBATCH --error=").append(config.remoteWorkDir()).append("/warpforge-%j.err\n");

        if (config.partition() != null && !config.partition().isBlank()) {
            sb.append("#SBATCH --partition=").append(config.partition()).append("\n");
        }

        sb.append("#SBATCH --cpus-per-task=").append(res.cpuCores()).append("\n");
        sb.append("#SBATCH --mem=").append(res.memoryMb()).append("M\n");

        if (res.requiresGpu()) {
            String gpuSpec = switch (res.gpuType()) {
                case NVIDIA -> "gpu:nvidia:" + res.gpuCount();
                case AMD -> "gpu:amd:" + res.gpuCount();
                default -> "gpu:" + res.gpuCount();
            };
            sb.append("#SBATCH --gres=").append(gpuSpec).append("\n");
        }

        if (!res.nodeAffinity().isEmpty()) {
            sb.append("#SBATCH --nodelist=").append(String.join(",", res.nodeAffinity())).append("\n");
        }

        if (def.timeout() != null) {
            long minutes = def.timeout().toMinutes();
            sb.append("#SBATCH --time=").append(minutes).append("\n");
        }

        sb.append("\n");
        sb.append("# WarpForge Pipeline Execution\n");
        sb.append("cd ").append(config.remoteWorkDir()).append("\n");
        sb.append("\n");
        sb.append("snakegrinder --trace-with-values \\\n");
        sb.append("  --source ").append(def.modelSource()).append(" \\\n");
        sb.append("  --class ").append(def.modelClass()).append(" \\\n");
        sb.append("  --inputs '").append(def.formatInputSpecs()).append("' \\\n");
        sb.append("  --seed ").append(def.seed()).append(" \\\n");
        sb.append("  --out ").append(config.remoteWorkDir()).append("/output-$SLURM_JOB_ID\n");

        return sb.toString();
    }

    private JobStatus parseSqueueOutput(String jobId, String output) {
        String[] parts = output.trim().split("\\|");
        String stateStr = parts.length > 0 ? parts[0] : "UNKNOWN";
        String elapsed = parts.length > 1 ? parts[1] : "0:00";
        String nodeName = parts.length > 2 ? parts[2] : null;

        return new JobStatus(
                jobId, null,
                mapSlurmState(stateStr),
                Instant.now(),
                nodeName != null && !nodeName.isBlank() ? nodeName : null,
                parseElapsed(elapsed),
                stateStr,
                Map.of("slurm_state", stateStr)
        );
    }

    private JobStatus parseSacctOutput(String jobId, String output) {
        String[] parts = output.trim().split("\\|");
        String stateStr = parts.length > 0 ? parts[0] : "UNKNOWN";
        String elapsed = parts.length > 1 ? parts[1] : "0:00";
        String nodeName = parts.length > 2 ? parts[2] : null;

        return new JobStatus(
                jobId, null,
                mapSlurmState(stateStr),
                Instant.now(),
                nodeName != null && !nodeName.isBlank() ? nodeName : null,
                parseElapsed(elapsed),
                stateStr,
                Map.of("slurm_state", stateStr)
        );
    }

    private JobState mapSlurmState(String slurmState) {
        return switch (slurmState.toUpperCase()) {
            case "PENDING", "CONFIGURING", "REQUEUED" -> JobState.PENDING;
            case "RUNNING", "COMPLETING", "STAGE_OUT" -> JobState.RUNNING;
            case "COMPLETED" -> JobState.COMPLETED;
            case "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "DEADLINE" -> JobState.FAILED;
            case "CANCELLED", "PREEMPTED", "SUSPENDED" -> JobState.CANCELLED;
            case "TIMEOUT" -> JobState.TIMEOUT;
            default -> JobState.PENDING;
        };
    }

    private Duration parseElapsed(String elapsed) {
        // Parse formats like "0:05", "1:23:45", "1-12:34:56"
        try {
            if (elapsed.contains("-")) {
                String[] dayTime = elapsed.split("-");
                int days = Integer.parseInt(dayTime[0]);
                Duration dayDuration = Duration.ofDays(days);
                return dayDuration.plus(parseHMS(dayTime[1]));
            }
            return parseHMS(elapsed);
        } catch (Exception e) {
            return Duration.ZERO;
        }
    }

    private Duration parseHMS(String hms) {
        String[] parts = hms.split(":");
        if (parts.length == 2) {
            return Duration.ofMinutes(Long.parseLong(parts[0]))
                    .plusSeconds(Long.parseLong(parts[1]));
        } else if (parts.length == 3) {
            return Duration.ofHours(Long.parseLong(parts[0]))
                    .plusMinutes(Long.parseLong(parts[1]))
                    .plusSeconds(Long.parseLong(parts[2]));
        }
        return Duration.ZERO;
    }

    private String getSlurmVersion() {
        try {
            String output = sshExec("sinfo --version 2>/dev/null || echo 'unknown'");
            return output.trim().replace("slurm ", "");
        } catch (Exception e) {
            return "unknown";
        }
    }

    private int parseGpuCount(String gres) {
        // Parse GRES format like "gpu:nvidia:2" or "gpu:2"
        Pattern p = Pattern.compile("gpu(?::[^:]+)?:(\\d+)");
        Matcher m = p.matcher(gres);
        if (m.find()) {
            return Integer.parseInt(m.group(1));
        }
        return 0;
    }

    private int parseInt(String s, int defaultValue) {
        try {
            return Integer.parseInt(s);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    private long parseLong(String s, long defaultValue) {
        try {
            return Long.parseLong(s);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    private String sshExec(String command) throws SchedulerException {
        return sshExecWithInput(command, null);
    }

    private String sshExecWithInput(String command, String stdin) throws SchedulerException {
        try {
            List<String> cmd = new ArrayList<>();
            cmd.add("ssh");
            cmd.add("-o");
            cmd.add("ConnectTimeout=" + config.sshConnectTimeoutSeconds());
            cmd.add("-o");
            cmd.add("BatchMode=yes");
            if (config.sshKeyPath() != null) {
                cmd.add("-i");
                cmd.add(config.sshKeyPath().toString());
            }
            cmd.add(config.sshTarget());
            cmd.add(command);

            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.redirectErrorStream(true);
            Process p = pb.start();

            if (stdin != null) {
                p.getOutputStream().write(stdin.getBytes());
                p.getOutputStream().close();
            }

            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                }
            }

            int exitCode = p.waitFor();
            if (exitCode != 0) {
                throw new SchedulerException("SSH command failed (exit " + exitCode + "): " + output);
            }

            return output.toString().trim();

        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new SchedulerException("SSH execution failed", e);
        }
    }

    private void ensureOpen() throws SchedulerException {
        if (closed) {
            throw new SchedulerException("Scheduler is closed");
        }
    }
}
