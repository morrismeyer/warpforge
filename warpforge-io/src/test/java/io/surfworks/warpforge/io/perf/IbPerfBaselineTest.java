package io.surfworks.warpforge.io.perf;

import org.junit.jupiter.api.*;

import java.io.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Baseline performance tests using native ibperf tools.
 *
 * <p>These tests establish the theoretical maximum throughput of the
 * Mellanox 100GbE hardware. The Java implementation should achieve
 * 95-98% of these baseline numbers.
 *
 * <h2>Prerequisites</h2>
 * <ul>
 *   <li>perftest package installed (ib_write_bw, ib_read_bw, etc.)</li>
 *   <li>RDMA-capable NIC configured and connected</li>
 *   <li>Server process running on remote node</li>
 * </ul>
 *
 * <h2>Running</h2>
 * <pre>{@code
 * # On remote node (server):
 * ib_write_bw
 *
 * # Run test:
 * ./gradlew :warpforge-io:rdmaPerfTest \
 *     --tests "*IbPerfBaselineTest*" \
 *     -Drdma.server.host=gpu-node-2
 * }</pre>
 */
@Tag("rdma-perf")
@Tag("rdma")
@DisplayName("ibperf Baseline Tests")
class IbPerfBaselineTest {

    private String serverHost;

    // Results stored for comparison with Java tests
    static double baselineWriteBandwidthGbps;
    static double baselineReadBandwidthGbps;
    static double baselineSendBandwidthGbps;
    static double baselineWriteLatencyUs;

    @BeforeEach
    void setUp() {
        serverHost = System.getProperty("rdma.server.host",
                System.getenv().getOrDefault("RDMA_SERVER_HOST", "gpu-node-2"));

        // Check if perftest is installed
        assumeTrue(isPerftestInstalled(),
                "Skipping: perftest package not installed");
    }

    private boolean isPerftestInstalled() {
        try {
            Process p = new ProcessBuilder("which", "ib_write_bw").start();
            return p.waitFor() == 0;
        } catch (Exception e) {
            return false;
        }
    }

    @Test
    @DisplayName("Baseline: RDMA Write Bandwidth (ib_write_bw)")
    void measureWriteBandwidth() throws Exception {
        String[] cmd = {
                "ib_write_bw",
                "--size=1048576",      // 1MB messages
                "--iters=5000",
                "--report_gbits",
                serverHost
        };

        PerfResult result = runPerftest(cmd);

        assertNotNull(result, "Failed to parse ib_write_bw output");
        assertTrue(result.bandwidthGbps > 0, "Bandwidth should be positive");

        baselineWriteBandwidthGbps = result.bandwidthGbps;

        System.out.printf("ib_write_bw baseline: %.2f Gbps (%.2f MB/s)%n",
                result.bandwidthGbps, result.bandwidthMBps);

        // Mellanox ConnectX-5 should achieve close to 100 Gbps
        assertTrue(result.bandwidthGbps >= 90.0,
                "Write bandwidth should be at least 90 Gbps on 100GbE, got " + result.bandwidthGbps);
    }

    @Test
    @DisplayName("Baseline: RDMA Read Bandwidth (ib_read_bw)")
    void measureReadBandwidth() throws Exception {
        String[] cmd = {
                "ib_read_bw",
                "--size=1048576",
                "--iters=5000",
                "--report_gbits",
                serverHost
        };

        PerfResult result = runPerftest(cmd);

        assertNotNull(result, "Failed to parse ib_read_bw output");
        assertTrue(result.bandwidthGbps > 0, "Bandwidth should be positive");

        baselineReadBandwidthGbps = result.bandwidthGbps;

        System.out.printf("ib_read_bw baseline: %.2f Gbps (%.2f MB/s)%n",
                result.bandwidthGbps, result.bandwidthMBps);

        assertTrue(result.bandwidthGbps >= 90.0,
                "Read bandwidth should be at least 90 Gbps on 100GbE, got " + result.bandwidthGbps);
    }

    @Test
    @DisplayName("Baseline: Send Bandwidth (ib_send_bw)")
    void measureSendBandwidth() throws Exception {
        String[] cmd = {
                "ib_send_bw",
                "--size=1048576",
                "--iters=5000",
                "--report_gbits",
                serverHost
        };

        PerfResult result = runPerftest(cmd);

        assertNotNull(result, "Failed to parse ib_send_bw output");
        assertTrue(result.bandwidthGbps > 0, "Bandwidth should be positive");

        baselineSendBandwidthGbps = result.bandwidthGbps;

        System.out.printf("ib_send_bw baseline: %.2f Gbps (%.2f MB/s)%n",
                result.bandwidthGbps, result.bandwidthMBps);

        // Send might be slightly lower due to two-sided overhead
        assertTrue(result.bandwidthGbps >= 80.0,
                "Send bandwidth should be at least 80 Gbps on 100GbE, got " + result.bandwidthGbps);
    }

    @Test
    @DisplayName("Baseline: Write Latency (ib_write_lat)")
    void measureWriteLatency() throws Exception {
        String[] cmd = {
                "ib_write_lat",
                "--size=8",            // Small message for latency
                "--iters=10000",
                serverHost
        };

        LatencyResult result = runLatencyTest(cmd);

        assertNotNull(result, "Failed to parse ib_write_lat output");
        assertTrue(result.avgLatencyUs > 0, "Latency should be positive");

        baselineWriteLatencyUs = result.avgLatencyUs;

        System.out.printf("ib_write_lat baseline: avg=%.2f us, p99=%.2f us%n",
                result.avgLatencyUs, result.p99LatencyUs);

        // ConnectX-5 should achieve < 2us latency
        assertTrue(result.avgLatencyUs < 5.0,
                "Write latency should be < 5us on 100GbE, got " + result.avgLatencyUs);
    }

    @Test
    @DisplayName("Baseline: Send Latency (ib_send_lat)")
    void measureSendLatency() throws Exception {
        String[] cmd = {
                "ib_send_lat",
                "--size=8",
                "--iters=10000",
                serverHost
        };

        LatencyResult result = runLatencyTest(cmd);

        assertNotNull(result, "Failed to parse ib_send_lat output");
        assertTrue(result.avgLatencyUs > 0, "Latency should be positive");

        System.out.printf("ib_send_lat baseline: avg=%.2f us, p99=%.2f us%n",
                result.avgLatencyUs, result.p99LatencyUs);

        assertTrue(result.avgLatencyUs < 10.0,
                "Send latency should be < 10us on 100GbE, got " + result.avgLatencyUs);
    }

    @Test
    @DisplayName("Baseline: Atomic Operations (ib_atomic_bw)")
    void measureAtomicBandwidth() throws Exception {
        String[] cmd = {
                "ib_atomic_bw",
                "--iters=100000",
                "--report_gbits",
                serverHost
        };

        PerfResult result = runPerftest(cmd);

        if (result != null) {
            System.out.printf("ib_atomic_bw baseline: %.2f M ops/s%n",
                    result.bandwidthMBps / 8); // 8 bytes per atomic op
        }
    }

    // ===== Helper Methods =====

    private PerfResult runPerftest(String[] cmd) throws Exception {
        System.out.println("Running: " + String.join(" ", cmd));

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        Process p = pb.start();

        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println("[perftest] " + line);
                output.append(line).append("\n");
            }
        }

        int exitCode = p.waitFor();
        if (exitCode != 0) {
            System.err.println("perftest failed with exit code " + exitCode);
            return null;
        }

        return parseBandwidthOutput(output.toString());
    }

    private LatencyResult runLatencyTest(String[] cmd) throws Exception {
        System.out.println("Running: " + String.join(" ", cmd));

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        Process p = pb.start();

        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println("[perftest] " + line);
                output.append(line).append("\n");
            }
        }

        int exitCode = p.waitFor();
        if (exitCode != 0) {
            return null;
        }

        return parseLatencyOutput(output.toString());
    }

    private PerfResult parseBandwidthOutput(String output) {
        // Parse output like:
        // #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
        // 1048576    5000           98.45              97.23                0.011594
        Pattern pattern = Pattern.compile(
                "\\d+\\s+\\d+\\s+([\\d.]+)\\s+([\\d.]+)\\s+[\\d.]+");
        Matcher matcher = pattern.matcher(output);

        if (matcher.find()) {
            double peakGbps = Double.parseDouble(matcher.group(1));
            double avgGbps = Double.parseDouble(matcher.group(2));
            return new PerfResult(avgGbps, avgGbps * 1000 / 8); // Gbps to MB/s
        }

        return null;
    }

    private LatencyResult parseLatencyOutput(String output) {
        // Parse output like:
        // #bytes #iterations    t_min[usec]    t_max[usec]  t_typical[usec]    t_avg[usec]    t_stdev[usec]   99% percentile[usec]
        // 8       10000          1.23           5.67         1.34               1.45           0.12            2.34
        Pattern pattern = Pattern.compile(
                "\\d+\\s+\\d+\\s+([\\d.]+)\\s+([\\d.]+)\\s+([\\d.]+)\\s+([\\d.]+)\\s+[\\d.]+\\s+([\\d.]+)");
        Matcher matcher = pattern.matcher(output);

        if (matcher.find()) {
            double minUs = Double.parseDouble(matcher.group(1));
            double maxUs = Double.parseDouble(matcher.group(2));
            double typicalUs = Double.parseDouble(matcher.group(3));
            double avgUs = Double.parseDouble(matcher.group(4));
            double p99Us = Double.parseDouble(matcher.group(5));
            return new LatencyResult(minUs, maxUs, avgUs, p99Us);
        }

        return null;
    }

    record PerfResult(double bandwidthGbps, double bandwidthMBps) {}

    record LatencyResult(double minLatencyUs, double maxLatencyUs,
                         double avgLatencyUs, double p99LatencyUs) {}
}
