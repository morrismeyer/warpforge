package io.surfworks.warpforge.io.integration;

import io.surfworks.warpforge.io.collective.Collective;
import io.surfworks.warpforge.io.collective.CollectiveApi;
import io.surfworks.warpforge.io.collective.CollectiveConfig;
import io.surfworks.warpforge.io.rdma.Rdma;
import io.surfworks.warpforge.io.rdma.RdmaApi;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Base class for Ray-launched integration tests.
 *
 * <p>These tests are designed to run on the Mark 1 lab hardware with
 * Mellanox ConnectX-5 100GbE NICs. They are launched via Ray to coordinate
 * multiple processes across nodes.
 *
 * <h2>Running Integration Tests</h2>
 * <pre>{@code
 * # On NUC (Ray head node)
 * ./gradlew :warpforge-io:rayIntegrationTest \
 *     -Dray.address=auto \
 *     -Drdma.world.size=2 \
 *     -Drdma.server.host=gpu-node-2
 * }</pre>
 *
 * <h2>Environment</h2>
 * <ul>
 *   <li>{@code RDMA_WORLD_SIZE} - Number of participating nodes (default: 2)</li>
 *   <li>{@code RDMA_RANK} - This process's rank (set by Ray)</li>
 *   <li>{@code RDMA_MASTER_ADDR} - Address of rank 0 (default: gpu-node-1)</li>
 *   <li>{@code RDMA_MASTER_PORT} - Port for coordination (default: 29500)</li>
 * </ul>
 */
@Tag("ray-integration")
public abstract class RayIntegrationTestBase {

    protected RdmaApi rdma;
    protected CollectiveApi collective;

    protected int worldSize;
    protected int rank;
    protected String masterAddress;
    protected int masterPort;

    @BeforeEach
    void setUpRay() {
        // Read configuration from system properties or environment
        worldSize = Integer.getInteger("rdma.world.size",
                Integer.parseInt(System.getenv().getOrDefault("RDMA_WORLD_SIZE", "2")));
        rank = Integer.getInteger("rdma.rank",
                Integer.parseInt(System.getenv().getOrDefault("RDMA_RANK", "0")));
        masterAddress = System.getProperty("rdma.master.addr",
                System.getenv().getOrDefault("RDMA_MASTER_ADDR", "gpu-node-1"));
        masterPort = Integer.getInteger("rdma.master.port",
                Integer.parseInt(System.getenv().getOrDefault("RDMA_MASTER_PORT", "29500")));

        // Skip if no RDMA hardware
        assumeTrue(Rdma.canUseRealRdma(),
                "Skipping: RDMA hardware not available");

        // Initialize RDMA
        rdma = Rdma.load();
        assumeTrue(!"mock".equals(rdma.backendName()),
                "Skipping: Using mock RDMA (real hardware required)");

        // Initialize collective
        CollectiveConfig config = CollectiveConfig.builder(worldSize, rank)
                .masterAddress(masterAddress)
                .masterPort(masterPort)
                .useRdma(true)
                .build();

        collective = Collective.load(config);

        System.out.printf("[Rank %d/%d] Integration test initialized with %s backend%n",
                rank, worldSize, rdma.backendName());
    }

    @AfterEach
    void tearDownRay() {
        if (collective != null) {
            try {
                // Barrier before cleanup to ensure all ranks are done
                collective.barrier().join();
            } catch (Exception e) {
                // Ignore errors during cleanup
            }
            collective.close();
        }
        if (rdma != null) {
            rdma.close();
        }
    }

    /**
     * Synchronizes all ranks before a test phase.
     */
    protected void sync() {
        try {
            collective.barrier().join();
        } catch (Exception e) {
            throw new RuntimeException("Barrier failed", e);
        }
    }

    /**
     * Logs a message with rank prefix.
     */
    protected void log(String format, Object... args) {
        System.out.printf("[Rank %d] " + format + "%n", rank, args);
    }

    /**
     * Returns true if this is rank 0 (master).
     */
    protected boolean isMaster() {
        return rank == 0;
    }
}
