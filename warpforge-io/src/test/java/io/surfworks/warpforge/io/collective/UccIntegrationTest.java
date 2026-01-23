package io.surfworks.warpforge.io.collective;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Integration tests for real UCC collective operations.
 *
 * <p>These tests require:
 * <ul>
 *   <li>Linux operating system</li>
 *   <li>UCC libraries installed (libucc.so)</li>
 *   <li>UCX libraries installed (libucp.so)</li>
 * </ul>
 *
 * <p>Run with: ./gradlew :warpforge-io:test --tests "*UccIntegrationTest*"
 */
@Tag("ucc")
@EnabledOnOs(OS.LINUX)
class UccIntegrationTest {

    @Test
    void testUccLibraryDetection() {
        System.out.println("=== UCC Library Detection ===");
        System.out.println(Collective.systemInfo());

        boolean hasLibs = Collective.hasUccLibraries();
        System.out.println("Has UCC libraries: " + hasLibs);

        if (!hasLibs) {
            System.out.println("UCC libraries not found - skipping integration tests");
            System.out.println("Install UCC with: ./gradlew :openucx-runtime:ensureOpenUCXReady");
        }

        // This test always passes - it's informational
        assertTrue(true);
    }

    @Test
    void testUccInitializationSingleRank() {
        if (!Collective.canUseRealUcc()) {
            System.out.println("Skipping: UCC not available on this system");
            return;
        }

        System.out.println("=== Testing Single-Rank UCC Initialization ===");

        // Single-rank configuration doesn't need network coordination
        CollectiveConfig config = CollectiveConfig.of(1, 0, "localhost", 29500);

        try (CollectiveApi collective = Collective.loadUcc(config)) {
            assertNotNull(collective);
            assertEquals("ucc", collective.backendName());
            assertEquals(1, collective.worldSize());
            assertEquals(0, collective.rank());

            System.out.println("UCC initialized successfully!");
            System.out.println("Backend: " + collective.backendName());
            System.out.println("World size: " + collective.worldSize());
            System.out.println("Rank: " + collective.rank());
        } catch (CollectiveException e) {
            System.err.println("UCC initialization failed: " + e.getMessage());
            e.printStackTrace();
            fail("UCC initialization should succeed with single rank: " + e.getMessage());
        }
    }

    @Test
    void testBarrierSingleRank() {
        if (!Collective.canUseRealUcc()) {
            System.out.println("Skipping: UCC not available on this system");
            return;
        }

        System.out.println("=== Testing Single-Rank Barrier ===");

        CollectiveConfig config = CollectiveConfig.of(1, 0, "localhost", 29501);

        try (CollectiveApi collective = Collective.loadUcc(config)) {
            // Barrier with single rank should complete immediately
            collective.barrier().join();
            System.out.println("Barrier completed successfully!");

            // Check stats
            CollectiveApi.CollectiveStats stats = collective.stats();
            assertEquals(1, stats.barrierCount());
            System.out.println("Stats: " + stats);
        } catch (Exception e) {
            System.err.println("Barrier test failed: " + e.getMessage());
            e.printStackTrace();
            fail("Single-rank barrier should succeed: " + e.getMessage());
        }
    }

    @Test
    void testAllReduceSingleRank() {
        if (!Collective.canUseRealUcc()) {
            System.out.println("Skipping: UCC not available on this system");
            return;
        }

        System.out.println("=== Testing Single-Rank AllReduce ===");

        CollectiveConfig config = CollectiveConfig.of(1, 0, "localhost", 29502);

        try (CollectiveApi collective = Collective.loadUcc(config)) {
            // Create a test tensor with known values
            var input = io.surfworks.warpforge.core.tensor.Tensor.fromFloatArray(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f}, 4
            );

            // AllReduce with single rank should return the same values
            var result = collective.allReduce(input, AllReduceOp.SUM).join();

            assertNotNull(result);
            assertEquals(input.shape()[0], result.shape()[0]);

            // Verify values (should be unchanged with single rank)
            for (int i = 0; i < 4; i++) {
                float expected = i + 1.0f;
                float actual = result.getFloatFlat(i);
                assertEquals(expected, actual, 0.001f, "Element " + i + " mismatch");
            }

            System.out.println("AllReduce completed successfully!");
            System.out.println("Stats: " + collective.stats());
        } catch (Exception e) {
            System.err.println("AllReduce test failed: " + e.getMessage());
            e.printStackTrace();
            fail("Single-rank allreduce should succeed: " + e.getMessage());
        }
    }

    @Test
    void testBroadcastSingleRank() {
        if (!Collective.canUseRealUcc()) {
            System.out.println("Skipping: UCC not available on this system");
            return;
        }

        System.out.println("=== Testing Single-Rank Broadcast ===");

        CollectiveConfig config = CollectiveConfig.of(1, 0, "localhost", 29503);

        try (CollectiveApi collective = Collective.loadUcc(config)) {
            var input = io.surfworks.warpforge.core.tensor.Tensor.fromFloatArray(
                new float[]{10.0f, 20.0f, 30.0f, 40.0f}, 4
            );

            // Broadcast from rank 0 (the only rank)
            var result = collective.broadcast(input, 0).join();

            assertNotNull(result);
            assertEquals(4, result.shape()[0]);

            // Verify values (should match input)
            for (int i = 0; i < 4; i++) {
                float expected = (i + 1) * 10.0f;
                float actual = result.getFloatFlat(i);
                assertEquals(expected, actual, 0.001f, "Element " + i + " mismatch");
            }

            System.out.println("Broadcast completed successfully!");
            System.out.println("Stats: " + collective.stats());
        } catch (Exception e) {
            System.err.println("Broadcast test failed: " + e.getMessage());
            e.printStackTrace();
            fail("Single-rank broadcast should succeed: " + e.getMessage());
        }
    }

    @Test
    void testAllGatherSingleRank() {
        if (!Collective.canUseRealUcc()) {
            System.out.println("Skipping: UCC not available on this system");
            return;
        }

        System.out.println("=== Testing Single-Rank AllGather ===");

        CollectiveConfig config = CollectiveConfig.of(1, 0, "localhost", 29504);

        try (CollectiveApi collective = Collective.loadUcc(config)) {
            var input = io.surfworks.warpforge.core.tensor.Tensor.fromFloatArray(
                new float[]{5.0f, 6.0f, 7.0f, 8.0f}, 4
            );

            // AllGather with single rank should return same tensor
            var result = collective.allGather(input).join();

            assertNotNull(result);
            // With worldSize=1, output should be same size as input
            assertEquals(4, result.shape()[0]);

            System.out.println("AllGather completed successfully!");
            System.out.println("Stats: " + collective.stats());
        } catch (Exception e) {
            System.err.println("AllGather test failed: " + e.getMessage());
            e.printStackTrace();
            fail("Single-rank allgather should succeed: " + e.getMessage());
        }
    }
}
