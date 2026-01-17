package io.surfworks.warpforge.io.collective;

/**
 * Configuration for collective communication.
 *
 * @param worldSize Total number of participating ranks
 * @param rank This process's rank (0 to worldSize-1)
 * @param masterAddress Address of rank 0 for coordination
 * @param masterPort Port for coordination
 * @param useRdma Whether to use RDMA transport (vs TCP)
 * @param inPlace Whether to allow in-place operations
 * @param asyncProgress Whether to use async progress threads
 * @param numProgressThreads Number of async progress threads (if asyncProgress)
 */
public record CollectiveConfig(
        int worldSize,
        int rank,
        String masterAddress,
        int masterPort,
        boolean useRdma,
        boolean inPlace,
        boolean asyncProgress,
        int numProgressThreads
) {

    /**
     * Creates a default configuration for the specified world size and rank.
     */
    public static CollectiveConfig of(int worldSize, int rank) {
        return new CollectiveConfig(
                worldSize,
                rank,
                "localhost",
                29500,
                true,   // prefer RDMA
                true,   // allow in-place
                true,   // async progress
                1       // 1 progress thread
        );
    }

    /**
     * Creates a configuration with master address.
     */
    public static CollectiveConfig of(int worldSize, int rank, String masterAddress, int masterPort) {
        return new CollectiveConfig(
                worldSize,
                rank,
                masterAddress,
                masterPort,
                true,
                true,
                true,
                1
        );
    }

    /**
     * Returns a new config with RDMA enabled/disabled.
     */
    public CollectiveConfig withRdma(boolean useRdma) {
        return new CollectiveConfig(worldSize, rank, masterAddress, masterPort,
                useRdma, inPlace, asyncProgress, numProgressThreads);
    }

    /**
     * Returns a new config with in-place operations enabled/disabled.
     */
    public CollectiveConfig withInPlace(boolean inPlace) {
        return new CollectiveConfig(worldSize, rank, masterAddress, masterPort,
                useRdma, inPlace, asyncProgress, numProgressThreads);
    }

    /**
     * Returns a new config with async progress enabled/disabled.
     */
    public CollectiveConfig withAsyncProgress(boolean asyncProgress, int numThreads) {
        return new CollectiveConfig(worldSize, rank, masterAddress, masterPort,
                useRdma, inPlace, asyncProgress, numThreads);
    }

    /**
     * Builder for creating custom configurations.
     */
    public static Builder builder(int worldSize, int rank) {
        return new Builder(worldSize, rank);
    }

    public static class Builder {
        private final int worldSize;
        private final int rank;
        private String masterAddress = "localhost";
        private int masterPort = 29500;
        private boolean useRdma = true;
        private boolean inPlace = true;
        private boolean asyncProgress = true;
        private int numProgressThreads = 1;

        Builder(int worldSize, int rank) {
            this.worldSize = worldSize;
            this.rank = rank;
        }

        public Builder masterAddress(String address) { this.masterAddress = address; return this; }
        public Builder masterPort(int port) { this.masterPort = port; return this; }
        public Builder useRdma(boolean use) { this.useRdma = use; return this; }
        public Builder inPlace(boolean allow) { this.inPlace = allow; return this; }
        public Builder asyncProgress(boolean enable) { this.asyncProgress = enable; return this; }
        public Builder numProgressThreads(int n) { this.numProgressThreads = n; return this; }

        public CollectiveConfig build() {
            return new CollectiveConfig(worldSize, rank, masterAddress, masterPort,
                    useRdma, inPlace, asyncProgress, numProgressThreads);
        }
    }
}
