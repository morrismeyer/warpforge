package io.surfworks.warpforge.io.rdma;

/**
 * Configuration for RDMA connections.
 *
 * @param deviceName Preferred device name (e.g., "mlx5_0"), or null for auto-detect
 * @param port Device port number (typically 1)
 * @param gidIndex GID index for RoCE (typically 0 for IB, 3 for RoCEv2)
 * @param maxSendWorkRequests Maximum send work requests in send queue
 * @param maxRecvWorkRequests Maximum receive work requests in receive queue
 * @param maxSendSge Maximum scatter-gather elements per send
 * @param maxRecvSge Maximum scatter-gather elements per receive
 * @param maxInlineData Maximum inline data size (0 to disable)
 * @param serviceLevel InfiniBand service level (0-15)
 * @param trafficClass Traffic class for RoCE
 * @param retryCount Retry count for RC connections (0-7)
 * @param rnrRetryCount RNR retry count (0-7, 7 means infinite)
 * @param minRnrTimer Minimum RNR NAK timer
 * @param timeout Local ack timeout (0-31, actual timeout = 4.096us * 2^timeout)
 */
public record RdmaConfig(
        String deviceName,
        int port,
        int gidIndex,
        int maxSendWorkRequests,
        int maxRecvWorkRequests,
        int maxSendSge,
        int maxRecvSge,
        int maxInlineData,
        int serviceLevel,
        int trafficClass,
        int retryCount,
        int rnrRetryCount,
        int minRnrTimer,
        int timeout
) {

    /**
     * Default configuration suitable for high-performance transfers.
     */
    public static final RdmaConfig DEFAULT = new RdmaConfig(
            null,   // auto-detect device
            1,      // port 1
            0,      // GID index 0 (will be 3 for RoCEv2)
            1024,   // send queue depth
            1024,   // receive queue depth
            4,      // SGE per send
            4,      // SGE per receive
            256,    // inline data
            0,      // service level
            0,      // traffic class
            7,      // retry count
            7,      // RNR retry (infinite)
            12,     // min RNR timer
            14      // timeout (~67ms)
    );

    /**
     * Configuration optimized for latency (smaller queues, more inline).
     */
    public static final RdmaConfig LOW_LATENCY = new RdmaConfig(
            null, 1, 0,
            256,    // smaller send queue
            256,    // smaller receive queue
            1, 1,
            512,    // more inline data
            0, 0, 7, 7, 1, 10
    );

    /**
     * Configuration optimized for throughput (larger queues).
     */
    public static final RdmaConfig HIGH_THROUGHPUT = new RdmaConfig(
            null, 1, 0,
            4096,   // large send queue
            4096,   // large receive queue
            8, 8,
            0,      // no inline (all DMA)
            0, 0, 7, 7, 12, 14
    );

    /**
     * Returns a new config with the specified device name.
     */
    public RdmaConfig withDeviceName(String deviceName) {
        return new RdmaConfig(deviceName, port, gidIndex, maxSendWorkRequests, maxRecvWorkRequests,
                maxSendSge, maxRecvSge, maxInlineData, serviceLevel, trafficClass,
                retryCount, rnrRetryCount, minRnrTimer, timeout);
    }

    /**
     * Returns a new config with the specified port.
     */
    public RdmaConfig withPort(int port) {
        return new RdmaConfig(deviceName, port, gidIndex, maxSendWorkRequests, maxRecvWorkRequests,
                maxSendSge, maxRecvSge, maxInlineData, serviceLevel, trafficClass,
                retryCount, rnrRetryCount, minRnrTimer, timeout);
    }

    /**
     * Returns a new config with the specified GID index.
     */
    public RdmaConfig withGidIndex(int gidIndex) {
        return new RdmaConfig(deviceName, port, gidIndex, maxSendWorkRequests, maxRecvWorkRequests,
                maxSendSge, maxRecvSge, maxInlineData, serviceLevel, trafficClass,
                retryCount, rnrRetryCount, minRnrTimer, timeout);
    }

    /**
     * Returns a new config with the specified queue depths.
     */
    public RdmaConfig withQueueDepth(int sendDepth, int recvDepth) {
        return new RdmaConfig(deviceName, port, gidIndex, sendDepth, recvDepth,
                maxSendSge, maxRecvSge, maxInlineData, serviceLevel, trafficClass,
                retryCount, rnrRetryCount, minRnrTimer, timeout);
    }

    /**
     * Builder for creating custom configurations.
     */
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String deviceName = null;
        private int port = 1;
        private int gidIndex = 0;
        private int maxSendWorkRequests = 1024;
        private int maxRecvWorkRequests = 1024;
        private int maxSendSge = 4;
        private int maxRecvSge = 4;
        private int maxInlineData = 256;
        private int serviceLevel = 0;
        private int trafficClass = 0;
        private int retryCount = 7;
        private int rnrRetryCount = 7;
        private int minRnrTimer = 12;
        private int timeout = 14;

        public Builder deviceName(String deviceName) { this.deviceName = deviceName; return this; }
        public Builder port(int port) { this.port = port; return this; }
        public Builder gidIndex(int gidIndex) { this.gidIndex = gidIndex; return this; }
        public Builder maxSendWorkRequests(int n) { this.maxSendWorkRequests = n; return this; }
        public Builder maxRecvWorkRequests(int n) { this.maxRecvWorkRequests = n; return this; }
        public Builder maxSendSge(int n) { this.maxSendSge = n; return this; }
        public Builder maxRecvSge(int n) { this.maxRecvSge = n; return this; }
        public Builder maxInlineData(int n) { this.maxInlineData = n; return this; }
        public Builder serviceLevel(int n) { this.serviceLevel = n; return this; }
        public Builder trafficClass(int n) { this.trafficClass = n; return this; }
        public Builder retryCount(int n) { this.retryCount = n; return this; }
        public Builder rnrRetryCount(int n) { this.rnrRetryCount = n; return this; }
        public Builder minRnrTimer(int n) { this.minRnrTimer = n; return this; }
        public Builder timeout(int n) { this.timeout = n; return this; }

        public RdmaConfig build() {
            return new RdmaConfig(deviceName, port, gidIndex, maxSendWorkRequests, maxRecvWorkRequests,
                    maxSendSge, maxRecvSge, maxInlineData, serviceLevel, trafficClass,
                    retryCount, rnrRetryCount, minRnrTimer, timeout);
        }
    }
}
