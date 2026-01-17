package io.surfworks.warpforge.io.rdma;

/**
 * Information about an RDMA-capable device.
 *
 * @param name Device name (e.g., "mlx5_0")
 * @param vendor Device vendor (e.g., "Mellanox")
 * @param portCount Number of ports
 * @param maxMtu Maximum MTU supported
 * @param linkSpeed Link speed in Gbps
 * @param supportsRoCE Whether RoCE (RDMA over Converged Ethernet) is supported
 * @param supportsInfiniBand Whether InfiniBand is supported
 */
public record RdmaDevice(
        String name,
        String vendor,
        int portCount,
        int maxMtu,
        double linkSpeed,
        boolean supportsRoCE,
        boolean supportsInfiniBand
) {

    /**
     * Returns true if this device supports 100GbE or higher speeds.
     */
    public boolean is100GbECapable() {
        return linkSpeed >= 100.0;
    }

    /**
     * Returns a human-readable description of the device.
     */
    public String description() {
        return String.format("%s (%s, %.0f Gbps, %s)",
                name, vendor, linkSpeed,
                supportsInfiniBand ? "IB" : (supportsRoCE ? "RoCE" : "Ethernet"));
    }
}
