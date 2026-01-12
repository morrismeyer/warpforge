package io.surfworks.warpforge.license;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.NetworkInterface;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.Enumeration;
import java.util.HexFormat;

/**
 * Generates a stable machine fingerprint for license activation binding.
 *
 * <p>Uses hardware identifiers that survive reboots but detect when a license
 * is moved to a different machine.
 */
public final class MachineFingerprint {

    private MachineFingerprint() {}

    /**
     * Generate a stable fingerprint for this machine.
     *
     * <p>Components (platform-dependent):
     * <ul>
     *   <li>macOS: IOPlatformSerialNumber (hardware serial)</li>
     *   <li>Linux: /etc/machine-id + /sys/class/dmi/id/product_uuid</li>
     * </ul>
     *
     * @return SHA-256 hash of hardware identifiers (first 32 chars)
     */
    public static String generate() {
        String os = System.getProperty("os.name", "").toLowerCase();

        String rawIdentifier;
        if (os.contains("mac")) {
            rawIdentifier = getMacSerialNumber();
        } else if (os.contains("linux")) {
            rawIdentifier = getLinuxMachineId();
        } else {
            // Fallback: use user.name + user.home (not great, but works)
            rawIdentifier = System.getProperty("user.name", "unknown") +
                           System.getProperty("user.home", "/unknown");
        }

        // Hash and truncate to 32 chars for readability
        return sha256(rawIdentifier).substring(0, 32);
    }

    /**
     * Get a short, human-readable machine name for activation.
     */
    public static String getMachineName() {
        String hostname = getHostname();
        String os = System.getProperty("os.name", "Unknown");

        if (os.toLowerCase().contains("mac")) {
            return hostname + " (macOS)";
        } else if (os.toLowerCase().contains("linux")) {
            return hostname + " (Linux)";
        } else {
            return hostname + " (" + os + ")";
        }
    }

    private static String getMacSerialNumber() {
        try {
            ProcessBuilder pb = new ProcessBuilder(
                "ioreg", "-rd1", "-c", "IOPlatformExpertDevice"
            );
            Process p = pb.start();

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(p.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.contains("IOPlatformSerialNumber")) {
                        int start = line.indexOf("\"", line.indexOf("=")) + 1;
                        int end = line.lastIndexOf("\"");
                        if (start > 0 && end > start) {
                            return line.substring(start, end);
                        }
                    }
                }
            }
            p.waitFor();
        } catch (Exception ignored) {
        }

        // Fallback to MAC address
        return getNetworkMac();
    }

    private static String getLinuxMachineId() {
        StringBuilder id = new StringBuilder();

        // /etc/machine-id is standard on systemd systems
        try {
            Path machineIdPath = Path.of("/etc/machine-id");
            if (Files.exists(machineIdPath)) {
                id.append(Files.readString(machineIdPath).trim());
            }
        } catch (Exception ignored) {
        }

        // DMI product UUID (if accessible)
        try {
            Path uuidPath = Path.of("/sys/class/dmi/id/product_uuid");
            if (Files.exists(uuidPath)) {
                id.append(Files.readString(uuidPath).trim());
            }
        } catch (Exception ignored) {
            // Requires root on many systems, fallback is fine
        }

        if (id.isEmpty()) {
            return getNetworkMac();
        }

        return id.toString();
    }

    private static String getNetworkMac() {
        try {
            Enumeration<NetworkInterface> interfaces = NetworkInterface.getNetworkInterfaces();
            while (interfaces.hasMoreElements()) {
                NetworkInterface ni = interfaces.nextElement();
                byte[] mac = ni.getHardwareAddress();
                if (mac != null && mac.length > 0 && !ni.isLoopback()) {
                    return HexFormat.of().formatHex(mac);
                }
            }
        } catch (Exception ignored) {
        }
        return "unknown-" + System.currentTimeMillis();
    }

    private static String getHostname() {
        try {
            return java.net.InetAddress.getLocalHost().getHostName();
        } catch (Exception e) {
            return "unknown";
        }
    }

    private static String sha256(String input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(input.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(hash);
        } catch (Exception e) {
            throw new RuntimeException("SHA-256 not available", e);
        }
    }
}
