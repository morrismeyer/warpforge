package io.surfworks.snakegrinder.cli;

import io.surfworks.snakegrinder.core.SnakeGrinder;

import java.util.Arrays;

public final class SnakeGrinderCli {

    public static void main(String[] args) {
        if (args == null) {
            args = new String[0];
        }

        if (hasFlag(args, "--self-test")) {
            assertGraalVmHost();
            System.out.println(SnakeGrinder.selfTest());
            return;
        }

        if (hasFlag(args, "--help") || hasFlag(args, "-h")) {
            printHelp();
            return;
        }

        System.err.println("Unknown or missing arguments. Use --help.");
        System.exit(2);
    }

    private static boolean hasFlag(String[] args, String flag) {
        return Arrays.asList(args).contains(flag);
    }

    private static void printHelp() {
        System.out.println("snakegrinder-cli");
        System.out.println();
        System.out.println("Usage:");
        System.out.println("  snakegrinder-cli --self-test");
        System.out.println("  snakegrinder-cli --help");
    }

    /**
     * We want SnakeGrinder to run on a GraalVM JDK so GraalPy can use runtime compilation.
     *
     * Do not key only off java.vm.name, because Oracle GraalVM JDK may still report HotSpot.
     * Use java.home and GraalVM specific properties instead.
     */
    private static void assertGraalVmHost() {
        String javaHome = System.getProperty("java.home", "");
        String javaHomeLc = javaHome.toLowerCase();

        String graalHome = System.getProperty("org.graalvm.home");
        String graalVersion = System.getProperty("org.graalvm.version");

        String vmName = System.getProperty("java.vm.name", "");
        String runtimeName = System.getProperty("java.runtime.name", "");
        String vendor = System.getProperty("java.vendor", "");

        boolean looksLikeGraal =
                (graalHome != null && !graalHome.isBlank()) ||
                (graalVersion != null && !graalVersion.isBlank()) ||
                javaHomeLc.contains("graalvm") ||
                vmName.toLowerCase().contains("graal") ||
                runtimeName.toLowerCase().contains("graal");

        if (!looksLikeGraal) {
            throw new IllegalStateException(
                    "SnakeGrinder must run on GraalVM JDK 25 (for GraalPy runtime compilation). " +
                    "Current VM: " + vmName + " / " + runtimeName + " / " + vendor + ". " +
                    "Current java.home: " + javaHome + ". " +
                    "If Gradle toolchains selected a non-Graal JDK 25, install Oracle GraalVM 25 " +
                    "and configure Gradle toolchain discovery to find it."
            );
        }
    }
}
