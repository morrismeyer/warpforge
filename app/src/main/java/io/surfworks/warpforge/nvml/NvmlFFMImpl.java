package io.surfworks.warpforge.nvml;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.*;

/**
 * Linux-only NVML implementation using Java 25 Foreign Function & Memory.
 *
 * This expects libnvidia-ml.so.1 to be available at:
 *   /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
 *
 * It is designed so that failures to bind or call NVML are thrown as RuntimeExceptions,
 * and the higher-level Nvml loader can fall back to NvmlMock when needed.
 */
public final class NvmlFFMImpl implements NvmlApi {

    private static final String NVML_LIB =
            "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1";

    private static final Linker LINKER = Linker.nativeLinker();
    private static final Arena LIB_ARENA = Arena.ofShared();
    private static final SymbolLookup LOOKUP;

    private static final MethodHandle nvmlInit_v2;
    private static final MethodHandle nvmlShutdown;
    private static final MethodHandle nvmlSystemGetDriverVersion;
    private static final MethodHandle nvmlDeviceGetCount;

    static {
        try {
            System.load(NVML_LIB);
            LOOKUP = SymbolLookup.libraryLookup(NVML_LIB, LIB_ARENA);

            nvmlInit_v2 = LINKER.downcallHandle(
                    LOOKUP.find("nvmlInit_v2").orElseThrow(),
                    FunctionDescriptor.of(JAVA_INT)
            );

            nvmlShutdown = LINKER.downcallHandle(
                    LOOKUP.find("nvmlShutdown").orElseThrow(),
                    FunctionDescriptor.of(JAVA_INT)
            );

            nvmlSystemGetDriverVersion = LINKER.downcallHandle(
                    LOOKUP.find("nvmlSystemGetDriverVersion").orElseThrow(),
                    FunctionDescriptor.of(
                            JAVA_INT,  // return
                            ADDRESS,   // char* version
                            JAVA_INT   // int length
                    )
            );

            nvmlDeviceGetCount = LINKER.downcallHandle(
                    LOOKUP.find("nvmlDeviceGetCount").orElseThrow(),
                    FunctionDescriptor.of(
                            JAVA_INT,  // return
                            ADDRESS    // int* deviceCount
                    )
            );

        } catch (Throwable t) {
            throw new RuntimeException("Failed to bind NVML functions", t);
        }
    }

    public NvmlFFMImpl() {
        int rc = callNvmlInit();
        if (rc != 0) {
            throw new RuntimeException("nvmlInit_v2 failed: rc=" + rc);
        }

        // Best-effort shutdown on JVM exit.
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                nvmlShutdown.invokeExact();
            } catch (Throwable ignored) {
                // ignore shutdown errors
            }
        }));
    }

    private int callNvmlInit() {
        try {
            return (int) nvmlInit_v2.invokeExact();
        } catch (Throwable t) {
            throw new RuntimeException("nvmlInit_v2 invocation failed", t);
        }
    }

    @Override
    public String backendName() {
        return "nvml-ffm";
    }

    @Override
    public String driverVersion() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(256);
            int rc = (int) nvmlSystemGetDriverVersion.invokeExact(buf, 256);
            if (rc != 0) {
                throw new RuntimeException("nvmlSystemGetDriverVersion failed: rc=" + rc);
            }
            return readCString(buf);
        } catch (Throwable t) {
            throw new RuntimeException("driverVersion() failed", t);
        }
    }

    @Override
    public int deviceCount() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment countBuf = arena.allocate(JAVA_INT);
            int rc = (int) nvmlDeviceGetCount.invokeExact(countBuf);
            if (rc != 0) {
                throw new RuntimeException("nvmlDeviceGetCount failed: rc=" + rc);
            }
            return countBuf.get(JAVA_INT, 0);
        } catch (Throwable t) {
            throw new RuntimeException("deviceCount() failed", t);
        }
    }

    private static String readCString(MemorySegment seg) {
        StringBuilder sb = new StringBuilder();
        long offset = 0;
        while (true) {
            byte b = seg.get(JAVA_BYTE, offset);
            if (b == 0) break;
            sb.append((char) (b & 0xFF));
            offset++;
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        return "NvmlFFMImpl{driverVersion=" + driverVersion()
               + ", deviceCount=" + deviceCount() + "}";
    }
}

