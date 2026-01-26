package io.surfworks.warpforge.backend.amd.hip;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;

/**
 * FFM bindings to HIPRTC (HIP Runtime Compilation) API.
 *
 * <p>HIPRTC allows compiling HIP C++ source code to AMD GPU code objects (HSACO)
 * at runtime. This is equivalent to NVRTC for CUDA.
 *
 * <p>Typical workflow:
 * <ol>
 *   <li>Create program from HIP C++ source</li>
 *   <li>Compile program to HSACO</li>
 *   <li>Get compiled code</li>
 *   <li>Load into HIP module</li>
 *   <li>Get kernel function</li>
 *   <li>Launch kernel</li>
 * </ol>
 *
 * @see <a href="https://rocm.docs.amd.com/projects/HIPRTC/en/latest/">HIPRTC Documentation</a>
 */
public final class HiprtcRuntime {

    // Version sentinel - helps verify correct code is deployed
    public static final String VERSION = "2026-01-26-v4";

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup HIPRTC;
    private static final boolean AVAILABLE;

    // HIPRTC result codes
    public static final int HIPRTC_SUCCESS = 0;
    public static final int HIPRTC_ERROR_OUT_OF_MEMORY = 1;
    public static final int HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = 2;
    public static final int HIPRTC_ERROR_INVALID_INPUT = 3;
    public static final int HIPRTC_ERROR_INVALID_PROGRAM = 4;
    public static final int HIPRTC_ERROR_INVALID_OPTION = 5;
    public static final int HIPRTC_ERROR_COMPILATION = 6;
    public static final int HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7;
    public static final int HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8;
    public static final int HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9;
    public static final int HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10;
    public static final int HIPRTC_ERROR_INTERNAL_ERROR = 11;

    // HIPRTC API function handles
    private static final MethodHandle hiprtcCreateProgram;
    private static final MethodHandle hiprtcDestroyProgram;
    private static final MethodHandle hiprtcCompileProgram;
    private static final MethodHandle hiprtcGetCodeSize;
    private static final MethodHandle hiprtcGetCode;
    private static final MethodHandle hiprtcGetProgramLogSize;
    private static final MethodHandle hiprtcGetProgramLog;
    private static final MethodHandle hiprtcGetErrorString;
    private static final MethodHandle hiprtcVersion;

    static {
        SymbolLookup lookup = null;
        boolean available = false;

        try {
            // Try to load libhiprtc.so (Linux)
            lookup = SymbolLookup.libraryLookup("hiprtc", Arena.global());
            available = true;
        } catch (IllegalArgumentException e) {
            // Try common ROCm installation path
            try {
                lookup = SymbolLookup.libraryLookup("/opt/rocm/lib/libhiprtc.so", Arena.global());
                available = true;
            } catch (IllegalArgumentException e2) {
                // HIPRTC not available
                lookup = SymbolLookup.loaderLookup();
            }
        }

        HIPRTC = lookup;
        AVAILABLE = available;

        if (available) {
            // hiprtcResult hiprtcCreateProgram(hiprtcProgram* prog, const char* src,
            //     const char* name, int numHeaders, const char** headers, const char** headerNames)
            hiprtcCreateProgram = downcall("hiprtcCreateProgram", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,  // prog (output)
                ValueLayout.ADDRESS,  // src
                ValueLayout.ADDRESS,  // name
                ValueLayout.JAVA_INT, // numHeaders
                ValueLayout.ADDRESS,  // headers
                ValueLayout.ADDRESS   // headerNames
            ));

            // hiprtcResult hiprtcDestroyProgram(hiprtcProgram* prog)
            hiprtcDestroyProgram = downcall("hiprtcDestroyProgram", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

            // hiprtcResult hiprtcCompileProgram(hiprtcProgram prog, int numOptions, const char** options)
            // Note: hiprtcProgram is a pointer type - use ADDRESS not JAVA_LONG
            hiprtcCompileProgram = downcall("hiprtcCompileProgram", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,   // prog (pointer type)
                ValueLayout.JAVA_INT,  // numOptions
                ValueLayout.ADDRESS    // options
            ));

            // hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t* codeSizeRet)
            hiprtcGetCodeSize = downcall("hiprtcGetCodeSize", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,   // prog (pointer type)
                ValueLayout.ADDRESS    // codeSizeRet
            ));

            // hiprtcResult hiprtcGetCode(hiprtcProgram prog, char* code)
            hiprtcGetCode = downcall("hiprtcGetCode", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,   // prog (pointer type)
                ValueLayout.ADDRESS    // code
            ));

            // hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog, size_t* logSizeRet)
            hiprtcGetProgramLogSize = downcall("hiprtcGetProgramLogSize", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,   // prog (pointer type)
                ValueLayout.ADDRESS    // logSizeRet
            ));

            // hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog, char* log)
            hiprtcGetProgramLog = downcall("hiprtcGetProgramLog", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,   // prog (pointer type)
                ValueLayout.ADDRESS    // log
            ));

            // const char* hiprtcGetErrorString(hiprtcResult result)
            hiprtcGetErrorString = downcall("hiprtcGetErrorString", FunctionDescriptor.of(
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

            // hiprtcResult hiprtcVersion(int* major, int* minor)
            hiprtcVersion = downcall("hiprtcVersion", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

        } else {
            hiprtcCreateProgram = null;
            hiprtcDestroyProgram = null;
            hiprtcCompileProgram = null;
            hiprtcGetCodeSize = null;
            hiprtcGetCode = null;
            hiprtcGetProgramLogSize = null;
            hiprtcGetProgramLog = null;
            hiprtcGetErrorString = null;
            hiprtcVersion = null;
        }
    }

    private static MethodHandle downcall(String name, FunctionDescriptor descriptor) {
        return HIPRTC.find(name)
            .map(symbol -> LINKER.downcallHandle(symbol, descriptor))
            .orElse(null);
    }

    private HiprtcRuntime() {}

    /**
     * Check if HIPRTC is available on this system.
     */
    public static boolean isAvailable() {
        return AVAILABLE;
    }

    /**
     * Ensure HIPRTC is available, throwing if not.
     */
    public static void ensureAvailable() {
        if (!AVAILABLE) {
            throw new UnsupportedOperationException("HIPRTC is not available on this system");
        }
    }

    /**
     * Get HIPRTC version.
     *
     * @return int array [major, minor]
     */
    public static int[] getVersion(Arena arena) throws Throwable {
        ensureAvailable();
        MemorySegment majorPtr = arena.allocate(ValueLayout.JAVA_INT);
        MemorySegment minorPtr = arena.allocate(ValueLayout.JAVA_INT);
        int result = (int) hiprtcVersion.invokeExact(majorPtr, minorPtr);
        checkError(result, "hiprtcVersion");
        return new int[]{
            majorPtr.get(ValueLayout.JAVA_INT, 0),
            minorPtr.get(ValueLayout.JAVA_INT, 0)
        };
    }

    /**
     * Compile HIP C++ source code to binary.
     *
     * @param source HIP C++ source code
     * @param kernelName Name of the program (for error messages)
     * @param options Compiler options (e.g., "-O3", "--gpu-architecture=gfx1100")
     * @return Compiled binary (HSACO) suitable for hipModuleLoadData
     */
    public static byte[] compile(String source, String kernelName, String... options) throws Throwable {
        ensureAvailable();

        String step = "init";
        try (Arena arena = Arena.ofConfined()) {
            step = "allocate-progPtr";
            // Create program - hiprtcCreateProgram takes hiprtcProgram* (pointer to pointer)
            // hiprtcProgram is a pointer type, so we allocate space for an ADDRESS
            MemorySegment progPtr = arena.allocate(ValueLayout.ADDRESS);

            step = "allocate-srcSegment";
            MemorySegment srcSegment = arena.allocateFrom(source + "\0", StandardCharsets.UTF_8);

            step = "allocate-nameSegment";
            MemorySegment nameSegment = arena.allocateFrom(kernelName + "\0", StandardCharsets.UTF_8);

            step = "hiprtcCreateProgram";
            int result = (int) hiprtcCreateProgram.invokeExact(
                progPtr, srcSegment, nameSegment,
                0, MemorySegment.NULL, MemorySegment.NULL);
            checkError(result, "hiprtcCreateProgram");

            step = "read-prog";
            // Read the hiprtcProgram handle as a MemorySegment (pointer type)
            MemorySegment prog = progPtr.get(ValueLayout.ADDRESS, 0);

            try {
                step = "setup-options";
                // Set up compiler options
                MemorySegment optionsPtr;
                if (options.length > 0) {
                    // Allocate array of pointers to strings
                    optionsPtr = arena.allocate(ValueLayout.ADDRESS, options.length);
                    for (int i = 0; i < options.length; i++) {
                        MemorySegment optStr = arena.allocateFrom(options[i] + "\0", StandardCharsets.UTF_8);
                        optionsPtr.setAtIndex(ValueLayout.ADDRESS, i, optStr);
                    }
                } else {
                    optionsPtr = MemorySegment.NULL;
                }

                step = "hiprtcCompileProgram";
                // Compile - prog is passed as ADDRESS (pointer type)
                result = (int) hiprtcCompileProgram.invokeExact(prog, options.length, optionsPtr);

                if (result != HIPRTC_SUCCESS) {
                    // Get compilation log
                    String log = getProgramLog(arena, prog);
                    throw new HiprtcException("Compilation failed for " + kernelName + ": " +
                        getErrorString(result) + "\nCompilation log:\n" + log);
                }

                step = "hiprtcGetCodeSize";
                // Get code size
                MemorySegment sizePtr = arena.allocate(ValueLayout.JAVA_LONG);
                result = (int) hiprtcGetCodeSize.invokeExact(prog, sizePtr);
                checkError(result, "hiprtcGetCodeSize");
                long codeSize = sizePtr.get(ValueLayout.JAVA_LONG, 0);

                step = "hiprtcGetCode";
                // Get code
                MemorySegment codeSegment = arena.allocate(codeSize);
                result = (int) hiprtcGetCode.invokeExact(prog, codeSegment);
                checkError(result, "hiprtcGetCode");

                return codeSegment.toArray(ValueLayout.JAVA_BYTE);

            } finally {
                step = "hiprtcDestroyProgram";
                // Destroy program - must capture return value due to invokeExact semantics
                @SuppressWarnings("unused")
                int destroyResult = (int) hiprtcDestroyProgram.invokeExact(progPtr);
            }
        } catch (Throwable t) {
            throw new HiprtcException("HIPRTC compile failed at step '" + step + "' for " + kernelName +
                " (version=" + VERSION + "): " + t.getMessage(), t);
        }
    }

    /**
     * Get the compilation log for a program.
     */
    private static String getProgramLog(Arena arena, MemorySegment prog) throws Throwable {
        MemorySegment sizePtr = arena.allocate(ValueLayout.JAVA_LONG);
        int result = (int) hiprtcGetProgramLogSize.invokeExact(prog, sizePtr);
        if (result != HIPRTC_SUCCESS) {
            return "(failed to get log size)";
        }

        long logSize = sizePtr.get(ValueLayout.JAVA_LONG, 0);
        if (logSize <= 1) {
            return "(no log)";
        }

        MemorySegment logSegment = arena.allocate(logSize);
        result = (int) hiprtcGetProgramLog.invokeExact(prog, logSegment);
        if (result != HIPRTC_SUCCESS) {
            return "(failed to get log)";
        }

        return logSegment.getString(0);
    }

    /**
     * Get a human-readable error string for a HIPRTC result code.
     */
    public static String getErrorString(int result) {
        if (!AVAILABLE || hiprtcGetErrorString == null) {
            return "HIPRTC_ERROR_" + result;
        }
        try {
            MemorySegment strPtr = (MemorySegment) hiprtcGetErrorString.invokeExact(result);
            if (!strPtr.equals(MemorySegment.NULL)) {
                return strPtr.reinterpret(256).getString(0);
            }
        } catch (Throwable t) {
            // Ignore
        }
        return "HIPRTC_ERROR_" + result;
    }

    /**
     * Check a HIPRTC result and throw if not success.
     */
    public static void checkError(int result, String operation) {
        if (result != HIPRTC_SUCCESS) {
            throw new HiprtcException(operation + " failed: " + getErrorString(result) + " (" + result + ")");
        }
    }

    /**
     * Exception thrown when a HIPRTC operation fails.
     */
    public static class HiprtcException extends RuntimeException {
        public HiprtcException(String message) {
            super(message);
        }

        public HiprtcException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
