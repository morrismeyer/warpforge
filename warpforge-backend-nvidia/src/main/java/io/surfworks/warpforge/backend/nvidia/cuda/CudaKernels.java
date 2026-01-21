package io.surfworks.warpforge.backend.nvidia.cuda;

/**
 * PTX kernel generation with salt-based instrumentation.
 *
 * <p>All kernels are generated from templates with optional instrumentation
 * injected based on the salt level. The instrumentation is part of the same
 * code path - not a separate implementation - ensuring that performance
 * measurements reflect actual production behavior with known, quantifiable overhead.
 *
 * <h2>Salt Levels</h2>
 * <ul>
 *   <li>{@link #SALT_NONE} - Production kernel, no instrumentation</li>
 *   <li>{@link #SALT_TIMING} - Adds cycle counters (known overhead: ~8 instructions)</li>
 *   <li>{@link #SALT_TRACE} - Adds memory access logging (higher overhead)</li>
 * </ul>
 *
 * <h2>Design Principle</h2>
 * <p>There is ONE implementation of each operation. Instrumentation variants
 * are generated from the same template with conditional sections enabled.
 * This avoids Heisenbug scenarios where switching between "debug" and "production"
 * paths would measure entirely different code.
 */
public final class CudaKernels {

    private CudaKernels() {}

    // ==================== Salt Levels ====================

    /** No instrumentation - production kernel */
    public static final int SALT_NONE = 0;

    /** Timing instrumentation - cycle counters with known overhead */
    public static final int SALT_TIMING = 1;

    /** Full trace - memory access patterns, warp divergence (high overhead) */
    public static final int SALT_TRACE = 2;

    // ==================== Kernel Configuration ====================

    /** Standard block size for elementwise operations */
    public static final int ELEMENTWISE_BLOCK_SIZE = 256;

    /** Module name for elementwise operations */
    public static final String ELEMENTWISE_MODULE = "elementwise_ops";

    // ==================== PTX Generation ====================

    /**
     * Generate PTX for element-wise float32 addition.
     *
     * <p>Kernel signature: add_f32(float* a, float* b, float* out, int n, long* timing)
     * <p>The timing parameter is only used when salt > SALT_NONE.
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateAddF32(int salt) {
        StringBuilder ptx = new StringBuilder();

        // PTX header
        ptx.append("""
            //
            // add_f32: Element-wise addition of float32 arrays
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            """.formatted(salt));

        // Entry point with parameters
        ptx.append("""
            .visible .entry add_f32(
                .param .u64 a_ptr,
                .param .u64 b_ptr,
                .param .u64 out_ptr,
                .param .u32 n
            """);

        // Add timing accumulator parameter if instrumented
        if (salt >= SALT_TIMING) {
            ptx.append("    ,.param .u64 timing_ptr\n");
        }

        ptx.append(")\n{\n");

        // Register declarations
        ptx.append("""
                .reg .pred  %p<2>;
                .reg .f32   %f<4>;
                .reg .b32   %r<6>;
                .reg .b64   %rd<10>;
            """);

        // Additional registers for timing
        if (salt >= SALT_TIMING) {
            ptx.append("""
                .reg .b64   %rd_t0, %rd_t1, %rd_delta;
            """);
        }

        // Calculate global thread index
        ptx.append("""

                // Calculate global thread index: i = blockIdx.x * blockDim.x + threadIdx.x
                mov.u32         %r1, %ctaid.x;
                mov.u32         %r2, %ntid.x;
                mov.u32         %r3, %tid.x;
                mad.lo.s32      %r4, %r1, %r2, %r3;

                // Load n and check bounds
                ld.param.u32    %r5, [n];
                setp.ge.s32     %p1, %r4, %r5;
                @%p1 bra        EXIT;

                // Load base pointers
                ld.param.u64    %rd1, [a_ptr];
                ld.param.u64    %rd2, [b_ptr];
                ld.param.u64    %rd3, [out_ptr];

                // Calculate byte offset: offset = i * 4 (sizeof float)
                cvt.s64.s32     %rd4, %r4;
                shl.b64         %rd5, %rd4, 2;

                // Calculate element addresses
                add.s64         %rd6, %rd1, %rd5;
                add.s64         %rd7, %rd2, %rd5;
                add.s64         %rd8, %rd3, %rd5;

                // Load a[i] and b[i]
                ld.global.f32   %f1, [%rd6];
                ld.global.f32   %f2, [%rd7];

            """);

        // Timing: capture start
        if (salt >= SALT_TIMING) {
            ptx.append("""
                // [SALT_TIMING] Capture start time
                mov.u64         %rd_t0, %globaltimer;

            """);
        }

        // Core computation
        ptx.append("""
                // Compute: out[i] = a[i] + b[i]
                add.f32         %f3, %f1, %f2;

            """);

        // Timing: capture end and accumulate
        if (salt >= SALT_TIMING) {
            ptx.append("""
                // [SALT_TIMING] Capture end time and accumulate
                mov.u64         %rd_t1, %globaltimer;
                sub.u64         %rd_delta, %rd_t1, %rd_t0;
                ld.param.u64    %rd9, [timing_ptr];
                atom.global.add.u64 [%rd9], %rd_delta;

            """);
        }

        // Store result
        ptx.append("""
                // Store result
                st.global.f32   [%rd8], %f3;

            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    /**
     * Generate PTX for element-wise float32 multiplication.
     */
    public static String generateMultiplyF32(int salt) {
        // Same structure as add, just different core operation
        StringBuilder ptx = new StringBuilder();

        ptx.append("""
            //
            // multiply_f32: Element-wise multiplication of float32 arrays
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            .visible .entry multiply_f32(
                .param .u64 a_ptr,
                .param .u64 b_ptr,
                .param .u64 out_ptr,
                .param .u32 n
            """.formatted(salt));

        if (salt >= SALT_TIMING) {
            ptx.append("    ,.param .u64 timing_ptr\n");
        }

        ptx.append("""
            )
            {
                .reg .pred  %p<2>;
                .reg .f32   %f<4>;
                .reg .b32   %r<6>;
                .reg .b64   %rd<10>;
            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    .reg .b64   %rd_t0, %rd_t1, %rd_delta;\n");
        }

        ptx.append("""

                mov.u32         %r1, %ctaid.x;
                mov.u32         %r2, %ntid.x;
                mov.u32         %r3, %tid.x;
                mad.lo.s32      %r4, %r1, %r2, %r3;

                ld.param.u32    %r5, [n];
                setp.ge.s32     %p1, %r4, %r5;
                @%p1 bra        EXIT;

                ld.param.u64    %rd1, [a_ptr];
                ld.param.u64    %rd2, [b_ptr];
                ld.param.u64    %rd3, [out_ptr];

                cvt.s64.s32     %rd4, %r4;
                shl.b64         %rd5, %rd4, 2;

                add.s64         %rd6, %rd1, %rd5;
                add.s64         %rd7, %rd2, %rd5;
                add.s64         %rd8, %rd3, %rd5;

                ld.global.f32   %f1, [%rd6];
                ld.global.f32   %f2, [%rd7];

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        ptx.append("    mul.f32         %f3, %f1, %f2;\n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("""
                    mov.u64         %rd_t1, %globaltimer;
                    sub.u64         %rd_delta, %rd_t1, %rd_t0;
                    ld.param.u64    %rd9, [timing_ptr];
                    atom.global.add.u64 [%rd9], %rd_delta;

            """);
        }

        ptx.append("""
                st.global.f32   [%rd8], %f3;

            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    /**
     * Generate PTX for element-wise float32 subtraction.
     */
    public static String generateSubtractF32(int salt) {
        return generateBinaryElementwiseF32("subtract", "sub.f32", "a[i] - b[i]", salt);
    }

    /**
     * Generate PTX for element-wise float32 division.
     */
    public static String generateDivideF32(int salt) {
        return generateBinaryElementwiseF32("divide", "div.approx.f32", "a[i] / b[i]", salt);
    }

    /**
     * Generate PTX for element-wise float32 maximum.
     */
    public static String generateMaximumF32(int salt) {
        return generateBinaryElementwiseF32("maximum", "max.f32", "max(a[i], b[i])", salt);
    }

    /**
     * Generate PTX for element-wise float32 minimum.
     */
    public static String generateMinimumF32(int salt) {
        return generateBinaryElementwiseF32("minimum", "min.f32", "min(a[i], b[i])", salt);
    }

    /**
     * Generate PTX for a binary elementwise float32 operation.
     *
     * @param opName Operation name for comments and entry point
     * @param ptxInstruction The PTX instruction (e.g., "add.f32", "mul.f32")
     * @param comment Description of the operation
     * @param salt Instrumentation level
     * @return PTX source code
     */
    private static String generateBinaryElementwiseF32(String opName, String ptxInstruction,
                                                        String comment, int salt) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("""
            //
            // %s_f32: Element-wise %s of float32 arrays
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            .visible .entry %s_f32(
                .param .u64 a_ptr,
                .param .u64 b_ptr,
                .param .u64 out_ptr,
                .param .u32 n
            """.formatted(opName, comment, salt, opName));

        if (salt >= SALT_TIMING) {
            ptx.append("    ,.param .u64 timing_ptr\n");
        }

        ptx.append("""
            )
            {
                .reg .pred  %p<2>;
                .reg .f32   %f<4>;
                .reg .b32   %r<6>;
                .reg .b64   %rd<10>;
            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    .reg .b64   %rd_t0, %rd_t1, %rd_delta;\n");
        }

        ptx.append("""

                mov.u32         %r1, %ctaid.x;
                mov.u32         %r2, %ntid.x;
                mov.u32         %r3, %tid.x;
                mad.lo.s32      %r4, %r1, %r2, %r3;

                ld.param.u32    %r5, [n];
                setp.ge.s32     %p1, %r4, %r5;
                @%p1 bra        EXIT;

                ld.param.u64    %rd1, [a_ptr];
                ld.param.u64    %rd2, [b_ptr];
                ld.param.u64    %rd3, [out_ptr];

                cvt.s64.s32     %rd4, %r4;
                shl.b64         %rd5, %rd4, 2;

                add.s64         %rd6, %rd1, %rd5;
                add.s64         %rd7, %rd2, %rd5;
                add.s64         %rd8, %rd3, %rd5;

                ld.global.f32   %f1, [%rd6];
                ld.global.f32   %f2, [%rd7];

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        // Core operation
        ptx.append("    ").append(ptxInstruction).append("         %f3, %f1, %f2;\n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("""
                    mov.u64         %rd_t1, %globaltimer;
                    sub.u64         %rd_delta, %rd_t1, %rd_t0;
                    ld.param.u64    %rd9, [timing_ptr];
                    atom.global.add.u64 [%rd9], %rd_delta;

            """);
        }

        ptx.append("""
                st.global.f32   [%rd8], %f3;

            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    // ==================== Unary Elementwise PTX Generation ====================

    /**
     * Generate PTX for element-wise float32 negation.
     */
    public static String generateNegateF32(int salt) {
        return generateUnaryElementwiseF32("negate", "neg.f32         %f2, %f1;", "-x", salt);
    }

    /**
     * Generate PTX for element-wise float32 absolute value.
     */
    public static String generateAbsF32(int salt) {
        return generateUnaryElementwiseF32("abs", "abs.f32         %f2, %f1;", "|x|", salt);
    }

    /**
     * Generate PTX for element-wise float32 exponential (e^x).
     * Uses: e^x = 2^(x * log2(e)) where log2(e) ≈ 1.4426950408889634
     */
    public static String generateExpF32(int salt) {
        String ptxOps = """
                    // e^x = 2^(x * log2(e))
                    mul.f32         %f2, %f1, 0f3FB8AA3B;  // log2(e) = 1.4426950408889634
                    ex2.approx.f32  %f2, %f2;""";
        return generateUnaryElementwiseF32("exp", ptxOps, "e^x", salt);
    }

    /**
     * Generate PTX for element-wise float32 natural logarithm (ln(x)).
     * Uses: ln(x) = log2(x) * ln(2) where ln(2) ≈ 0.6931471805599453
     */
    public static String generateLogF32(int salt) {
        String ptxOps = """
                    // ln(x) = log2(x) * ln(2)
                    lg2.approx.f32  %f2, %f1;
                    mul.f32         %f2, %f2, 0f3F317218;  // ln(2) = 0.6931471805599453""";
        return generateUnaryElementwiseF32("log", ptxOps, "ln(x)", salt);
    }

    /**
     * Generate PTX for element-wise float32 square root.
     */
    public static String generateSqrtF32(int salt) {
        return generateUnaryElementwiseF32("sqrt", "sqrt.approx.f32 %f2, %f1;", "sqrt(x)", salt);
    }

    /**
     * Generate PTX for element-wise float32 hyperbolic tangent.
     * Uses: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
     */
    public static String generateTanhF32(int salt) {
        String ptxOps = """
                    // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
                    // First compute 2x
                    add.f32         %f2, %f1, %f1;
                    // Compute e^(2x) using 2^(2x * log2(e))
                    mul.f32         %f2, %f2, 0f3FB8AA3B;  // log2(e)
                    ex2.approx.f32  %f2, %f2;
                    // Now %f2 = e^(2x), compute (e^(2x) - 1) and (e^(2x) + 1)
                    add.f32         %f3, %f2, 0f3F800000;  // e^(2x) + 1 (1.0f)
                    add.f32         %f2, %f2, 0fBF800000;  // e^(2x) - 1 (-1.0f, so adding it)
                    // Divide: (e^(2x) - 1) / (e^(2x) + 1)
                    div.approx.f32  %f2, %f2, %f3;""";
        return generateUnaryElementwiseF32("tanh", ptxOps, "tanh(x)", salt, 4);
    }

    /**
     * Generate PTX for element-wise float32 reciprocal square root.
     * rsqrt(x) = 1/sqrt(x)
     */
    public static String generateRsqrtF32(int salt) {
        return generateUnaryElementwiseF32("rsqrt", "rsqrt.approx.f32 %f2, %f1;", "1/sqrt(x)", salt);
    }

    /**
     * Generate PTX for element-wise float32 sine.
     */
    public static String generateSinF32(int salt) {
        return generateUnaryElementwiseF32("sin", "sin.approx.f32 %f2, %f1;", "sin(x)", salt);
    }

    /**
     * Generate PTX for element-wise float32 cosine.
     */
    public static String generateCosF32(int salt) {
        return generateUnaryElementwiseF32("cos", "cos.approx.f32 %f2, %f1;", "cos(x)", salt);
    }

    /**
     * Generate PTX for element-wise float32 ceiling.
     * Uses cvt with round-to-positive-infinity mode.
     */
    public static String generateCeilF32(int salt) {
        return generateUnaryElementwiseF32("ceil", "cvt.rpi.f32.f32 %f2, %f1;", "ceil(x)", salt);
    }

    /**
     * Generate PTX for element-wise float32 floor.
     * Uses cvt with round-to-negative-infinity mode.
     */
    public static String generateFloorF32(int salt) {
        return generateUnaryElementwiseF32("floor", "cvt.rmi.f32.f32 %f2, %f1;", "floor(x)", salt);
    }

    /**
     * Generate PTX for element-wise float32 sign function.
     * Returns -1.0 if x < 0, 0.0 if x == 0, 1.0 if x > 0.
     */
    public static String generateSignF32(int salt) {
        // Use copysign to get the sign, but need special handling for zero
        // Approach: set %f2 = 1.0 if x > 0, -1.0 if x < 0, 0.0 if x == 0
        // Note: %p1 is used for bounds check, so we use %p2 and %p3 here
        String ptxOps = """
                    // sign(x): -1 if x<0, 0 if x==0, 1 if x>0
                    // Use predicate comparison and selection
                    setp.gt.f32     %p2, %f1, 0f00000000;  // p2 = (x > 0)
                    setp.lt.f32     %p3, %f1, 0f00000000;  // p3 = (x < 0)
                    mov.f32         %f2, 0f00000000;       // default: 0.0
                    @%p2 mov.f32    %f2, 0f3F800000;       // if x > 0: 1.0
                    @%p3 mov.f32    %f2, 0fBF800000;       // if x < 0: -1.0""";
        return generateUnaryElementwiseF32("sign", ptxOps, "sign(x)", salt, 3, 4);
    }

    /**
     * Generate PTX for a unary elementwise float32 operation.
     *
     * @param opName Operation name for comments and entry point
     * @param ptxInstructions The PTX instruction(s) (input in %f1, output in %f2)
     * @param comment Description of the operation
     * @param salt Instrumentation level
     * @return PTX source code
     */
    private static String generateUnaryElementwiseF32(String opName, String ptxInstructions,
                                                       String comment, int salt) {
        return generateUnaryElementwiseF32(opName, ptxInstructions, comment, salt, 3);
    }

    /**
     * Generate PTX for a unary elementwise float32 operation with custom register count.
     */
    private static String generateUnaryElementwiseF32(String opName, String ptxInstructions,
                                                       String comment, int salt, int floatRegs) {
        return generateUnaryElementwiseF32(opName, ptxInstructions, comment, salt, floatRegs, 2);
    }

    /**
     * Generate PTX for a unary elementwise float32 operation with custom register counts.
     *
     * @param opName The operation name
     * @param ptxInstructions The PTX instruction(s) (input in %f1, output in %f2)
     * @param comment Description of the operation
     * @param salt Instrumentation level
     * @param floatRegs Number of float registers needed
     * @param predicateRegs Number of predicate registers needed (must include %p1 for bounds check)
     * @return PTX source code
     */
    private static String generateUnaryElementwiseF32(String opName, String ptxInstructions,
                                                       String comment, int salt, int floatRegs,
                                                       int predicateRegs) {
        StringBuilder ptx = new StringBuilder();

        // Header with operation name and salt level
        ptx.append("//\n");
        ptx.append("// ").append(opName).append("_f32: Element-wise ").append(comment).append(" of float32 arrays\n");
        ptx.append("// Salt level: ").append(salt).append("\n");
        ptx.append("//\n");
        ptx.append(".version 7.0\n");
        ptx.append(".target sm_50\n");
        ptx.append(".address_size 64\n\n");

        ptx.append(".visible .entry ").append(opName).append("_f32(\n");
        ptx.append("    .param .u64 in_ptr,\n");
        ptx.append("    .param .u64 out_ptr,\n");
        ptx.append("    .param .u32 n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("    ,.param .u64 timing_ptr\n");
        }

        ptx.append(")\n{\n");
        ptx.append("    .reg .pred  %p<").append(predicateRegs).append(">;\n");
        ptx.append("    .reg .f32   %f<").append(floatRegs).append(">;\n");
        ptx.append("    .reg .b32   %r<6>;\n");
        ptx.append("    .reg .b64   %rd<8>;\n");

        if (salt >= SALT_TIMING) {
            ptx.append("    .reg .b64   %rd_t0, %rd_t1, %rd_delta;\n");
        }

        ptx.append("\n");
        ptx.append("    mov.u32         %r1, %ctaid.x;\n");
        ptx.append("    mov.u32         %r2, %ntid.x;\n");
        ptx.append("    mov.u32         %r3, %tid.x;\n");
        ptx.append("    mad.lo.s32      %r4, %r1, %r2, %r3;\n\n");

        ptx.append("    ld.param.u32    %r5, [n];\n");
        ptx.append("    setp.ge.s32     %p1, %r4, %r5;\n");
        ptx.append("    @%p1 bra        EXIT;\n\n");

        ptx.append("    ld.param.u64    %rd1, [in_ptr];\n");
        ptx.append("    ld.param.u64    %rd2, [out_ptr];\n\n");

        ptx.append("    cvt.s64.s32     %rd3, %r4;\n");
        ptx.append("    shl.b64         %rd4, %rd3, 2;\n\n");

        ptx.append("    add.s64         %rd5, %rd1, %rd4;\n");
        ptx.append("    add.s64         %rd6, %rd2, %rd4;\n\n");

        ptx.append("    ld.global.f32   %f1, [%rd5];\n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        // Core operation(s)
        ptx.append(ptxInstructions).append("\n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t1, %globaltimer;\n");
            ptx.append("    sub.u64         %rd_delta, %rd_t1, %rd_t0;\n");
            ptx.append("    ld.param.u64    %rd7, [timing_ptr];\n");
            ptx.append("    atom.global.add.u64 [%rd7], %rd_delta;\n\n");
        }

        ptx.append("    st.global.f32   [%rd6], %f2;\n\n");
        ptx.append("EXIT:\n");
        ptx.append("    ret;\n");
        ptx.append("}\n");

        return ptx.toString();
    }

    // ==================== Utility Methods ====================

    /**
     * Calculate grid size for the given element count.
     */
    public static int calculateGridSize(int elementCount, int blockSize) {
        return (elementCount + blockSize - 1) / blockSize;
    }

    /**
     * Calculate grid size using default block size.
     */
    public static int calculateGridSize(int elementCount) {
        return calculateGridSize(elementCount, ELEMENTWISE_BLOCK_SIZE);
    }

    /**
     * Get the function name for an operation.
     */
    public static String getFunctionName(String opName) {
        return switch (opName) {
            case "add" -> "add_f32";
            case "multiply" -> "multiply_f32";
            case "subtract" -> "subtract_f32";
            case "divide" -> "divide_f32";
            case "maximum" -> "maximum_f32";
            case "minimum" -> "minimum_f32";
            case "negate" -> "negate_f32";
            case "abs" -> "abs_f32";
            case "exp" -> "exp_f32";
            case "log" -> "log_f32";
            case "sqrt" -> "sqrt_f32";
            case "tanh" -> "tanh_f32";
            default -> throw new IllegalArgumentException("Unknown operation: " + opName);
        };
    }
}
