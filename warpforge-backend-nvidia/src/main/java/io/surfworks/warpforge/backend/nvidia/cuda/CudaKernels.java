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

    /**
     * Generate PTX for a binary elementwise float32 operation with multi-line instructions.
     *
     * @param opName Operation name for comments and entry point
     * @param ptxInstructions The PTX instructions (inputs in %f1 and %f2, output in %f3)
     * @param comment Description of the operation
     * @param salt Instrumentation level
     * @param floatRegs Number of float registers needed
     * @return PTX source code
     */
    private static String generateBinaryElementwiseF32(String opName, String ptxInstructions,
                                                        String comment, int salt, int floatRegs) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("//\n");
        ptx.append("// ").append(opName).append("_f32: Element-wise ").append(comment).append(" of float32 arrays\n");
        ptx.append("// Salt level: ").append(salt).append("\n");
        ptx.append("//\n");
        ptx.append(".version 7.0\n");
        ptx.append(".target sm_50\n");
        ptx.append(".address_size 64\n\n");

        ptx.append(".visible .entry ").append(opName).append("_f32(\n");
        ptx.append("    .param .u64 a_ptr,\n");
        ptx.append("    .param .u64 b_ptr,\n");
        ptx.append("    .param .u64 out_ptr,\n");
        ptx.append("    .param .u32 n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("    ,.param .u64 timing_ptr\n");
        }

        ptx.append(")\n{\n");
        ptx.append("    .reg .pred  %p<2>;\n");
        ptx.append("    .reg .f32   %f<").append(floatRegs).append(">;\n");
        ptx.append("    .reg .b32   %r<6>;\n");
        ptx.append("    .reg .b64   %rd<10>;\n");

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

        ptx.append("    ld.param.u64    %rd1, [a_ptr];\n");
        ptx.append("    ld.param.u64    %rd2, [b_ptr];\n");
        ptx.append("    ld.param.u64    %rd3, [out_ptr];\n\n");

        ptx.append("    cvt.s64.s32     %rd4, %r4;\n");
        ptx.append("    shl.b64         %rd5, %rd4, 2;\n\n");

        ptx.append("    add.s64         %rd6, %rd1, %rd5;\n");
        ptx.append("    add.s64         %rd7, %rd2, %rd5;\n");
        ptx.append("    add.s64         %rd8, %rd3, %rd5;\n\n");

        ptx.append("    ld.global.f32   %f1, [%rd6];\n");
        ptx.append("    ld.global.f32   %f2, [%rd7];\n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        // Core operation(s)
        ptx.append("    ").append(ptxInstructions).append("\n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t1, %globaltimer;\n");
            ptx.append("    sub.u64         %rd_delta, %rd_t1, %rd_t0;\n");
            ptx.append("    ld.param.u64    %rd9, [timing_ptr];\n");
            ptx.append("    atom.global.add.u64 [%rd9], %rd_delta;\n\n");
        }

        ptx.append("    st.global.f32   [%rd8], %f3;\n\n");
        ptx.append("EXIT:\n");
        ptx.append("    ret;\n");
        ptx.append("}\n");

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
     * Generate PTX for element-wise float32 tangent.
     * tan(x) = sin(x) / cos(x)
     */
    public static String generateTanF32(int salt) {
        String ptxOps = """
                    // tan(x) = sin(x) / cos(x)
                    sin.approx.f32  %f2, %f1;              // f2 = sin(x)
                    cos.approx.f32  %f3, %f1;              // f3 = cos(x)
                    div.approx.f32  %f2, %f2, %f3;         // f2 = sin(x) / cos(x)""";
        return generateUnaryElementwiseF32("tan", ptxOps, "tan(x)", salt, 4);
    }

    /**
     * Generate PTX for element-wise float32 logistic (sigmoid) function.
     * logistic(x) = 1 / (1 + exp(-x))
     */
    public static String generateLogisticF32(int salt) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        // Using base-2 exp: exp(x) = 2^(x * log2(e))
        // log2(e) = 1.4426950408889634 = 0x3FB8AA3B in IEEE 754
        String ptxOps = """
                    // logistic(x) = 1 / (1 + exp(-x))
                    neg.f32         %f2, %f1;              // f2 = -x
                    mul.f32         %f2, %f2, 0f3FB8AA3B;  // f2 = -x * log2(e)
                    ex2.approx.f32  %f2, %f2;              // f2 = exp(-x)
                    add.f32         %f2, %f2, 0f3F800000;  // f2 = 1 + exp(-x)
                    rcp.approx.f32  %f2, %f2;              // f2 = 1 / (1 + exp(-x))""";
        return generateUnaryElementwiseF32("logistic", ptxOps, "sigmoid(x)", salt);
    }

    /**
     * Generate PTX for element-wise float32 expm1.
     * expm1(x) = exp(x) - 1 (more accurate than exp(x)-1 for small x)
     */
    public static String generateExpm1F32(int salt) {
        // exp(x) - 1 using base-2 exp: exp(x) = 2^(x * log2(e))
        // log2(e) = 1.4426950408889634 = 0x3FB8AA3B
        String ptxOps = """
                    // expm1(x) = exp(x) - 1
                    mul.f32         %f2, %f1, 0f3FB8AA3B;  // f2 = x * log2(e)
                    ex2.approx.f32  %f2, %f2;              // f2 = exp(x)
                    sub.f32         %f2, %f2, 0f3F800000;  // f2 = exp(x) - 1""";
        return generateUnaryElementwiseF32("expm1", ptxOps, "exp(x)-1", salt);
    }

    /**
     * Generate PTX for element-wise float32 log1p.
     * log1p(x) = log(1 + x) (more accurate than log(1+x) for small x)
     */
    public static String generateLog1pF32(int salt) {
        // log(1+x) using base-2 log: log(x) = log2(x) / log2(e)
        // 1/log2(e) = ln(2) = 0.6931471805599453 = 0x3F317218
        String ptxOps = """
                    // log1p(x) = log(1 + x)
                    add.f32         %f2, %f1, 0f3F800000;  // f2 = 1 + x
                    lg2.approx.f32  %f2, %f2;              // f2 = log2(1 + x)
                    mul.f32         %f2, %f2, 0f3F317218;  // f2 = log(1 + x)""";
        return generateUnaryElementwiseF32("log1p", ptxOps, "log(1+x)", salt);
    }

    /**
     * Generate PTX for element-wise float32 cube root.
     * cbrt(x) = x^(1/3), handling negative values correctly.
     */
    public static String generateCbrtF32(int salt) {
        // cbrt(x) = sign(x) * |x|^(1/3)
        // |x|^(1/3) = exp(log(|x|) / 3) = 2^(log2(|x|) / 3)
        // 1/3 = 0.3333... = 0x3EAAAAAB
        // Need to handle sign separately since log requires positive input
        String ptxOps = """
                    // cbrt(x) = sign(x) * |x|^(1/3)
                    abs.f32         %f2, %f1;              // f2 = |x|
                    lg2.approx.f32  %f2, %f2;              // f2 = log2(|x|)
                    mul.f32         %f2, %f2, 0f3EAAAAAB;  // f2 = log2(|x|) / 3
                    ex2.approx.f32  %f2, %f2;              // f2 = |x|^(1/3)
                    copysign.f32    %f2, %f2, %f1;         // f2 = sign(x) * |x|^(1/3)""";
        return generateUnaryElementwiseF32("cbrt", ptxOps, "cbrt(x)", salt);
    }

    /**
     * Generate PTX for element-wise float32 is_finite check.
     * Returns 1.0 if x is finite (not NaN or Inf), 0.0 otherwise.
     */
    public static String generateIsFiniteF32(int salt) {
        // A value is finite if it's not NaN and not Inf
        // Use testp.finite to check
        String ptxOps = """
                    // is_finite(x): 1.0 if finite, 0.0 if NaN or Inf
                    testp.finite.f32 %p2, %f1;             // p2 = isfinite(x)
                    selp.f32        %f2, 0f3F800000, 0f00000000, %p2;  // f2 = p2 ? 1.0 : 0.0""";
        return generateUnaryElementwiseF32("is_finite", ptxOps, "isfinite(x)", salt, 3, 3);
    }

    /**
     * Generate PTX for element-wise float32 round to nearest even.
     * Rounds to nearest integer, with ties going to even (banker's rounding).
     */
    public static String generateRoundNearestEvenF32(int salt) {
        // cvt.rni rounds to nearest integer, ties to even
        return generateUnaryElementwiseF32("round_nearest_even", "cvt.rni.f32.f32 %f2, %f1;",
            "round_nearest_even(x)", salt);
    }

    /**
     * Generate PTX for element-wise float32 round to nearest, ties away from zero.
     * Rounds to nearest integer, with ties going away from zero.
     * E.g., 0.5 -> 1.0, -0.5 -> -1.0
     */
    public static String generateRoundNearestAfzF32(int salt) {
        // Round away from zero: add 0.5 with same sign as input, then truncate
        // sign(x) * floor(|x| + 0.5)
        // 0.5 = 0x3F000000
        String ptxOps = """
                    // round_afz(x) = sign(x) * floor(|x| + 0.5)
                    abs.f32         %f2, %f1;              // f2 = |x|
                    add.f32         %f2, %f2, 0f3F000000;  // f2 = |x| + 0.5
                    cvt.rmi.f32.f32 %f2, %f2;              // f2 = floor(|x| + 0.5)
                    copysign.f32    %f2, %f2, %f1;         // f2 = sign(x) * floor(|x| + 0.5)""";
        return generateUnaryElementwiseF32("round_nearest_afz", ptxOps, "round_afz(x)", salt);
    }

    /**
     * Generate PTX for element-wise float32 power.
     * power(a, b) = a^b = 2^(b * log2(a))
     * Note: Only works correctly for positive base values.
     */
    public static String generatePowerF32(int salt) {
        // a^b = 2^(b * log2(a))
        // For negative bases with integer exponents, we'd need special handling
        // This implementation works for positive bases
        String ptxOps = "lg2.approx.f32  %f3, %f1;\n" +           // f3 = log2(a)
                        "            mul.f32         %f3, %f3, %f2;\n" +  // f3 = b * log2(a)
                        "            ex2.approx.f32  %f3, %f3;";          // f3 = 2^(b * log2(a)) = a^b
        return generateBinaryElementwiseF32("power", ptxOps, "a^b", salt, 4);
    }

    /**
     * Generate PTX for element-wise float32 remainder.
     * remainder(a, b) = a - b * trunc(a / b)
     */
    public static String generateRemainderF32(int salt) {
        // a % b = a - b * trunc(a / b)
        String ptxOps = "div.approx.f32  %f3, %f1, %f2;\n" +      // f3 = a / b
                        "            cvt.rzi.f32.f32 %f3, %f3;\n" +       // f3 = trunc(a / b)
                        "            mul.f32         %f3, %f3, %f2;\n" +  // f3 = b * trunc(a / b)
                        "            sub.f32         %f3, %f1, %f3;";     // f3 = a - b * trunc(a / b)
        return generateBinaryElementwiseF32("remainder", ptxOps, "a%b", salt, 4);
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

    // ==================== Additional Unary Operations ====================

    /**
     * Generate PTX for element-wise logical NOT.
     * Returns 1.0 if input is 0.0, 0.0 otherwise.
     * This is logical NOT for float tensors representing boolean values.
     */
    public static String generateNotF32(int salt) {
        // Logical NOT: output = (input == 0) ? 1.0 : 0.0
        String ptxOps = """
                    // logical not: 1.0 if x == 0, else 0.0
                    setp.eq.f32     %p2, %f1, 0f00000000;  // p2 = (x == 0)
                    selp.f32        %f2, 0f3F800000, 0f00000000, %p2;  // f2 = p2 ? 1.0 : 0.0""";
        return generateUnaryElementwiseF32("not", ptxOps, "logical_not(x)", salt, 3, 3);
    }

    /**
     * Generate PTX for element-wise atan2(y, x).
     * Returns the angle in radians in [-π, π].
     *
     * <p>Implementation uses polynomial approximation for atan
     * and quadrant correction for atan2.
     */
    public static String generateAtan2F32(int salt) {
        // atan2(y, x) implementation using atan approximation
        // For the approximation, we use: atan(z) ≈ z * (1 - 0.28125*z^2) for |z| <= 1
        // For |z| > 1, use: atan(z) = sign(z) * π/2 - atan(1/z)
        //
        // Then adjust for quadrants:
        // x > 0: atan(y/x)
        // x < 0, y >= 0: atan(y/x) + π
        // x < 0, y < 0: atan(y/x) - π
        // x = 0, y > 0: π/2
        // x = 0, y < 0: -π/2
        // x = 0, y = 0: 0
        //
        // This is complex in PTX. Using a simplified approach with div and atan approximation.
        // π/2 = 1.5707963 = 0x3FC90FDB
        // π = 3.1415927 = 0x40490FDB
        // -0.28125 = 0xBE900000

        String ptxOps = """
lg2.approx.f32  %f4, %f2;              // temp for checking if x is positive
            abs.f32         %f5, %f1;              // |y|
            abs.f32         %f6, %f2;              // |x|
            // Compute |y|/|x| or |x|/|y| depending on which is larger
            setp.gt.f32     %p2, %f5, %f6;         // p2 = |y| > |x|
            @%p2 div.approx.f32 %f3, %f6, %f5;     // if |y| > |x|: z = |x|/|y|
            @!%p2 div.approx.f32 %f3, %f5, %f6;    // else: z = |y|/|x|
            // Compute atan(z) ≈ z - z^3/3 (simplified)
            mul.f32         %f4, %f3, %f3;         // z^2
            mul.f32         %f4, %f4, 0fBE2AAAAB;  // -z^2/3 (approx -0.333)
            add.f32         %f4, %f4, 0f3F800000;  // 1 - z^2/3
            mul.f32         %f3, %f3, %f4;         // z * (1 - z^2/3) ≈ atan(z)
            // If |y| > |x|, result = π/2 - atan(z)
            @%p2 sub.f32    %f3, 0f3FC90FDB, %f3;  // π/2 - atan(z)
            // Apply signs based on quadrant
            // If x < 0, add/sub π
            setp.lt.f32     %p3, %f2, 0f00000000;  // p3 = x < 0
            setp.ge.f32     %p4, %f1, 0f00000000;  // p4 = y >= 0
            @%p3 @%p4 add.f32 %f3, %f3, 0f40490FDB; // if x<0 and y>=0: add π
            @%p3 @!%p4 sub.f32 %f3, %f3, 0f40490FDB; // if x<0 and y<0: sub π
            // Apply sign of y
            copysign.f32    %f3, %f3, %f1;""";
        return generateBinaryElementwiseF32("atan2", ptxOps, "atan2(y,x)", salt, 7, 5);
    }

    // ==================== Comparison and Selection Operations ====================

    /**
     * Generate PTX for element-wise float32 comparison.
     * Returns 1.0f for true, 0.0f for false.
     *
     * @param direction Comparison direction (EQ, NE, LT, LE, GT, GE)
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateCompareF32(String direction, int salt) {
        String setpOp = switch (direction) {
            case "EQ" -> "setp.eq.f32";
            case "NE" -> "setp.ne.f32";
            case "LT" -> "setp.lt.f32";
            case "LE" -> "setp.le.f32";
            case "GT" -> "setp.gt.f32";
            case "GE" -> "setp.ge.f32";
            default -> throw new IllegalArgumentException("Unknown comparison direction: " + direction);
        };

        String ptxOps = setpOp + "    %p2, %f1, %f2;\n" +
                        "            selp.f32        %f3, 0f3F800000, 0f00000000, %p2;";
        return generateBinaryElementwiseF32("compare_" + direction.toLowerCase(), ptxOps,
            direction.toLowerCase() + " comparison", salt, 4, 3);
    }

    /**
     * Generate PTX for element-wise select operation.
     * result = pred ? onTrue : onFalse (where pred is 1.0 for true, 0.0 for false)
     */
    public static String generateSelectF32(int salt) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("//\n");
        ptx.append("// select_f32: Element-wise ternary selection\n");
        ptx.append("// Salt level: ").append(salt).append("\n");
        ptx.append("//\n");
        ptx.append(".version 7.0\n");
        ptx.append(".target sm_50\n");
        ptx.append(".address_size 64\n\n");

        ptx.append(".visible .entry select_f32(\n");
        ptx.append("    .param .u64 pred_ptr,\n");
        ptx.append("    .param .u64 on_true_ptr,\n");
        ptx.append("    .param .u64 on_false_ptr,\n");
        ptx.append("    .param .u64 out_ptr,\n");
        ptx.append("    .param .u32 n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("    ,.param .u64 timing_ptr\n");
        }

        ptx.append(")\n{\n");
        ptx.append("    .reg .pred  %p<3>;\n");
        ptx.append("    .reg .f32   %f<5>;\n");
        ptx.append("    .reg .b32   %r<6>;\n");
        ptx.append("    .reg .b64   %rd<12>;\n");

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

        ptx.append("    ld.param.u64    %rd1, [pred_ptr];\n");
        ptx.append("    ld.param.u64    %rd2, [on_true_ptr];\n");
        ptx.append("    ld.param.u64    %rd3, [on_false_ptr];\n");
        ptx.append("    ld.param.u64    %rd4, [out_ptr];\n\n");

        ptx.append("    cvt.s64.s32     %rd5, %r4;\n");
        ptx.append("    shl.b64         %rd6, %rd5, 2;\n\n");

        ptx.append("    add.s64         %rd7, %rd1, %rd6;\n");
        ptx.append("    add.s64         %rd8, %rd2, %rd6;\n");
        ptx.append("    add.s64         %rd9, %rd3, %rd6;\n");
        ptx.append("    add.s64         %rd10, %rd4, %rd6;\n\n");

        ptx.append("    ld.global.f32   %f1, [%rd7];   // pred\n");
        ptx.append("    ld.global.f32   %f2, [%rd8];   // on_true\n");
        ptx.append("    ld.global.f32   %f3, [%rd9];   // on_false\n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        // Convert pred float to predicate (pred != 0.0)
        ptx.append("    setp.ne.f32     %p2, %f1, 0f00000000;  // pred != 0\n");
        ptx.append("    selp.f32        %f4, %f2, %f3, %p2;    // result = pred ? on_true : on_false\n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t1, %globaltimer;\n");
            ptx.append("    sub.u64         %rd_delta, %rd_t1, %rd_t0;\n");
            ptx.append("    ld.param.u64    %rd11, [timing_ptr];\n");
            ptx.append("    atom.global.add.u64 [%rd11], %rd_delta;\n\n");
        }

        ptx.append("    st.global.f32   [%rd10], %f4;\n\n");
        ptx.append("EXIT:\n");
        ptx.append("    ret;\n");
        ptx.append("}\n");

        return ptx.toString();
    }

    /**
     * Generate PTX for element-wise clamp operation.
     * result = max(min_val, min(operand, max_val))
     */
    public static String generateClampF32(int salt) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("//\n");
        ptx.append("// clamp_f32: Element-wise clamp\n");
        ptx.append("// Salt level: ").append(salt).append("\n");
        ptx.append("//\n");
        ptx.append(".version 7.0\n");
        ptx.append(".target sm_50\n");
        ptx.append(".address_size 64\n\n");

        ptx.append(".visible .entry clamp_f32(\n");
        ptx.append("    .param .u64 min_ptr,\n");
        ptx.append("    .param .u64 operand_ptr,\n");
        ptx.append("    .param .u64 max_ptr,\n");
        ptx.append("    .param .u64 out_ptr,\n");
        ptx.append("    .param .u32 n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("    ,.param .u64 timing_ptr\n");
        }

        ptx.append(")\n{\n");
        ptx.append("    .reg .pred  %p<2>;\n");
        ptx.append("    .reg .f32   %f<5>;\n");
        ptx.append("    .reg .b32   %r<6>;\n");
        ptx.append("    .reg .b64   %rd<12>;\n");

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

        ptx.append("    ld.param.u64    %rd1, [min_ptr];\n");
        ptx.append("    ld.param.u64    %rd2, [operand_ptr];\n");
        ptx.append("    ld.param.u64    %rd3, [max_ptr];\n");
        ptx.append("    ld.param.u64    %rd4, [out_ptr];\n\n");

        ptx.append("    cvt.s64.s32     %rd5, %r4;\n");
        ptx.append("    shl.b64         %rd6, %rd5, 2;\n\n");

        ptx.append("    add.s64         %rd7, %rd1, %rd6;\n");
        ptx.append("    add.s64         %rd8, %rd2, %rd6;\n");
        ptx.append("    add.s64         %rd9, %rd3, %rd6;\n");
        ptx.append("    add.s64         %rd10, %rd4, %rd6;\n\n");

        ptx.append("    ld.global.f32   %f1, [%rd7];   // min\n");
        ptx.append("    ld.global.f32   %f2, [%rd8];   // operand\n");
        ptx.append("    ld.global.f32   %f3, [%rd9];   // max\n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        // clamp = max(min_val, min(operand, max_val))
        ptx.append("    min.f32         %f4, %f2, %f3;  // f4 = min(operand, max)\n");
        ptx.append("    max.f32         %f4, %f4, %f1;  // f4 = max(f4, min)\n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t1, %globaltimer;\n");
            ptx.append("    sub.u64         %rd_delta, %rd_t1, %rd_t0;\n");
            ptx.append("    ld.param.u64    %rd11, [timing_ptr];\n");
            ptx.append("    atom.global.add.u64 [%rd11], %rd_delta;\n\n");
        }

        ptx.append("    st.global.f32   [%rd10], %f4;\n\n");
        ptx.append("EXIT:\n");
        ptx.append("    ret;\n");
        ptx.append("}\n");

        return ptx.toString();
    }

    /**
     * Generate PTX for a binary elementwise float32 operation with custom predicate count.
     */
    private static String generateBinaryElementwiseF32(String opName, String ptxInstructions,
                                                        String comment, int salt, int floatRegs,
                                                        int predicateRegs) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("//\n");
        ptx.append("// ").append(opName).append("_f32: Element-wise ").append(comment).append(" of float32 arrays\n");
        ptx.append("// Salt level: ").append(salt).append("\n");
        ptx.append("//\n");
        ptx.append(".version 7.0\n");
        ptx.append(".target sm_50\n");
        ptx.append(".address_size 64\n\n");

        ptx.append(".visible .entry ").append(opName).append("_f32(\n");
        ptx.append("    .param .u64 a_ptr,\n");
        ptx.append("    .param .u64 b_ptr,\n");
        ptx.append("    .param .u64 out_ptr,\n");
        ptx.append("    .param .u32 n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("    ,.param .u64 timing_ptr\n");
        }

        ptx.append(")\n{\n");
        ptx.append("    .reg .pred  %p<").append(predicateRegs).append(">;\n");
        ptx.append("    .reg .f32   %f<").append(floatRegs).append(">;\n");
        ptx.append("    .reg .b32   %r<6>;\n");
        ptx.append("    .reg .b64   %rd<10>;\n");

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

        ptx.append("    ld.param.u64    %rd1, [a_ptr];\n");
        ptx.append("    ld.param.u64    %rd2, [b_ptr];\n");
        ptx.append("    ld.param.u64    %rd3, [out_ptr];\n\n");

        ptx.append("    cvt.s64.s32     %rd4, %r4;\n");
        ptx.append("    shl.b64         %rd5, %rd4, 2;\n\n");

        ptx.append("    add.s64         %rd6, %rd1, %rd5;\n");
        ptx.append("    add.s64         %rd7, %rd2, %rd5;\n");
        ptx.append("    add.s64         %rd8, %rd3, %rd5;\n\n");

        ptx.append("    ld.global.f32   %f1, [%rd6];\n");
        ptx.append("    ld.global.f32   %f2, [%rd7];\n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        // Core operation(s)
        ptx.append("    ").append(ptxInstructions).append("\n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t1, %globaltimer;\n");
            ptx.append("    sub.u64         %rd_delta, %rd_t1, %rd_t0;\n");
            ptx.append("    ld.param.u64    %rd9, [timing_ptr];\n");
            ptx.append("    atom.global.add.u64 [%rd9], %rd_delta;\n\n");
        }

        ptx.append("    st.global.f32   [%rd8], %f3;\n\n");
        ptx.append("EXIT:\n");
        ptx.append("    ret;\n");
        ptx.append("}\n");

        return ptx.toString();
    }

    // ==================== Integer Bitwise Operations ====================

    /**
     * Generate PTX for element-wise int32 bitwise AND.
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateAndI32(int salt) {
        return generateBinaryElementwiseI32("and", "and.b32", "a[i] & b[i]", salt);
    }

    /**
     * Generate PTX for element-wise int32 bitwise OR.
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateOrI32(int salt) {
        return generateBinaryElementwiseI32("or", "or.b32", "a[i] | b[i]", salt);
    }

    /**
     * Generate PTX for element-wise int32 bitwise XOR.
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateXorI32(int salt) {
        return generateBinaryElementwiseI32("xor", "xor.b32", "a[i] ^ b[i]", salt);
    }

    /**
     * Generate PTX for a binary elementwise int32 operation.
     *
     * @param opName Operation name for comments and entry point
     * @param ptxInstruction The PTX instruction to use
     * @param comment Description of the operation
     * @param salt Instrumentation level
     * @return PTX source code
     */
    private static String generateBinaryElementwiseI32(String opName, String ptxInstruction,
                                                        String comment, int salt) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("""
            //
            // %s_i32: Element-wise %s of int32 arrays
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            .visible .entry %s_i32(
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
                .reg .s32   %i<4>;
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

                ld.global.s32   %i1, [%rd6];
                ld.global.s32   %i2, [%rd7];

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        // Core operation
        ptx.append("    ").append(ptxInstruction).append("         %i3, %i1, %i2;\n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("""
                    mov.u64         %rd_t1, %globaltimer;
                    sub.u64         %rd_delta, %rd_t1, %rd_t0;
                    ld.param.u64    %rd9, [timing_ptr];
                    atom.global.add.u64 [%rd9], %rd_delta;

            """);
        }

        ptx.append("""
                st.global.s32   [%rd8], %i3;

            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    // ==================== Integer Shift Operations ====================

    /**
     * Generate PTX for element-wise int32 left shift.
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateShiftLeftI32(int salt) {
        return generateBinaryElementwiseI32("shift_left", "shl.b32", "a[i] << b[i]", salt);
    }

    /**
     * Generate PTX for element-wise int32 arithmetic right shift (sign-extending).
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateShiftRightArithmeticI32(int salt) {
        return generateBinaryElementwiseI32("shift_right_arithmetic", "shr.s32", "a[i] >> b[i] (arithmetic)", salt);
    }

    /**
     * Generate PTX for element-wise int32 logical right shift (zero-extending).
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateShiftRightLogicalI32(int salt) {
        return generateBinaryElementwiseI32("shift_right_logical", "shr.u32", "a[i] >>> b[i] (logical)", salt);
    }

    // ==================== Integer Unary Operations ====================

    /**
     * Generate PTX for element-wise int32 population count (count set bits).
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generatePopcntI32(int salt) {
        return generateUnaryElementwiseI32("popcnt", "popc.b32", "popcount(a[i])", salt);
    }

    /**
     * Generate PTX for element-wise int32 count leading zeros.
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateClzI32(int salt) {
        return generateUnaryElementwiseI32("clz", "clz.b32", "clz(a[i])", salt);
    }

    /**
     * Generate PTX for a unary elementwise int32 operation.
     *
     * @param opName Operation name for comments and entry point
     * @param ptxInstruction The PTX instruction to use
     * @param comment Description of the operation
     * @param salt Instrumentation level
     * @return PTX source code
     */
    private static String generateUnaryElementwiseI32(String opName, String ptxInstruction,
                                                       String comment, int salt) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("""
            //
            // %s_i32: Element-wise %s of int32 array
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            .visible .entry %s_i32(
                .param .u64 in_ptr,
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
                .reg .s32   %i<3>;
                .reg .b32   %r<6>;
                .reg .b64   %rd<8>;
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

                ld.param.u64    %rd1, [in_ptr];
                ld.param.u64    %rd2, [out_ptr];

                cvt.s64.s32     %rd3, %r4;
                shl.b64         %rd4, %rd3, 2;

                add.s64         %rd5, %rd1, %rd4;
                add.s64         %rd6, %rd2, %rd4;

                ld.global.s32   %i1, [%rd5];

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        // Core operation
        ptx.append("    ").append(ptxInstruction).append("         %i2, %i1;\n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("""
                    mov.u64         %rd_t1, %globaltimer;
                    sub.u64         %rd_delta, %rd_t1, %rd_t0;
                    ld.param.u64    %rd7, [timing_ptr];
                    atom.global.add.u64 [%rd7], %rd_delta;

            """);
        }

        ptx.append("""
                st.global.s32   [%rd6], %i2;

            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    // ==================== Matrix Operations ====================

    /** Block size for matrix multiplication */
    public static final int DOT_BLOCK_SIZE = 16;

    /**
     * Generate PTX for matrix multiplication (dot product).
     * C[M,N] = A[M,K] * B[K,N]
     *
     * <p>This is a naive implementation where each thread computes one output element.
     * For production, a tiled implementation with shared memory would be faster.
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateDotF32(int salt) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("""
            //
            // dot_f32: Matrix multiplication C = A * B
            // A[M,K] * B[K,N] = C[M,N]
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            .visible .entry dot_f32(
                .param .u64 a_ptr,
                .param .u64 b_ptr,
                .param .u64 c_ptr,
                .param .u32 M,
                .param .u32 N,
                .param .u32 K
            """.formatted(salt));

        if (salt >= SALT_TIMING) {
            ptx.append("    ,.param .u64 timing_ptr\n");
        }

        ptx.append("""
            )
            {
                .reg .pred  %p<2>;
                .reg .f32   %f<5>;
                .reg .b32   %r<16>;
                .reg .b64   %rd<16>;
            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    .reg .b64   %rd_t0, %rd_t1, %rd_delta;\n");
        }

        ptx.append("""

                // Calculate row (i) and col (j) from thread indices
                // Using 2D grid: blockIdx.x * blockDim.x + threadIdx.x = col (j)
                //                blockIdx.y * blockDim.y + threadIdx.y = row (i)
                mov.u32         %r1, %ctaid.x;      // blockIdx.x
                mov.u32         %r2, %ntid.x;       // blockDim.x
                mov.u32         %r3, %tid.x;        // threadIdx.x
                mad.lo.s32      %r4, %r1, %r2, %r3; // j = col

                mov.u32         %r5, %ctaid.y;      // blockIdx.y
                mov.u32         %r6, %ntid.y;       // blockDim.y
                mov.u32         %r7, %tid.y;        // threadIdx.y
                mad.lo.s32      %r8, %r5, %r6, %r7; // i = row

                // Load dimensions
                ld.param.u32    %r9, [M];
                ld.param.u32    %r10, [N];
                ld.param.u32    %r11, [K];

                // Bounds check: if (i >= M || j >= N) return
                setp.ge.s32     %p1, %r8, %r9;
                @%p1 bra        EXIT;
                setp.ge.s32     %p1, %r4, %r10;
                @%p1 bra        EXIT;

                // Load base pointers
                ld.param.u64    %rd1, [a_ptr];
                ld.param.u64    %rd2, [b_ptr];
                ld.param.u64    %rd3, [c_ptr];

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        ptx.append("""
                // Initialize accumulator
                mov.f32         %f1, 0f00000000;    // sum = 0.0

                // Loop: for k = 0; k < K; k++
                mov.u32         %r12, 0;            // k = 0
            LOOP:
                setp.ge.s32     %p1, %r12, %r11;
                @%p1 bra        LOOP_END;

                // Load A[i, k] = A[i * K + k]
                mul.lo.s32      %r13, %r8, %r11;    // i * K
                add.s32         %r13, %r13, %r12;   // i * K + k
                cvt.s64.s32     %rd4, %r13;
                shl.b64         %rd4, %rd4, 2;      // * 4 bytes
                add.s64         %rd5, %rd1, %rd4;
                ld.global.f32   %f2, [%rd5];

                // Load B[k, j] = B[k * N + j]
                mul.lo.s32      %r14, %r12, %r10;   // k * N
                add.s32         %r14, %r14, %r4;    // k * N + j
                cvt.s64.s32     %rd6, %r14;
                shl.b64         %rd6, %rd6, 2;      // * 4 bytes
                add.s64         %rd7, %rd2, %rd6;
                ld.global.f32   %f3, [%rd7];

                // sum += A[i,k] * B[k,j]
                fma.rn.f32      %f1, %f2, %f3, %f1;

                // k++
                add.s32         %r12, %r12, 1;
                bra             LOOP;

            LOOP_END:

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("""
                    mov.u64         %rd_t1, %globaltimer;
                    sub.u64         %rd_delta, %rd_t1, %rd_t0;
                    ld.param.u64    %rd15, [timing_ptr];
                    atom.global.add.u64 [%rd15], %rd_delta;

            """);
        }

        ptx.append("""
                // Store C[i, j] = C[i * N + j]
                mul.lo.s32      %r15, %r8, %r10;    // i * N
                add.s32         %r15, %r15, %r4;    // i * N + j
                cvt.s64.s32     %rd8, %r15;
                shl.b64         %rd8, %rd8, 2;      // * 4 bytes
                add.s64         %rd9, %rd3, %rd8;
                st.global.f32   [%rd9], %f1;

            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    // ==================== Reduction Operations ====================

    /** Block size for reduction operations */
    public static final int REDUCE_BLOCK_SIZE = 256;

    /**
     * Generate PTX for parallel reduction with specified operation.
     *
     * <p>This implements a two-phase reduction:
     * 1. Each block reduces its portion to a single value using shared memory
     * 2. Final reduction across blocks (handled by launching with single block for small inputs
     *    or by the kernel caller for large inputs)
     *
     * @param reducer The reduction operation: "add", "max", "min", "mul"
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateReduceF32(String reducer, int salt) {
        String ptxOp = switch (reducer) {
            case "add" -> "add.f32";
            case "max" -> "max.f32";
            case "min" -> "min.f32";
            case "mul" -> "mul.f32";
            default -> throw new IllegalArgumentException("Unknown reducer: " + reducer);
        };

        StringBuilder ptx = new StringBuilder();

        ptx.append("""
            //
            // reduce_%s_f32: Parallel reduction with %s operation
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            .visible .entry reduce_%s_f32(
                .param .u64 in_ptr,
                .param .u64 out_ptr,
                .param .u32 n,
                .param .f32 init_value
            """.formatted(reducer, reducer, salt, reducer));

        if (salt >= SALT_TIMING) {
            ptx.append("    ,.param .u64 timing_ptr\n");
        }

        ptx.append("""
            )
            {
                // Shared memory for block reduction (256 floats = 1KB)
                .shared .align 4 .f32 sdata[256];

                .reg .pred  %p<4>;
                .reg .f32   %f<8>;
                .reg .b32   %r<16>;
                .reg .b64   %rd<12>;
            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    .reg .b64   %rd_t0, %rd_t1, %rd_delta;\n");
        }

        ptx.append("""

                // Global thread ID
                mov.u32         %r1, %ctaid.x;
                mov.u32         %r2, %ntid.x;
                mov.u32         %r3, %tid.x;
                mad.lo.s32      %r4, %r1, %r2, %r3;  // global_id

                // Load n and init_value
                ld.param.u32    %r5, [n];
                ld.param.f32    %f1, [init_value];   // accumulator = init_value

                // Load input pointer
                ld.param.u64    %rd1, [in_ptr];

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        ptx.append("""
                // Phase 1: Each thread loads and reduces multiple elements (grid-stride loop)
                // This handles inputs larger than grid size
                mov.u32         %%r6, %%nctaid.x;     // gridDim.x
                mul.lo.s32      %%r7, %%r6, %%r2;      // stride = gridDim.x * blockDim.x

            LOAD_LOOP:
                setp.ge.s32     %%p1, %%r4, %%r5;      // if (idx >= n)
                @%%p1 bra        LOAD_DONE;

                // Load input[idx]
                cvt.s64.s32     %%rd2, %%r4;
                shl.b64         %%rd3, %%rd2, 2;
                add.s64         %%rd4, %%rd1, %%rd3;
                ld.global.f32   %%f2, [%%rd4];

                // accumulator = op(accumulator, input[idx])
                %s         %%f1, %%f1, %%f2;

                // idx += stride
                add.s32         %%r4, %%r4, %%r7;
                bra             LOAD_LOOP;

            LOAD_DONE:
                // Store to shared memory
                mul.lo.s32      %%r8, %%r3, 4;        // tid * 4
                mov.u32         %%r9, sdata;
                add.s32         %%r9, %%r9, %%r8;
                st.shared.f32   [%%r9], %%f1;

                // Synchronize threads in block
                bar.sync        0;

                // Phase 2: Tree reduction in shared memory
                mov.u32         %%r10, 128;          // s = blockDim.x / 2

            REDUCE_LOOP:
                setp.eq.s32     %%p2, %%r10, 0;
                @%%p2 bra        REDUCE_DONE;

                setp.ge.s32     %%p3, %%r3, %%r10;     // if (tid >= s)
                @%%p3 bra        REDUCE_SKIP;

                // Load sdata[tid + s]
                add.s32         %%r11, %%r3, %%r10;
                mul.lo.s32      %%r12, %%r11, 4;
                mov.u32         %%r13, sdata;
                add.s32         %%r13, %%r13, %%r12;
                ld.shared.f32   %%f3, [%%r13];

                // Load sdata[tid]
                ld.shared.f32   %%f4, [%%r9];

                // sdata[tid] = op(sdata[tid], sdata[tid + s])
                %s         %%f4, %%f4, %%f3;
                st.shared.f32   [%%r9], %%f4;

            REDUCE_SKIP:
                bar.sync        0;
                shr.u32         %%r10, %%r10, 1;      // s /= 2
                bra             REDUCE_LOOP;

            REDUCE_DONE:

            """.formatted(ptxOp, ptxOp));

        if (salt >= SALT_TIMING) {
            ptx.append("""
                    mov.u64         %rd_t1, %globaltimer;
                    sub.u64         %rd_delta, %rd_t1, %rd_t0;
                    ld.param.u64    %rd11, [timing_ptr];
                    atom.global.add.u64 [%rd11], %rd_delta;

            """);
        }

        ptx.append("""
                // Thread 0 writes block result to output
                setp.ne.s32     %p1, %r3, 0;
                @%p1 bra        EXIT;

                ld.param.u64    %rd5, [out_ptr];
                // Write to out[blockIdx.x]
                cvt.s64.s32     %rd6, %r1;
                shl.b64         %rd7, %rd6, 2;
                add.s64         %rd8, %rd5, %rd7;
                ld.shared.f32   %f5, [sdata];
                st.global.f32   [%rd8], %f5;

            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    /**
     * Generate PTX for reduce with add operation.
     */
    public static String generateReduceAddF32(int salt) {
        return generateReduceF32("add", salt);
    }

    /**
     * Generate PTX for reduce with max operation.
     */
    public static String generateReduceMaxF32(int salt) {
        return generateReduceF32("max", salt);
    }

    /**
     * Generate PTX for reduce with min operation.
     */
    public static String generateReduceMinF32(int salt) {
        return generateReduceF32("min", salt);
    }

    /**
     * Generate PTX for reduce with mul operation.
     */
    public static String generateReduceMulF32(int salt) {
        return generateReduceF32("mul", salt);
    }

    // ==================== Shape Manipulation Operations ====================

    /**
     * Generate PTX for reshape operation.
     *
     * <p>Reshape is essentially a memory copy since data layout doesn't change,
     * only the interpretation of dimensions. This kernel performs a simple
     * element-wise copy from input to output.
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateReshapeF32(int salt) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("""
            //
            // reshape_f32: Copy data with new shape interpretation
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            .visible .entry reshape_f32(
                .param .u64 in_ptr,
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
                .reg .f32   %f<2>;
                .reg .b32   %r<8>;
                .reg .b64   %rd<8>;
            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    .reg .b64   %rd_t0, %rd_t1, %rd_delta;\n");
        }

        ptx.append("""

                // Calculate global thread index
                mov.u32         %r1, %ctaid.x;
                mov.u32         %r2, %ntid.x;
                mov.u32         %r3, %tid.x;
                mad.lo.s32      %r4, %r1, %r2, %r3;

                // Bounds check
                ld.param.u32    %r5, [n];
                setp.ge.s32     %p1, %r4, %r5;
                @%p1 bra        EXIT;

                // Load pointers
                ld.param.u64    %rd1, [in_ptr];
                ld.param.u64    %rd2, [out_ptr];

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        ptx.append("""
                // Calculate offset
                cvt.s64.s32     %rd3, %r4;
                shl.b64         %rd3, %rd3, 2;      // * 4 bytes

                // Load from input
                add.s64         %rd4, %rd1, %rd3;
                ld.global.f32   %f1, [%rd4];

                // Store to output
                add.s64         %rd5, %rd2, %rd3;
                st.global.f32   [%rd5], %f1;

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("""
                    mov.u64         %rd_t1, %globaltimer;
                    sub.u64         %rd_delta, %rd_t1, %rd_t0;
                    ld.param.u64    %rd7, [timing_ptr];
                    atom.global.add.u64 [%rd7], %rd_delta;

            """);
        }

        ptx.append("""
            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    /**
     * Generate PTX for 2D transpose operation.
     *
     * <p>Transposes a 2D matrix: output[j,i] = input[i,j].
     * Uses a simple element-wise kernel. For better performance on large
     * matrices, a tiled shared-memory version would be preferred.
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateTranspose2DF32(int salt) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("""
            //
            // transpose_2d_f32: Transpose 2D matrix
            // output[j,i] = input[i,j]
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            .visible .entry transpose_2d_f32(
                .param .u64 in_ptr,
                .param .u64 out_ptr,
                .param .u32 rows,
                .param .u32 cols
            """.formatted(salt));

        if (salt >= SALT_TIMING) {
            ptx.append("    ,.param .u64 timing_ptr\n");
        }

        ptx.append("""
            )
            {
                .reg .pred  %p<2>;
                .reg .f32   %f<2>;
                .reg .b32   %r<16>;
                .reg .b64   %rd<12>;
            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    .reg .b64   %rd_t0, %rd_t1, %rd_delta;\n");
        }

        ptx.append("""

                // Calculate row (i) and col (j) from 2D grid
                mov.u32         %r1, %ctaid.x;
                mov.u32         %r2, %ntid.x;
                mov.u32         %r3, %tid.x;
                mad.lo.s32      %r4, %r1, %r2, %r3; // j = col

                mov.u32         %r5, %ctaid.y;
                mov.u32         %r6, %ntid.y;
                mov.u32         %r7, %tid.y;
                mad.lo.s32      %r8, %r5, %r6, %r7; // i = row

                // Load dimensions
                ld.param.u32    %r9, [rows];
                ld.param.u32    %r10, [cols];

                // Bounds check
                setp.ge.s32     %p1, %r8, %r9;
                @%p1 bra        EXIT;
                setp.ge.s32     %p1, %r4, %r10;
                @%p1 bra        EXIT;

                // Load pointers
                ld.param.u64    %rd1, [in_ptr];
                ld.param.u64    %rd2, [out_ptr];

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        ptx.append("""
                // Input index: i * cols + j
                mul.lo.s32      %r11, %r8, %r10;
                add.s32         %r11, %r11, %r4;
                cvt.s64.s32     %rd3, %r11;
                shl.b64         %rd3, %rd3, 2;
                add.s64         %rd4, %rd1, %rd3;
                ld.global.f32   %f1, [%rd4];

                // Output index: j * rows + i (transposed)
                mul.lo.s32      %r12, %r4, %r9;
                add.s32         %r12, %r12, %r8;
                cvt.s64.s32     %rd5, %r12;
                shl.b64         %rd5, %rd5, 2;
                add.s64         %rd6, %rd2, %rd5;
                st.global.f32   [%rd6], %f1;

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("""
                    mov.u64         %rd_t1, %globaltimer;
                    sub.u64         %rd_delta, %rd_t1, %rd_t0;
                    ld.param.u64    %rd11, [timing_ptr];
                    atom.global.add.u64 [%rd11], %rd_delta;

            """);
        }

        ptx.append("""
            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    /**
     * Generate PTX for broadcast operation.
     *
     * <p>Broadcasts a tensor to a larger shape. Each output element reads
     * from the corresponding input element based on the broadcast mapping.
     *
     * <p>This kernel handles the common case of broadcasting a 1D tensor
     * to 2D by replicating along rows or columns.
     *
     * @param salt Instrumentation level
     * @return PTX source code for row broadcast (input[j] -> output[i,j])
     */
    public static String generateBroadcast1Dto2DRowF32(int salt) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("""
            //
            // broadcast_1d_to_2d_row_f32: Broadcast 1D to 2D along rows
            // input[j] -> output[i,j] for all i
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            .visible .entry broadcast_1d_to_2d_row_f32(
                .param .u64 in_ptr,
                .param .u64 out_ptr,
                .param .u32 rows,
                .param .u32 cols
            """.formatted(salt));

        if (salt >= SALT_TIMING) {
            ptx.append("    ,.param .u64 timing_ptr\n");
        }

        ptx.append("""
            )
            {
                .reg .pred  %p<2>;
                .reg .f32   %f<2>;
                .reg .b32   %r<16>;
                .reg .b64   %rd<12>;
            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    .reg .b64   %rd_t0, %rd_t1, %rd_delta;\n");
        }

        ptx.append("""

                // Calculate row (i) and col (j) from 2D grid
                mov.u32         %r1, %ctaid.x;
                mov.u32         %r2, %ntid.x;
                mov.u32         %r3, %tid.x;
                mad.lo.s32      %r4, %r1, %r2, %r3; // j = col

                mov.u32         %r5, %ctaid.y;
                mov.u32         %r6, %ntid.y;
                mov.u32         %r7, %tid.y;
                mad.lo.s32      %r8, %r5, %r6, %r7; // i = row

                // Load dimensions
                ld.param.u32    %r9, [rows];
                ld.param.u32    %r10, [cols];

                // Bounds check
                setp.ge.s32     %p1, %r8, %r9;
                @%p1 bra        EXIT;
                setp.ge.s32     %p1, %r4, %r10;
                @%p1 bra        EXIT;

                // Load pointers
                ld.param.u64    %rd1, [in_ptr];
                ld.param.u64    %rd2, [out_ptr];

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        ptx.append("""
                // Input index: just j (broadcast along rows)
                cvt.s64.s32     %rd3, %r4;
                shl.b64         %rd3, %rd3, 2;
                add.s64         %rd4, %rd1, %rd3;
                ld.global.f32   %f1, [%rd4];

                // Output index: i * cols + j
                mul.lo.s32      %r11, %r8, %r10;
                add.s32         %r11, %r11, %r4;
                cvt.s64.s32     %rd5, %r11;
                shl.b64         %rd5, %rd5, 2;
                add.s64         %rd6, %rd2, %rd5;
                st.global.f32   [%rd6], %f1;

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("""
                    mov.u64         %rd_t1, %globaltimer;
                    sub.u64         %rd_delta, %rd_t1, %rd_t0;
                    ld.param.u64    %rd11, [timing_ptr];
                    atom.global.add.u64 [%rd11], %rd_delta;

            """);
        }

        ptx.append("""
            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    /**
     * Generate PTX for broadcast operation along columns.
     *
     * @param salt Instrumentation level
     * @return PTX source code for column broadcast (input[i] -> output[i,j])
     */
    public static String generateBroadcast1Dto2DColF32(int salt) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("""
            //
            // broadcast_1d_to_2d_col_f32: Broadcast 1D to 2D along columns
            // input[i] -> output[i,j] for all j
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            .visible .entry broadcast_1d_to_2d_col_f32(
                .param .u64 in_ptr,
                .param .u64 out_ptr,
                .param .u32 rows,
                .param .u32 cols
            """.formatted(salt));

        if (salt >= SALT_TIMING) {
            ptx.append("    ,.param .u64 timing_ptr\n");
        }

        ptx.append("""
            )
            {
                .reg .pred  %p<2>;
                .reg .f32   %f<2>;
                .reg .b32   %r<16>;
                .reg .b64   %rd<12>;
            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    .reg .b64   %rd_t0, %rd_t1, %rd_delta;\n");
        }

        ptx.append("""

                // Calculate row (i) and col (j) from 2D grid
                mov.u32         %r1, %ctaid.x;
                mov.u32         %r2, %ntid.x;
                mov.u32         %r3, %tid.x;
                mad.lo.s32      %r4, %r1, %r2, %r3; // j = col

                mov.u32         %r5, %ctaid.y;
                mov.u32         %r6, %ntid.y;
                mov.u32         %r7, %tid.y;
                mad.lo.s32      %r8, %r5, %r6, %r7; // i = row

                // Load dimensions
                ld.param.u32    %r9, [rows];
                ld.param.u32    %r10, [cols];

                // Bounds check
                setp.ge.s32     %p1, %r8, %r9;
                @%p1 bra        EXIT;
                setp.ge.s32     %p1, %r4, %r10;
                @%p1 bra        EXIT;

                // Load pointers
                ld.param.u64    %rd1, [in_ptr];
                ld.param.u64    %rd2, [out_ptr];

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        ptx.append("""
                // Input index: just i (broadcast along columns)
                cvt.s64.s32     %rd3, %r8;
                shl.b64         %rd3, %rd3, 2;
                add.s64         %rd4, %rd1, %rd3;
                ld.global.f32   %f1, [%rd4];

                // Output index: i * cols + j
                mul.lo.s32      %r11, %r8, %r10;
                add.s32         %r11, %r11, %r4;
                cvt.s64.s32     %rd5, %r11;
                shl.b64         %rd5, %rd5, 2;
                add.s64         %rd6, %rd2, %rd5;
                st.global.f32   [%rd6], %f1;

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("""
                    mov.u64         %rd_t1, %globaltimer;
                    sub.u64         %rd_delta, %rd_t1, %rd_t0;
                    ld.param.u64    %rd11, [timing_ptr];
                    atom.global.add.u64 [%rd11], %rd_delta;

            """);
        }

        ptx.append("""
            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    /**
     * Generate PTX for scalar broadcast to any shape.
     *
     * <p>Broadcasts a single scalar value to fill an entire tensor.
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateBroadcastScalarF32(int salt) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("""
            //
            // broadcast_scalar_f32: Broadcast scalar to tensor
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            .visible .entry broadcast_scalar_f32(
                .param .u64 in_ptr,
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
                .reg .f32   %f<2>;
                .reg .b32   %r<8>;
                .reg .b64   %rd<8>;
            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    .reg .b64   %rd_t0, %rd_t1, %rd_delta;\n");
        }

        ptx.append("""

                // Calculate global thread index
                mov.u32         %r1, %ctaid.x;
                mov.u32         %r2, %ntid.x;
                mov.u32         %r3, %tid.x;
                mad.lo.s32      %r4, %r1, %r2, %r3;

                // Bounds check
                ld.param.u32    %r5, [n];
                setp.ge.s32     %p1, %r4, %r5;
                @%p1 bra        EXIT;

                // Load scalar value (same for all threads)
                ld.param.u64    %rd1, [in_ptr];
                ld.global.f32   %f1, [%rd1];

                // Load output pointer
                ld.param.u64    %rd2, [out_ptr];

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        ptx.append("""
                // Calculate output offset
                cvt.s64.s32     %rd3, %r4;
                shl.b64         %rd3, %rd3, 2;
                add.s64         %rd4, %rd2, %rd3;
                st.global.f32   [%rd4], %f1;

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("""
                    mov.u64         %rd_t1, %globaltimer;
                    sub.u64         %rd_delta, %rd_t1, %rd_t0;
                    ld.param.u64    %rd7, [timing_ptr];
                    atom.global.add.u64 [%rd7], %rd_delta;

            """);
        }

        ptx.append("""
            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    // ==================== Type Conversion Operations ====================

    /**
     * Generate PTX for F32 to I32 conversion.
     *
     * <p>Converts floating-point values to signed 32-bit integers using
     * round-toward-zero (truncation) semantics.
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateConvertF32toI32(int salt) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("""
            //
            // convert_f32_to_i32: Convert float to int (truncate)
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            .visible .entry convert_f32_to_i32(
                .param .u64 in_ptr,
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
                .reg .f32   %f<2>;
                .reg .b32   %r<8>;
                .reg .b64   %rd<8>;
            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    .reg .b64   %rd_t0, %rd_t1, %rd_delta;\n");
        }

        ptx.append("""

                // Calculate global thread index
                mov.u32         %r1, %ctaid.x;
                mov.u32         %r2, %ntid.x;
                mov.u32         %r3, %tid.x;
                mad.lo.s32      %r4, %r1, %r2, %r3;

                // Bounds check
                ld.param.u32    %r5, [n];
                setp.ge.s32     %p1, %r4, %r5;
                @%p1 bra        EXIT;

                // Load pointers
                ld.param.u64    %rd1, [in_ptr];
                ld.param.u64    %rd2, [out_ptr];

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        ptx.append("""
                // Calculate offset
                cvt.s64.s32     %rd3, %r4;
                shl.b64         %rd3, %rd3, 2;      // * 4 bytes

                // Load float from input
                add.s64         %rd4, %rd1, %rd3;
                ld.global.f32   %f1, [%rd4];

                // Convert to int (round toward zero)
                cvt.rzi.s32.f32 %r6, %f1;

                // Store int to output
                add.s64         %rd5, %rd2, %rd3;
                st.global.s32   [%rd5], %r6;

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("""
                    mov.u64         %rd_t1, %globaltimer;
                    sub.u64         %rd_delta, %rd_t1, %rd_t0;
                    ld.param.u64    %rd7, [timing_ptr];
                    atom.global.add.u64 [%rd7], %rd_delta;

            """);
        }

        ptx.append("""
            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    /**
     * Generate PTX for I32 to F32 conversion.
     *
     * <p>Converts signed 32-bit integers to floating-point values.
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateConvertI32toF32(int salt) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("""
            //
            // convert_i32_to_f32: Convert int to float
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            .visible .entry convert_i32_to_f32(
                .param .u64 in_ptr,
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
                .reg .f32   %f<2>;
                .reg .b32   %r<8>;
                .reg .b64   %rd<8>;
            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    .reg .b64   %rd_t0, %rd_t1, %rd_delta;\n");
        }

        ptx.append("""

                // Calculate global thread index
                mov.u32         %r1, %ctaid.x;
                mov.u32         %r2, %ntid.x;
                mov.u32         %r3, %tid.x;
                mad.lo.s32      %r4, %r1, %r2, %r3;

                // Bounds check
                ld.param.u32    %r5, [n];
                setp.ge.s32     %p1, %r4, %r5;
                @%p1 bra        EXIT;

                // Load pointers
                ld.param.u64    %rd1, [in_ptr];
                ld.param.u64    %rd2, [out_ptr];

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        ptx.append("""
                // Calculate offset
                cvt.s64.s32     %rd3, %r4;
                shl.b64         %rd3, %rd3, 2;      // * 4 bytes

                // Load int from input
                add.s64         %rd4, %rd1, %rd3;
                ld.global.s32   %r6, [%rd4];

                // Convert to float (round to nearest even)
                cvt.rn.f32.s32  %f1, %r6;

                // Store float to output
                add.s64         %rd5, %rd2, %rd3;
                st.global.f32   [%rd5], %f1;

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("""
                    mov.u64         %rd_t1, %globaltimer;
                    sub.u64         %rd_delta, %rd_t1, %rd_t0;
                    ld.param.u64    %rd7, [timing_ptr];
                    atom.global.add.u64 [%rd7], %rd_delta;

            """);
        }

        ptx.append("""
            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    /**
     * Generate PTX for F32 to F32 (no-op copy, for identity conversions).
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateConvertF32toF32(int salt) {
        // Same as reshape - just a copy
        return generateReshapeF32(salt);
    }

    /**
     * Generate PTX for I32 to I32 (no-op copy, for identity conversions).
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateConvertI32toI32(int salt) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("""
            //
            // convert_i32_to_i32: Copy int to int (identity)
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            .visible .entry convert_i32_to_i32(
                .param .u64 in_ptr,
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
                .reg .b32   %r<8>;
                .reg .b64   %rd<8>;
            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    .reg .b64   %rd_t0, %rd_t1, %rd_delta;\n");
        }

        ptx.append("""

                // Calculate global thread index
                mov.u32         %r1, %ctaid.x;
                mov.u32         %r2, %ntid.x;
                mov.u32         %r3, %tid.x;
                mad.lo.s32      %r4, %r1, %r2, %r3;

                // Bounds check
                ld.param.u32    %r5, [n];
                setp.ge.s32     %p1, %r4, %r5;
                @%p1 bra        EXIT;

                // Load pointers
                ld.param.u64    %rd1, [in_ptr];
                ld.param.u64    %rd2, [out_ptr];

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        ptx.append("""
                // Calculate offset
                cvt.s64.s32     %rd3, %r4;
                shl.b64         %rd3, %rd3, 2;      // * 4 bytes

                // Load from input
                add.s64         %rd4, %rd1, %rd3;
                ld.global.s32   %r6, [%rd4];

                // Store to output
                add.s64         %rd5, %rd2, %rd3;
                st.global.s32   [%rd5], %r6;

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("""
                    mov.u64         %rd_t1, %globaltimer;
                    sub.u64         %rd_delta, %rd_t1, %rd_t0;
                    ld.param.u64    %rd7, [timing_ptr];
                    atom.global.add.u64 [%rd7], %rd_delta;

            """);
        }

        ptx.append("""
            EXIT:
                ret;
            }
            """);

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
