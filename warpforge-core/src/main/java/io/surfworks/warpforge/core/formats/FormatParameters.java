package io.surfworks.warpforge.core.formats;

/**
 * P3109 format parameters defining a small binary floating-point format.
 *
 * <p>Per IEEE P3109 Section 3, a format is defined by four parameters:
 * <ul>
 *   <li><b>K</b> - Total bit width (must be &ge; 3)</li>
 *   <li><b>P</b> - Precision (significand width including implicit leading bit)</li>
 *   <li><b>&Sigma;</b> - Signedness: signed (true) or unsigned (false)</li>
 *   <li><b>&Delta;</b> - Domain: finite-only (true) or extended with infinities (false)</li>
 * </ul>
 *
 * <p>The exponent width E is derived:
 * <ul>
 *   <li>For signed: E = K - P (since K = 1 + E + M and M = P - 1)</li>
 *   <li>For unsigned: E = K - P + 1 (since K = E + M and M = P - 1)</li>
 * </ul>
 *
 * <p>Common format names follow the pattern: Binary{K}p{P}{s|u}{e|f}
 * <ul>
 *   <li>s = signed, u = unsigned</li>
 *   <li>e = extended (has infinities), f = finite-only</li>
 * </ul>
 *
 * @param bitWidth Total bit width K (must be &ge; 3)
 * @param precision Significand precision P (includes implicit leading bit)
 * @param signed Whether the format is signed (&Sigma;)
 * @param finiteOnly Whether the domain excludes infinities (&Delta;)
 */
public record FormatParameters(
        int bitWidth,
        int precision,
        boolean signed,
        boolean finiteOnly
) {
    // Well-known P3109 format definitions

    /** Binary8p3se: FP8 E5M2 - wide dynamic range, signed, extended */
    public static final FormatParameters FP8_E5M2 = new FormatParameters(8, 3, true, false);

    /** Binary8p4se: FP8 E4M3 - higher precision, signed, extended */
    public static final FormatParameters FP8_E4M3 = new FormatParameters(8, 4, true, false);

    /** Binary8p4sf: FP8 E4M3FN - E4M3 finite-only (NVIDIA variant) */
    public static final FormatParameters FP8_E4M3FN = new FormatParameters(8, 4, true, true);

    /** Binary8p1uf: FP8 E8M0 - exponent-only scale factor, unsigned, finite */
    public static final FormatParameters FP8_E8M0 = new FormatParameters(8, 1, false, true);

    /** Binary4p2sf: FP4 E2M1 - 4-bit with 1 mantissa bit, finite-only for ML use */
    public static final FormatParameters FP4_E2M1 = new FormatParameters(4, 2, true, true);

    /** Binary4p3sf: FP4 E1M2 - 4-bit with 2 mantissa bits, finite-only for ML use */
    public static final FormatParameters FP4_E1M2 = new FormatParameters(4, 3, true, true);

    /** Binary6p3sf: FP6 E3M2 - 6-bit with 2 mantissa bits, finite-only for ML use */
    public static final FormatParameters FP6_E3M2 = new FormatParameters(6, 3, true, true);

    /** Binary6p4sf: FP6 E2M3 - 6-bit with 3 mantissa bits, finite-only for ML use */
    public static final FormatParameters FP6_E2M3 = new FormatParameters(6, 4, true, true);

    public FormatParameters {
        if (bitWidth < 3) {
            throw new IllegalArgumentException("Bit width must be >= 3, got " + bitWidth);
        }
        if (precision < 1) {
            throw new IllegalArgumentException("Precision must be >= 1, got " + precision);
        }
        // For signed: K = 1 + E + M, M = P - 1, so E = K - P
        // For unsigned: K = E + M, M = P - 1, so E = K - P + 1
        int exponentWidth = signed ? (bitWidth - precision) : (bitWidth - precision + 1);
        if (exponentWidth < 1) {
            throw new IllegalArgumentException(
                    "Invalid format: K=" + bitWidth + ", P=" + precision + ", signed=" + signed +
                    " yields exponent width " + exponentWidth + " (must be >= 1)");
        }
    }

    /**
     * Exponent field width in bits.
     * For signed: E = K - P
     * For unsigned: E = K - P + 1
     */
    public int exponentWidth() {
        return signed ? (bitWidth - precision) : (bitWidth - precision + 1);
    }

    /**
     * Mantissa field width in bits (trailing significand bits stored).
     * M = P - 1 (since P includes the implicit leading bit)
     */
    public int mantissaWidth() {
        return precision - 1;
    }

    /**
     * The exponent bias.
     * bias = 2^(E-1) - 1
     */
    public int exponentBias() {
        return (1 << (exponentWidth() - 1)) - 1;
    }

    /**
     * Maximum biased exponent value (all 1s).
     */
    public int maxBiasedExponent() {
        return (1 << exponentWidth()) - 1;
    }

    /**
     * Minimum unbiased exponent for normal numbers.
     * emin = 1 - bias
     */
    public int minExponent() {
        return 1 - exponentBias();
    }

    /**
     * Maximum unbiased exponent for normal numbers.
     *
     * <p>In P3109, the max biased exponent is usable for finite values in all formats.
     * For extended signed formats, only the single code with max exp AND max mantissa
     * is reserved for infinity (not the entire exponent range like IEEE 754).
     */
    public int maxExponent() {
        return maxBiasedExponent() - exponentBias();
    }

    /**
     * Maximum representable finite value.
     *
     * <p>For P3109 signed extended formats, only the max code (max exp, max mantissa)
     * is infinity. All other codes at max exponent are finite. So max finite value
     * uses max exponent but (maxMantissa - 1).
     */
    public double maxValue() {
        int M = mantissaWidth();
        int emax = maxExponent();
        int maxMant = (1 << M) - 1;

        if (!finiteOnly && signed) {
            // P3109 signed extended: max code is +Inf, so max finite uses (maxMant - 1)
            maxMant = maxMant - 1;
        } else if (!finiteOnly && !signed) {
            // Unsigned extended: NaN at max code, +Inf at second-to-last
            // Max finite is at (maxMant - 2)
            maxMant = maxMant - 2;
        }
        // For finite-only, use full maxMant

        double mantissaVal = 1.0 + (double) maxMant / (1 << M);
        return Math.scalb(mantissaVal, emax);
    }

    /**
     * Minimum positive normal value.
     */
    public double minNormal() {
        return Math.scalb(1.0, minExponent());
    }

    /**
     * Minimum positive subnormal value.
     */
    public double minSubnormal() {
        // Smallest subnormal: 2^(emin - (P-1)) = 2^(emin - M)
        return Math.scalb(1.0, minExponent() - mantissaWidth());
    }

    /**
     * Machine epsilon: difference between 1.0 and next representable value.
     * eps = 2^(1-P)
     */
    public double epsilon() {
        return Math.scalb(1.0, 1 - precision);
    }

    /**
     * Total number of distinct bit patterns.
     */
    public int totalPatterns() {
        return 1 << bitWidth;
    }

    /**
     * P3109 canonical name: Binary{K}p{P}{s|u}{e|f}
     */
    public String canonicalName() {
        return "Binary" + bitWidth + "p" + precision +
                (signed ? "s" : "u") +
                (finiteOnly ? "f" : "e");
    }

    /**
     * Short name commonly used: E{e}M{m} where e=exponent width, m=mantissa width.
     */
    public String shortName() {
        return "E" + exponentWidth() + "M" + mantissaWidth();
    }

    @Override
    public String toString() {
        return canonicalName() + " (" + shortName() + ")";
    }
}
