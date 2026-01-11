package io.surfworks.warpforge.core.io;

import io.surfworks.warpforge.core.tensor.ScalarType;

import java.nio.ByteOrder;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Represents the header of a NumPy .npy file.
 * Contains dtype, shape, and memory layout information.
 */
public record NpyHeader(
    int majorVersion,
    int minorVersion,
    ScalarType dtype,
    ByteOrder byteOrder,
    boolean fortranOrder,
    int[] shape
) {
    // Pattern to parse the Python dict header
    // Example: {'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }
    private static final Pattern DESCR_PATTERN = Pattern.compile("'descr'\\s*:\\s*'([^']+)'");
    private static final Pattern FORTRAN_PATTERN = Pattern.compile("'fortran_order'\\s*:\\s*(True|False)");
    private static final Pattern SHAPE_PATTERN = Pattern.compile("'shape'\\s*:\\s*\\(([^)]*)\\)");

    /**
     * Parse a NumPy header string (the Python dict literal).
     */
    public static NpyHeader parse(int majorVersion, int minorVersion, String headerStr) {
        // Parse descr (dtype)
        Matcher descrMatcher = DESCR_PATTERN.matcher(headerStr);
        if (!descrMatcher.find()) {
            throw new IllegalArgumentException("Missing 'descr' in header: " + headerStr);
        }
        String descr = descrMatcher.group(1);

        // Parse byte order and dtype from descr
        ByteOrder byteOrder = parseByteOrder(descr);
        ScalarType dtype = ScalarType.fromNpyDtype(descr);

        // Parse fortran_order
        Matcher fortranMatcher = FORTRAN_PATTERN.matcher(headerStr);
        boolean fortranOrder = false;
        if (fortranMatcher.find()) {
            fortranOrder = "True".equals(fortranMatcher.group(1));
        }

        // Parse shape
        Matcher shapeMatcher = SHAPE_PATTERN.matcher(headerStr);
        if (!shapeMatcher.find()) {
            throw new IllegalArgumentException("Missing 'shape' in header: " + headerStr);
        }
        int[] shape = parseShape(shapeMatcher.group(1));

        return new NpyHeader(majorVersion, minorVersion, dtype, byteOrder, fortranOrder, shape);
    }

    /**
     * Parse byte order from dtype string prefix.
     */
    private static ByteOrder parseByteOrder(String descr) {
        if (descr.isEmpty()) {
            return ByteOrder.nativeOrder();
        }
        char prefix = descr.charAt(0);
        return switch (prefix) {
            case '<' -> ByteOrder.LITTLE_ENDIAN;
            case '>' -> ByteOrder.BIG_ENDIAN;
            case '=' -> ByteOrder.nativeOrder();
            case '|' -> ByteOrder.nativeOrder(); // Single byte, endianness doesn't matter
            default -> ByteOrder.nativeOrder();
        };
    }

    /**
     * Parse shape tuple from string (e.g., "2, 3, 4" or empty for scalar).
     */
    private static int[] parseShape(String shapeStr) {
        String trimmed = shapeStr.trim();
        if (trimmed.isEmpty()) {
            return new int[0]; // Scalar
        }

        String[] parts = trimmed.split("\\s*,\\s*");
        // Handle trailing comma: (3,) -> ["3", ""]
        int count = 0;
        for (String part : parts) {
            if (!part.isEmpty()) count++;
        }

        int[] shape = new int[count];
        int idx = 0;
        for (String part : parts) {
            if (!part.isEmpty()) {
                shape[idx++] = Integer.parseInt(part);
            }
        }
        return shape;
    }

    /**
     * Compute total element count from shape.
     */
    public long elementCount() {
        if (shape.length == 0) {
            return 1; // Scalar
        }
        long count = 1;
        for (int dim : shape) {
            count *= dim;
        }
        return count;
    }

    /**
     * Generate the header string for writing.
     */
    public String toHeaderString() {
        StringBuilder sb = new StringBuilder();
        sb.append("{'descr': '").append(dtype.toNpyDtype()).append("', ");
        sb.append("'fortran_order': ").append(fortranOrder ? "True" : "False").append(", ");
        sb.append("'shape': (");
        for (int i = 0; i < shape.length; i++) {
            sb.append(shape[i]);
            if (i < shape.length - 1 || shape.length == 1) {
                sb.append(", ");
            }
        }
        sb.append("), }");
        return sb.toString();
    }

    /**
     * Check if byte swapping is needed when reading.
     */
    public boolean needsByteSwap() {
        return byteOrder != ByteOrder.nativeOrder() &&
               dtype.byteSize() > 1;
    }
}
