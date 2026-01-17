package io.surfworks.warpforge.io.collective;

/**
 * Reduction operations for collective communication.
 *
 * <p>These operations define how values from all ranks are combined
 * during allreduce, reduce, and reduce-scatter operations.
 */
public enum AllReduceOp {

    /** Sum of all values */
    SUM,

    /** Product of all values */
    PROD,

    /** Minimum value */
    MIN,

    /** Maximum value */
    MAX,

    /** Average (sum / world_size) */
    AVG,

    /** Bitwise AND */
    BAND,

    /** Bitwise OR */
    BOR,

    /** Bitwise XOR */
    BXOR,

    /** Logical AND */
    LAND,

    /** Logical OR */
    LOR,

    /** Minimum value with location (index of min) */
    MINLOC,

    /** Maximum value with location (index of max) */
    MAXLOC;

    /**
     * Returns the UCC operation code for this reduction operation.
     */
    public int uccCode() {
        return switch (this) {
            case SUM -> 0;
            case PROD -> 1;
            case MIN -> 2;
            case MAX -> 3;
            case AVG -> 4;
            case BAND -> 5;
            case BOR -> 6;
            case BXOR -> 7;
            case LAND -> 8;
            case LOR -> 9;
            case MINLOC -> 10;
            case MAXLOC -> 11;
        };
    }

    /**
     * Returns the reduction operation from its UCC code.
     */
    public static AllReduceOp fromUccCode(int code) {
        return switch (code) {
            case 0 -> SUM;
            case 1 -> PROD;
            case 2 -> MIN;
            case 3 -> MAX;
            case 4 -> AVG;
            case 5 -> BAND;
            case 6 -> BOR;
            case 7 -> BXOR;
            case 8 -> LAND;
            case 9 -> LOR;
            case 10 -> MINLOC;
            case 11 -> MAXLOC;
            default -> throw new IllegalArgumentException("Unknown UCC reduction code: " + code);
        };
    }
}
