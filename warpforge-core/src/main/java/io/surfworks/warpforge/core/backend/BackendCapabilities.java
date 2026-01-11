package io.surfworks.warpforge.core.backend;

import io.surfworks.warpforge.core.tensor.ScalarType;

import java.util.Set;

/**
 * Describes the capabilities of a backend.
 */
public record BackendCapabilities(
    Set<ScalarType> supportedDtypes,
    boolean supportsVectorOps,
    boolean supportsAsync,
    int maxTensorRank,
    long maxElementCount
) {
    /**
     * Default capabilities for a basic CPU backend.
     */
    public static BackendCapabilities cpu() {
        return new BackendCapabilities(
            Set.of(ScalarType.F32, ScalarType.F64, ScalarType.I32, ScalarType.I64),
            true,  // Vector API support
            false, // No async
            8,     // Max rank
            Integer.MAX_VALUE // Max elements
        );
    }

    /**
     * Builder for custom capabilities.
     */
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private Set<ScalarType> dtypes = Set.of(ScalarType.F32);
        private boolean vectorOps = false;
        private boolean async = false;
        private int maxRank = 8;
        private long maxElements = Integer.MAX_VALUE;

        public Builder supportedDtypes(Set<ScalarType> dtypes) {
            this.dtypes = dtypes;
            return this;
        }

        public Builder supportsVectorOps(boolean supports) {
            this.vectorOps = supports;
            return this;
        }

        public Builder supportsAsync(boolean supports) {
            this.async = supports;
            return this;
        }

        public Builder maxTensorRank(int maxRank) {
            this.maxRank = maxRank;
            return this;
        }

        public Builder maxElementCount(long maxElements) {
            this.maxElements = maxElements;
            return this;
        }

        public BackendCapabilities build() {
            return new BackendCapabilities(dtypes, vectorOps, async, maxRank, maxElements);
        }
    }
}
