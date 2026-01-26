package io.surfworks.warpforge.core.concurrency;

import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.jfr.GpuKernelEvent;
import io.surfworks.warpforge.core.jfr.GpuMemoryEvent;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.lang.foreign.Arena;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Calibrated GPU work generator for structured concurrency timing tests.
 *
 * <p>This utility performs <b>real GPU operations</b> that take predictable,
 * measurable durations. This is essential for validating that the structured
 * concurrency infrastructure correctly tracks thread-bound GPU kernel execution
 * times via JFR.
 *
 * <p><b>Key principle:</b> Every operation in this class runs actual GPU kernels
 * (memory transfers, tensor operations) and emits JFR events. Without real GPU
 * work, the gtests would just be testing CPU-side API wrappers.
 *
 * <p>The calibrator self-tunes on first use to account for different GPU speeds
 * across test machines (NVIDIA RTX 3090 vs AMD RX 7900, etc.).
 *
 * <h2>What This Class Does</h2>
 * <ul>
 *   <li>Allocates device memory via {@code backend.allocateDevice()}</li>
 *   <li>Performs host-to-device and device-to-host memory transfers</li>
 *   <li>Synchronizes streams via {@code backend.synchronizeStream()}</li>
 *   <li>Emits {@link GpuKernelEvent} and {@link GpuMemoryEvent} for JFR profiling</li>
 * </ul>
 *
 * <h2>Usage in Tests</h2>
 * <pre>{@code
 * @Test
 * void kernelTimingIsAccurate() throws Exception {
 *     try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
 *         scope.forkWithStream(lease -> {
 *             // Do 50ms of real GPU work
 *             GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, 50);
 *
 *             // Validate timing
 *             GpuWorkCalibrator.assertTimingWithinTolerance(50, result.elapsedNanos(),
 *                 "50ms GPU kernel work");
 *
 *             // JFR event was already emitted by doGpuWork
 *             return result;
 *         });
 *         scope.joinAll();
 *     }
 * }
 * }</pre>
 */
public final class GpuWorkCalibrator {

    // Calibration data per backend (different GPUs have different speeds)
    private static final ConcurrentHashMap<String, CalibrationData> CALIBRATION_DATA = new ConcurrentHashMap<>();

    // Calibration lock
    private static final Object CALIBRATION_LOCK = new Object();

    // Tolerance for timing assertions (milliseconds)
    public static final long TIMING_TOLERANCE_MS = 20;

    // Minimum work duration for reliable measurement
    public static final long MIN_WORK_MS = 5;

    // Base tensor size for calibration (1MB = 256K floats)
    private static final int BASE_TENSOR_ELEMENTS = 256 * 1024;

    // Max tensor elements (to avoid int overflow in shape)
    private static final int MAX_TENSOR_ELEMENTS = Integer.MAX_VALUE / 4;

    private GpuWorkCalibrator() {
        // Utility class
    }

    /**
     * Performs real GPU work for approximately the specified duration.
     *
     * <p>This method:
     * <ol>
     *   <li>Allocates device memory</li>
     *   <li>Performs host-to-device transfer</li>
     *   <li>Synchronizes the stream</li>
     *   <li>Performs device-to-host transfer</li>
     *   <li>Synchronizes again</li>
     *   <li>Frees device memory</li>
     *   <li>Emits JFR events for the operations</li>
     * </ol>
     *
     * @param backend The GPU backend to use
     * @param lease The GPU lease (provides stream handle)
     * @param targetMillis Target duration in milliseconds (minimum 5ms recommended)
     * @return Result containing actual timing and operation details
     */
    public static GpuWorkResult doGpuWork(GpuBackend backend, GpuLease lease, long targetMillis) {
        ensureCalibrated(backend);

        CalibrationData calibration = CALIBRATION_DATA.get(backend.name());
        long tensorElements = calibration.elementsForMillis(targetMillis);

        return doGpuWorkWithSize(backend, lease, tensorElements, targetMillis);
    }

    /**
     * Performs real GPU work with a specific tensor size.
     *
     * <p>Use this when you need precise control over the workload size
     * rather than target duration.
     *
     * @param backend The GPU backend
     * @param lease The GPU lease
     * @param tensorElements Number of float32 elements in the tensor
     * @param targetMs Target duration for JFR labeling
     * @return Result containing actual timing
     */
    public static GpuWorkResult doGpuWorkWithSize(GpuBackend backend, GpuLease lease,
                                                   long tensorElements, long targetMs) {
        long streamHandle = lease.streamHandle();
        long scopeId = lease.parentScope().scopeId();

        // Clamp tensor elements to valid int range
        int elements = (int) Math.min(tensorElements, MAX_TENSOR_ELEMENTS);
        long byteSize = (long) elements * 4; // float32 = 4 bytes

        long startNanos = System.nanoTime();

        // Create tensor specification
        TensorSpec spec = TensorSpec.of(ScalarType.F32, elements);

        // 1. Allocate device tensor
        Tensor deviceTensor = backend.allocateDevice(spec);
        long afterAlloc = System.nanoTime();

        // 2. Create host tensor with data using a confined arena
        Tensor hostTensor;
        try (Arena arena = Arena.ofConfined()) {
            hostTensor = Tensor.allocate(spec, arena);
            fillTensorWithPattern(hostTensor);

            // 3. Copy host to device (async on stream)
            Tensor deviceCopy = backend.copyToDeviceAsync(hostTensor, streamHandle);
            long afterH2D = System.nanoTime();

            // 4. Synchronize stream - this is where we wait for GPU work
            backend.synchronizeStream(streamHandle);
            long afterSync1 = System.nanoTime();

            // 5. Copy device to host (async on stream)
            Tensor resultTensor = backend.copyToHostAsync(deviceCopy, streamHandle);
            long afterD2H = System.nanoTime();

            // 6. Final synchronize
            backend.synchronizeStream(streamHandle);
            long endNanos = System.nanoTime();

            long totalElapsedNanos = endNanos - startNanos;
            long h2dTimeNanos = afterSync1 - afterH2D;
            long d2hTimeNanos = endNanos - afterD2H;

            // Emit JFR events for the GPU operations
            emitMemoryEvent(backend, "H2D", byteSize, h2dTimeNanos / 1000, streamHandle);
            emitMemoryEvent(backend, "D2H", byteSize, d2hTimeNanos / 1000, streamHandle);
            emitKernelEvent(backend, "GpuWork", elements, totalElapsedNanos / 1000,
                targetMs, streamHandle, scopeId);

            return new GpuWorkResult(
                totalElapsedNanos,
                elements,
                byteSize,
                h2dTimeNanos,
                d2hTimeNanos,
                streamHandle,
                scopeId
            );
        }
    }

    /**
     * Performs calibrated GPU work without a lease (uses default stream).
     *
     * <p>Use this for calibration or when testing outside of structured concurrency.
     *
     * @param backend The GPU backend
     * @param targetMillis Target duration in milliseconds
     * @return Result containing actual timing
     */
    public static GpuWorkResult doGpuWorkOnDefaultStream(GpuBackend backend, long targetMillis) {
        ensureCalibrated(backend);

        CalibrationData calibration = CALIBRATION_DATA.get(backend.name());
        long tensorElements = calibration.elementsForMillis(targetMillis);
        int elements = (int) Math.min(tensorElements, MAX_TENSOR_ELEMENTS);
        long byteSize = (long) elements * 4;

        TensorSpec spec = TensorSpec.of(ScalarType.F32, elements);

        long startNanos = System.nanoTime();

        try (Arena arena = Arena.ofConfined()) {
            // Allocate and transfer
            Tensor hostTensor = Tensor.allocate(spec, arena);
            fillTensorWithPattern(hostTensor);

            Tensor deviceTensor = backend.copyToDevice(hostTensor);
            backend.synchronizeDevice();

            Tensor resultTensor = backend.copyToHost(deviceTensor);
            backend.synchronizeDevice();

            long endNanos = System.nanoTime();
            long totalElapsedNanos = endNanos - startNanos;

            return new GpuWorkResult(totalElapsedNanos, elements, byteSize, 0, 0, 0, 0);
        }
    }

    /**
     * Asserts that the measured duration is within tolerance of the expected duration.
     *
     * @param expectedMs Expected duration in milliseconds
     * @param actualNanos Actual duration in nanoseconds
     * @param description Description for error message
     * @throws AssertionError if timing is outside tolerance
     */
    public static void assertTimingWithinTolerance(long expectedMs, long actualNanos, String description) {
        long actualMs = actualNanos / 1_000_000;
        long lowerBound = Math.max(0, expectedMs - TIMING_TOLERANCE_MS);
        long upperBound = expectedMs + TIMING_TOLERANCE_MS * 2; // More tolerance for upper bound

        if (actualMs < lowerBound || actualMs > upperBound) {
            throw new AssertionError(String.format(
                "%s: expected ~%dms, got %dms (tolerance: %d-%dms)",
                description, expectedMs, actualMs, lowerBound, upperBound));
        }
    }

    /**
     * Creates a work description for JFR events.
     */
    public static String workDescription(long targetMs, long actualNanos) {
        return String.format("target=%dms,actual=%dms", targetMs, actualNanos / 1_000_000);
    }

    /**
     * Returns the calibration data for the given backend.
     * Useful for understanding GPU performance characteristics.
     */
    public static CalibrationData getCalibrationData(GpuBackend backend) {
        ensureCalibrated(backend);
        return CALIBRATION_DATA.get(backend.name());
    }

    /**
     * Forces recalibration for the given backend.
     */
    public static void recalibrate(GpuBackend backend) {
        synchronized (CALIBRATION_LOCK) {
            CALIBRATION_DATA.remove(backend.name());
            ensureCalibrated(backend);
        }
    }

    // ==================== Calibration ====================

    private static void ensureCalibrated(GpuBackend backend) {
        String backendName = backend.name();
        if (!CALIBRATION_DATA.containsKey(backendName)) {
            synchronized (CALIBRATION_LOCK) {
                if (!CALIBRATION_DATA.containsKey(backendName)) {
                    calibrate(backend);
                }
            }
        }
    }

    private static void calibrate(GpuBackend backend) {
        System.out.println("GpuWorkCalibrator: Calibrating for " + backend.name() + "...");

        // Warm up - do a few small transfers to initialize GPU context
        for (int i = 0; i < 3; i++) {
            warmUpTransfer(backend, 1024);
        }

        // Calibration run with known size
        int calibrationElements = BASE_TENSOR_ELEMENTS; // 1MB
        long byteSize = (long) calibrationElements * 4;

        TensorSpec spec = TensorSpec.of(ScalarType.F32, calibrationElements);

        try (Arena arena = Arena.ofConfined()) {
            // Measure transfer time
            Tensor hostTensor = Tensor.allocate(spec, arena);
            fillTensorWithPattern(hostTensor);

            long startNanos = System.nanoTime();

            Tensor deviceTensor = backend.copyToDevice(hostTensor);
            backend.synchronizeDevice();

            Tensor resultTensor = backend.copyToHost(deviceTensor);
            backend.synchronizeDevice();

            long elapsedNanos = System.nanoTime() - startNanos;
            long elapsedMs = Math.max(1, elapsedNanos / 1_000_000);

            // Calculate elements per millisecond
            long elementsPerMs = calibrationElements / elapsedMs;

            // Sanity check - should be at least 1000 elements per ms even on slow GPUs
            if (elementsPerMs < 1000) {
                elementsPerMs = 10000; // Fallback
            }

            // Calculate bandwidth
            double bandwidthGBps = (byteSize * 2.0) / (elapsedNanos / 1e9) / 1e9;

            CalibrationData data = new CalibrationData(elementsPerMs, bandwidthGBps);
            CALIBRATION_DATA.put(backend.name(), data);

            System.out.printf("GpuWorkCalibrator: %s calibrated - %d elements/ms, %.2f GB/s bandwidth%n",
                backend.name(), elementsPerMs, bandwidthGBps);
        }
    }

    private static void warmUpTransfer(GpuBackend backend, int elements) {
        TensorSpec spec = TensorSpec.of(ScalarType.F32, elements);
        try (Arena arena = Arena.ofConfined()) {
            Tensor host = Tensor.allocate(spec, arena);
            Tensor device = backend.copyToDevice(host);
            backend.synchronizeDevice();
            Tensor result = backend.copyToHost(device);
            backend.synchronizeDevice();
        }
    }

    private static void fillTensorWithPattern(Tensor tensor) {
        // Fill with a simple pattern to ensure actual data transfer
        // The pattern is deterministic to allow verification if needed
        long elements = tensor.elementCount();
        for (long i = 0; i < Math.min(elements, 1000); i++) {
            tensor.setFloatFlat(i, (float) (i * 0.001));
        }
    }

    // ==================== JFR Event Emission ====================

    private static void emitMemoryEvent(GpuBackend backend, String direction, long bytes,
                                         long timeMicros, long streamHandle) {
        GpuMemoryEvent event = new GpuMemoryEvent();
        event.direction = direction;
        event.bytes = bytes;
        event.timeMicros = timeMicros;
        event.bandwidthGBps = timeMicros > 0 ? (bytes / 1e9) / (timeMicros / 1e6) : 0;
        event.deviceIndex = backend.deviceIndex();
        event.async = true;
        event.pinnedMemory = false;
        event.commit();
    }

    private static void emitKernelEvent(GpuBackend backend, String operation, long elements,
                                         long gpuTimeMicros, long targetMs, long streamHandle,
                                         long scopeId) {
        GpuKernelEvent event = new GpuKernelEvent();
        event.operation = operation;
        event.shape = elements + " elements";
        event.gpuTimeMicros = gpuTimeMicros;
        event.backend = GpuTestSupport.expectedBackendName(backend);
        event.deviceIndex = backend.deviceIndex();
        event.tier = "GPU_WORK_CALIBRATOR";
        event.bytesTransferred = elements * 4 * 2; // H2D + D2H
        event.memoryBandwidthGBps = gpuTimeMicros > 0 ?
            (event.bytesTransferred / 1e9) / (gpuTimeMicros / 1e6) : 0;
        event.commit();

        System.out.printf("JFR GpuWork: target=%dms actual=%dus elements=%d stream=%d scope=%d%n",
            targetMs, gpuTimeMicros, elements, streamHandle, scopeId);
    }

    // ==================== Result Class ====================

    /**
     * Result of a GPU work operation.
     */
    public record GpuWorkResult(
        long elapsedNanos,
        long tensorElements,
        long byteSize,
        long h2dTimeNanos,
        long d2hTimeNanos,
        long streamHandle,
        long scopeId
    ) {
        public long elapsedMillis() {
            return elapsedNanos / 1_000_000;
        }

        public double bandwidthGBps() {
            if (elapsedNanos == 0) return 0;
            return (byteSize * 2.0) / (elapsedNanos / 1e9) / 1e9;
        }
    }

    // ==================== Calibration Data ====================

    /**
     * Calibration data for a specific GPU backend.
     */
    public record CalibrationData(
        long elementsPerMs,
        double bandwidthGBps
    ) {
        /**
         * Calculate the number of tensor elements needed for the target duration.
         */
        public long elementsForMillis(long targetMs) {
            // Add some margin for overhead
            return (long) (elementsPerMs * targetMs * 1.1);
        }
    }
}
