package io.surfworks.warpforge.data.model;

import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import java.util.Map;
import java.util.Set;

/**
 * Interface for accessing model weights from various formats.
 *
 * <p>ModelSource provides a unified API for loading model tensors from
 * SafeTensors, GGUF, PyTorch checkpoints, or other formats.
 */
public interface ModelSource extends AutoCloseable {

    /**
     * Get the model identifier (e.g., "meta-llama/Llama-3.1-8B").
     */
    String id();

    /**
     * Get names of all tensors in the model.
     */
    Set<String> tensorNames();

    /**
     * Check if a tensor exists.
     */
    boolean hasTensor(String name);

    /**
     * Get metadata about a tensor.
     */
    TensorInfo tensorInfo(String name);

    /**
     * Get a view of a tensor's data.
     * Zero-copy where possible.
     */
    TensorView tensor(String name);

    /**
     * Get all tensor infos.
     */
    Map<String, TensorInfo> allTensorInfos();

    /**
     * Model metadata (architecture, config, etc.).
     */
    Map<String, Object> metadata();

    /**
     * Total number of parameters in the model.
     */
    default long parameterCount() {
        return allTensorInfos().values().stream()
                .mapToLong(TensorInfo::elementCount)
                .sum();
    }

    /**
     * Estimated memory footprint in bytes.
     */
    default long memorySizeBytes() {
        return allTensorInfos().values().stream()
                .mapToLong(TensorInfo::size)
                .sum();
    }

    @Override
    void close();
}
