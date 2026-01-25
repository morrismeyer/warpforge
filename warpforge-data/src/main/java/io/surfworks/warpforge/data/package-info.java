/**
 * WarpForge Data - "It Just Works" model and dataset loading.
 *
 * <h2>Quick Start</h2>
 * <pre>{@code
 * // Load a model from HuggingFace
 * var model = WarpForge.model("meta-llama/Llama-3.1-8B");
 *
 * // Access tensors
 * var weights = model.tensor("model.layers.0.self_attn.q_proj.weight");
 * float[] data = weights.toFloatArray();
 *
 * // Load from local path
 * var localModel = WarpForge.model(Path.of("./my-model.safetensors"));
 *
 * // Load a dataset
 * var squad = WarpForge.dataset("rajpurkar/squad");
 * for (var example : squad) {
 *     String question = (String) example.get("question");
 *     String context = (String) example.get("context");
 * }
 * }</pre>
 *
 * <h2>Configuration</h2>
 * <pre>{@code
 * // Set custom cache directory (e.g., NAS)
 * WarpForge.configure()
 *     .cacheDir("/mnt/nas/warpforge-data")
 *     .hubToken("hf_xxx")  // or set HF_TOKEN env var
 *     .apply();
 * }</pre>
 *
 * <h2>Supported Formats</h2>
 * <ul>
 *   <li>SafeTensors (.safetensors) - HuggingFace standard, zero-copy mmap</li>
 *   <li>GGUF (.gguf) - Quantized models (planned)</li>
 *   <li>JSON - Datasets like SQuAD</li>
 * </ul>
 *
 * @see io.surfworks.warpforge.data.WarpForge
 * @see io.surfworks.warpforge.data.model.ModelSource
 * @see io.surfworks.warpforge.data.dataset.DatasetSource
 */
package io.surfworks.warpforge.data;
