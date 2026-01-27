package io.surfworks.warpforge.core.endtoend;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Generator for EndToEnd test fixtures using the snakegrinder-dist native binary.
 *
 * <p>This is NOT a regular test - it's a fixture generator that:
 * <ol>
 *   <li>Invokes snakegrinder-dist/build/dist/bin/snakegrinder (native binary)</li>
 *   <li>Uses --trace-with-values to capture tensor data</li>
 *   <li>Writes fixtures to warpforge-core/src/test/resources/fixtures/endtoend/</li>
 * </ol>
 *
 * <p>Run manually with: ./gradlew :warpforge-core:generateEndToEndFixtures
 *
 * <p>Prerequisites:
 * <ul>
 *   <li>./gradlew :snakegrinder-dist:assembleDist (build the native distribution)</li>
 * </ul>
 */
@Tag("fixture-generator")
@Tag("requires-snakegrinder-dist")
@DisplayName("EndToEnd Fixture Generator")
class EndToEndFixtureGenerator {

    private static final Path PROJECT_ROOT = findProjectRoot();
    private static final Path SNAKEGRINDER_BINARY = PROJECT_ROOT.resolve(
        "snakegrinder-dist/build/dist/bin/snakegrinder"
    );

    /**
     * ALL fixtures go here - build/ is always gitignored, so data can NEVER
     * be accidentally committed to the repository.
     */
    private static final Path FIXTURES_OUTPUT_DIR = PROJECT_ROOT.resolve(
        "warpforge-core/build/generated-fixtures/e2e"
    );

    private static final long DEFAULT_SEED = 42;
    private static final int TIMEOUT_SECONDS = 120;

    @TempDir
    Path tempDir;

    private static Path findProjectRoot() {
        Path current = Paths.get("").toAbsolutePath();
        while (current != null) {
            if (Files.exists(current.resolve("settings.gradle")) ||
                Files.exists(current.resolve("settings.gradle.kts"))) {
                return current;
            }
            current = current.getParent();
        }
        return Paths.get("").toAbsolutePath();
    }

    @BeforeAll
    static void checkPrerequisites() {
        assumeTrue(Files.exists(SNAKEGRINDER_BINARY),
            "snakegrinder binary not found at " + SNAKEGRINDER_BINARY +
            ". Run: ./gradlew :snakegrinder-dist:assembleDist");
        assumeTrue(Files.isExecutable(SNAKEGRINDER_BINARY),
            "snakegrinder binary is not executable");
    }

    // ==================== Tier 1: Elementwise Operations ====================

    @Test
    @DisplayName("Generate: add")
    void generateAdd() throws Exception {
        generateFixture("add", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x, y):
                    return x + y
            """, "[(4,), (4,)]");
    }

    @Test
    @DisplayName("Generate: subtract")
    void generateSubtract() throws Exception {
        generateFixture("subtract", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x, y):
                    return x - y
            """, "[(4,), (4,)]");
    }

    @Test
    @DisplayName("Generate: multiply")
    void generateMultiply() throws Exception {
        generateFixture("multiply", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x, y):
                    return x * y
            """, "[(4,), (4,)]");
    }

    @Test
    @DisplayName("Generate: negate")
    void generateNegate() throws Exception {
        generateFixture("negate", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x):
                    return -x
            """, "[(4,)]");
    }

    @Test
    @DisplayName("Generate: abs")
    void generateAbs() throws Exception {
        generateFixture("abs", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x):
                    return torch.abs(x)
            """, "[(4,)]");
    }

    // ==================== Tier 2: Transcendental Operations ====================

    @Test
    @DisplayName("Generate: exp")
    void generateExp() throws Exception {
        generateFixture("exp", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x):
                    return torch.exp(x)
            """, "[(4,)]");
    }

    @Test
    @DisplayName("Generate: tanh")
    void generateTanh() throws Exception {
        generateFixture("tanh", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x):
                    return torch.tanh(x)
            """, "[(4,)]");
    }

    @Test
    @DisplayName("Generate: sigmoid")
    void generateSigmoid() throws Exception {
        generateFixture("sigmoid", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x):
                    return torch.sigmoid(x)
            """, "[(4,)]");
    }

    // ==================== Tier 3: ReLU (composite) ====================

    @Test
    @DisplayName("Generate: relu")
    void generateRelu() throws Exception {
        generateFixture("relu", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x):
                    return torch.relu(x)
            """, "[(4,)]");
    }

    // ==================== Tier 3.5: Linear Layer (with weights) ====================

    @Test
    @DisplayName("Generate: linear")
    void generateLinear() throws Exception {
        generateFixture("linear", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(8, 4)

                def forward(self, x):
                    return self.fc(x)
            """, "[(2, 8)]");
    }

    @Test
    @DisplayName("Generate: linear_no_bias")
    void generateLinearNoBias() throws Exception {
        generateFixture("linear_no_bias", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(8, 4, bias=False)

                def forward(self, x):
                    return self.fc(x)
            """, "[(2, 8)]");
    }

    // ==================== Tier 4: Transformer Operations ====================

    @Test
    @DisplayName("Generate: layer_norm")
    void generateLayerNorm() throws Exception {
        generateFixture("layer_norm", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.ln = nn.LayerNorm(8)

                def forward(self, x):
                    return self.ln(x)
            """, "[(2, 8)]");
    }

    @Test
    @DisplayName("Generate: layer_norm_3d")
    void generateLayerNorm3D() throws Exception {
        generateFixture("layer_norm_3d", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.ln = nn.LayerNorm(16)

                def forward(self, x):
                    return self.ln(x)
            """, "[(2, 4, 16)]");
    }

    @Test
    @DisplayName("Generate: gelu")
    void generateGelu() throws Exception {
        generateFixture("gelu", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gelu = nn.GELU()

                def forward(self, x):
                    return self.gelu(x)
            """, "[(4, 8)]");
    }

    @Test
    @DisplayName("Generate: gelu_tanh")
    void generateGeluTanh() throws Exception {
        generateFixture("gelu_tanh", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gelu = nn.GELU(approximate='tanh')

                def forward(self, x):
                    return self.gelu(x)
            """, "[(4, 8)]");
    }

    @Test
    @DisplayName("Generate: silu")
    void generateSilu() throws Exception {
        generateFixture("silu", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.silu = nn.SiLU()

                def forward(self, x):
                    return self.silu(x)
            """, "[(4, 8)]");
    }

    @Test
    @DisplayName("Generate: softmax")
    void generateSoftmax() throws Exception {
        generateFixture("softmax", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.softmax = nn.Softmax(dim=-1)

                def forward(self, x):
                    return self.softmax(x)
            """, "[(2, 8)]");
    }

    @Test
    @DisplayName("Generate: softmax_3d")
    void generateSoftmax3D() throws Exception {
        generateFixture("softmax_3d", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.softmax = nn.Softmax(dim=-1)

                def forward(self, x):
                    return self.softmax(x)
            """, "[(2, 4, 8)]");
    }

    // ==================== Tier 5: Composite Transformer Patterns ====================

    // @Test
    // @DisplayName("Generate: attention_scores")
    // void generateAttentionScores() throws Exception {
    //     // Q @ K^T / sqrt(d_k) pattern
    //     // DISABLED: q.size(-1) generates malformed MLIR (undefined variable)
    //     generateFixture("attention_scores", """
    //         import torch
    //         import torch.nn as nn
    //         import math
    //
    //         class Model(nn.Module):
    //             def forward(self, q, k):
    //                 d_k = q.size(-1)
    //                 k_t = k.transpose(-2, -1)
    //                 scores = torch.matmul(q, k_t) / math.sqrt(d_k)
    //                 return scores
    //         """, "[(1, 4, 8), (1, 4, 8)]");
    // }

    @Test
    @DisplayName("Generate: ffn_block")
    void generateFfnBlock() throws Exception {
        // Feed-forward network block: Linear -> GELU -> Linear
        generateFixture("ffn_block", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(16, 64, bias=False)
                    self.gelu = nn.GELU()
                    self.fc2 = nn.Linear(64, 16, bias=False)

                def forward(self, x):
                    x = self.fc1(x)
                    x = self.gelu(x)
                    x = self.fc2(x)
                    return x
            """, "[(2, 4, 16)]");
    }

    @Test
    @DisplayName("Generate: pre_norm_residual")
    void generatePreNormResidual() throws Exception {
        // Pre-normalization residual block: LayerNorm -> Linear -> add residual
        generateFixture("pre_norm_residual", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.ln = nn.LayerNorm(16)
                    self.fc = nn.Linear(16, 16, bias=False)

                def forward(self, x):
                    residual = x
                    x = self.ln(x)
                    x = self.fc(x)
                    return x + residual
            """, "[(2, 4, 16)]");
    }

    // ==================== Tier 5.5: Attention Mechanisms ====================

    @Test
    @DisplayName("Generate: scaled_dot_product_attention")
    void generateScaledDotProductAttention() throws Exception {
        // Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V
        // Uses constant scale factor to avoid dynamic q.size(-1)
        // Input: Q, K, V tensors of shape [batch, seq_len, d_k]
        generateFixture("scaled_dot_product_attention", """
            import torch
            import torch.nn as nn
            import math

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.d_k = 16  # Head dimension (constant)
                    self.scale = 1.0 / math.sqrt(self.d_k)

                def forward(self, q, k, v):
                    # Q @ K^T -> [batch, seq, seq]
                    scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                    # Softmax over last dim
                    attn_weights = torch.softmax(scores, dim=-1)
                    # Attention @ V -> [batch, seq, d_k]
                    output = torch.matmul(attn_weights, v)
                    return output
            """, "[(2, 8, 16), (2, 8, 16), (2, 8, 16)]");
    }

    @Test
    @DisplayName("Generate: multi_head_attention")
    void generateMultiHeadAttention() throws Exception {
        // Multi-head attention with Q/K/V projections
        // Fixed dimensions: batch=2, seq=8, hidden=32, heads=4, head_dim=8
        generateFixture("multi_head_attention", """
            import torch
            import torch.nn as nn
            import math

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.scale = 1.0 / math.sqrt(8)  # 1/sqrt(head_dim)

                    # Q, K, V projections
                    self.q_proj = nn.Linear(32, 32, bias=False)
                    self.k_proj = nn.Linear(32, 32, bias=False)
                    self.v_proj = nn.Linear(32, 32, bias=False)
                    self.out_proj = nn.Linear(32, 32, bias=False)

                def forward(self, x):
                    # x: [2, 8, 32] -> Q, K, V: [2, 8, 32]
                    q = self.q_proj(x)
                    k = self.k_proj(x)
                    v = self.v_proj(x)

                    # Reshape for multi-head: [2, 8, 32] -> [2, 8, 4, 8] -> [2, 4, 8, 8]
                    q = q.reshape(2, 8, 4, 8).transpose(1, 2)
                    k = k.reshape(2, 8, 4, 8).transpose(1, 2)
                    v = v.reshape(2, 8, 4, 8).transpose(1, 2)

                    # Attention: [2, 4, 8, 8] @ [2, 4, 8, 8]^T -> [2, 4, 8, 8]
                    scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                    attn_weights = torch.softmax(scores, dim=-1)
                    attn_output = torch.matmul(attn_weights, v)

                    # Reshape back: [2, 4, 8, 8] -> [2, 8, 4, 8] -> [2, 8, 32]
                    attn_output = attn_output.transpose(1, 2).reshape(2, 8, 32)

                    # Output projection
                    return self.out_proj(attn_output)
            """, "[(2, 8, 32)]");
    }

    @Test
    @DisplayName("Generate: transformer_encoder_block")
    void generateTransformerEncoderBlock() throws Exception {
        // Complete transformer encoder block (BERT-style):
        // 1. Self-attention with residual + LayerNorm
        // 2. FFN with residual + LayerNorm
        // Fixed dimensions: batch=2, seq=8, hidden=32, heads=4, head_dim=8
        generateFixture("transformer_encoder_block", """
            import torch
            import torch.nn as nn
            import math

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.scale = 1.0 / math.sqrt(8)  # 1/sqrt(head_dim)

                    # Attention components
                    self.ln1 = nn.LayerNorm(32)
                    self.q_proj = nn.Linear(32, 32, bias=False)
                    self.k_proj = nn.Linear(32, 32, bias=False)
                    self.v_proj = nn.Linear(32, 32, bias=False)
                    self.out_proj = nn.Linear(32, 32, bias=False)

                    # FFN components
                    self.ln2 = nn.LayerNorm(32)
                    self.fc1 = nn.Linear(32, 128, bias=False)
                    self.gelu = nn.GELU()
                    self.fc2 = nn.Linear(128, 32, bias=False)

                def forward(self, x):
                    # x: [2, 8, 32]

                    # === Self-attention with residual ===
                    residual = x
                    x = self.ln1(x)

                    # Q, K, V projections
                    q = self.q_proj(x)
                    k = self.k_proj(x)
                    v = self.v_proj(x)

                    # Reshape for multi-head: [2, 8, 32] -> [2, 4, 8, 8]
                    q = q.reshape(2, 8, 4, 8).transpose(1, 2)
                    k = k.reshape(2, 8, 4, 8).transpose(1, 2)
                    v = v.reshape(2, 8, 4, 8).transpose(1, 2)

                    # Attention
                    scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                    attn_weights = torch.softmax(scores, dim=-1)
                    attn_output = torch.matmul(attn_weights, v)

                    # Reshape back: [2, 4, 8, 8] -> [2, 8, 32]
                    attn_output = attn_output.transpose(1, 2).reshape(2, 8, 32)
                    attn_output = self.out_proj(attn_output)

                    x = residual + attn_output

                    # === FFN with residual ===
                    residual = x
                    x = self.ln2(x)
                    x = self.fc1(x)
                    x = self.gelu(x)
                    x = self.fc2(x)
                    x = residual + x

                    return x
            """, "[(2, 8, 32)]");
    }

    // ==================== Tier 6: Full BERT Models ====================

    @Test
    @DisplayName("Generate: bert_squad_base")
    void generateBertSquadBase() throws Exception {
        // Full BERT-base model for SQuAD question answering
        // Architecture matches HuggingFace bert-base-uncased:
        // - 12 encoder layers
        // - 768 hidden dimensions
        // - 12 attention heads (64 per head)
        // - 3072 FFN intermediate size
        // - 30522 vocab size (using 1000 for testing)
        // - 512 max position embeddings (using 128 for testing)
        // Note: token_type_embeddings removed to avoid index range issues with random i64 inputs
        generateFixture("bert_squad_base", """
            import torch
            import torch.nn as nn
            import math

            class BertEmbeddings(nn.Module):
                def __init__(self, vocab_size, hidden_size, max_position):
                    super().__init__()
                    self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
                    self.position_embeddings = nn.Embedding(max_position, hidden_size)
                    self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

                def forward(self, input_ids, position_ids):
                    embeddings = self.word_embeddings(input_ids)
                    embeddings = embeddings + self.position_embeddings(position_ids)
                    embeddings = self.LayerNorm(embeddings)
                    return embeddings

            class BertSelfAttention(nn.Module):
                def __init__(self, hidden_size, num_heads):
                    super().__init__()
                    self.num_heads = num_heads
                    self.head_dim = hidden_size // num_heads
                    self.scale = 1.0 / math.sqrt(self.head_dim)

                    self.query = nn.Linear(hidden_size, hidden_size)
                    self.key = nn.Linear(hidden_size, hidden_size)
                    self.value = nn.Linear(hidden_size, hidden_size)

                def forward(self, hidden_states, batch_size, seq_len):
                    q = self.query(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    k = self.key(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    v = self.value(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

                    scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                    attn_probs = torch.softmax(scores, dim=-1)
                    context = torch.matmul(attn_probs, v)
                    context = context.transpose(1, 2).reshape(batch_size, seq_len, -1)
                    return context

            class BertSelfOutput(nn.Module):
                def __init__(self, hidden_size):
                    super().__init__()
                    self.dense = nn.Linear(hidden_size, hidden_size)
                    self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

                def forward(self, hidden_states, input_tensor):
                    hidden_states = self.dense(hidden_states)
                    hidden_states = self.LayerNorm(hidden_states + input_tensor)
                    return hidden_states

            class BertIntermediate(nn.Module):
                def __init__(self, hidden_size, intermediate_size):
                    super().__init__()
                    self.dense = nn.Linear(hidden_size, intermediate_size)
                    self.gelu = nn.GELU()

                def forward(self, hidden_states):
                    return self.gelu(self.dense(hidden_states))

            class BertOutput(nn.Module):
                def __init__(self, intermediate_size, hidden_size):
                    super().__init__()
                    self.dense = nn.Linear(intermediate_size, hidden_size)
                    self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

                def forward(self, hidden_states, input_tensor):
                    hidden_states = self.dense(hidden_states)
                    hidden_states = self.LayerNorm(hidden_states + input_tensor)
                    return hidden_states

            class BertLayer(nn.Module):
                def __init__(self, hidden_size, num_heads, intermediate_size):
                    super().__init__()
                    self.attention = BertSelfAttention(hidden_size, num_heads)
                    self.attention_output = BertSelfOutput(hidden_size)
                    self.intermediate = BertIntermediate(hidden_size, intermediate_size)
                    self.output = BertOutput(intermediate_size, hidden_size)

                def forward(self, hidden_states, batch_size, seq_len):
                    attention_output = self.attention(hidden_states, batch_size, seq_len)
                    attention_output = self.attention_output(attention_output, hidden_states)
                    intermediate_output = self.intermediate(attention_output)
                    layer_output = self.output(intermediate_output, attention_output)
                    return layer_output

            class BertForQuestionAnswering(nn.Module):
                def __init__(self, vocab_size=1000, hidden_size=768, num_layers=12,
                             num_heads=12, intermediate_size=3072, max_position=128):
                    super().__init__()
                    self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position)
                    self.layers = nn.ModuleList([
                        BertLayer(hidden_size, num_heads, intermediate_size)
                        for _ in range(num_layers)
                    ])
                    self.qa_outputs = nn.Linear(hidden_size, 2)

                def forward(self, input_ids, position_ids):
                    batch_size = 1
                    seq_len = 128

                    hidden_states = self.embeddings(input_ids, position_ids)
                    for layer in self.layers:
                        hidden_states = layer(hidden_states, batch_size, seq_len)
                    logits = self.qa_outputs(hidden_states)
                    return logits

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.bert = BertForQuestionAnswering()

                def forward(self, input_ids, position_ids):
                    return self.bert(input_ids, position_ids)
            """, "[(1, 128, 'i64'), (1, 128, 'i64')]");
    }

    @Test
    @DisplayName("Generate: bert_squad_small")
    void generateBertSquadSmall() throws Exception {
        // Smaller BERT for faster testing (6 layers, 384 hidden)
        // Uses only word + position embeddings (no token_type to avoid index range issues)
        generateFixture("bert_squad_small", """
            import torch
            import torch.nn as nn
            import math

            class BertEmbeddings(nn.Module):
                def __init__(self, vocab_size, hidden_size, max_position):
                    super().__init__()
                    self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
                    self.position_embeddings = nn.Embedding(max_position, hidden_size)
                    self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

                def forward(self, input_ids, position_ids):
                    embeddings = self.word_embeddings(input_ids)
                    embeddings = embeddings + self.position_embeddings(position_ids)
                    embeddings = self.LayerNorm(embeddings)
                    return embeddings

            class BertSelfAttention(nn.Module):
                def __init__(self, hidden_size, num_heads):
                    super().__init__()
                    self.num_heads = num_heads
                    self.head_dim = hidden_size // num_heads
                    self.scale = 1.0 / math.sqrt(self.head_dim)

                    self.query = nn.Linear(hidden_size, hidden_size)
                    self.key = nn.Linear(hidden_size, hidden_size)
                    self.value = nn.Linear(hidden_size, hidden_size)

                def forward(self, hidden_states, batch_size, seq_len):
                    q = self.query(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    k = self.key(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    v = self.value(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

                    scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                    attn_probs = torch.softmax(scores, dim=-1)
                    context = torch.matmul(attn_probs, v)
                    context = context.transpose(1, 2).reshape(batch_size, seq_len, -1)
                    return context

            class BertSelfOutput(nn.Module):
                def __init__(self, hidden_size):
                    super().__init__()
                    self.dense = nn.Linear(hidden_size, hidden_size)
                    self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

                def forward(self, hidden_states, input_tensor):
                    hidden_states = self.dense(hidden_states)
                    hidden_states = self.LayerNorm(hidden_states + input_tensor)
                    return hidden_states

            class BertIntermediate(nn.Module):
                def __init__(self, hidden_size, intermediate_size):
                    super().__init__()
                    self.dense = nn.Linear(hidden_size, intermediate_size)
                    self.gelu = nn.GELU()

                def forward(self, hidden_states):
                    return self.gelu(self.dense(hidden_states))

            class BertOutput(nn.Module):
                def __init__(self, intermediate_size, hidden_size):
                    super().__init__()
                    self.dense = nn.Linear(intermediate_size, hidden_size)
                    self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

                def forward(self, hidden_states, input_tensor):
                    hidden_states = self.dense(hidden_states)
                    hidden_states = self.LayerNorm(hidden_states + input_tensor)
                    return hidden_states

            class BertLayer(nn.Module):
                def __init__(self, hidden_size, num_heads, intermediate_size):
                    super().__init__()
                    self.attention = BertSelfAttention(hidden_size, num_heads)
                    self.attention_output = BertSelfOutput(hidden_size)
                    self.intermediate = BertIntermediate(hidden_size, intermediate_size)
                    self.output = BertOutput(intermediate_size, hidden_size)

                def forward(self, hidden_states, batch_size, seq_len):
                    attention_output = self.attention(hidden_states, batch_size, seq_len)
                    attention_output = self.attention_output(attention_output, hidden_states)
                    intermediate_output = self.intermediate(attention_output)
                    layer_output = self.output(intermediate_output, attention_output)
                    return layer_output

            class BertForQuestionAnswering(nn.Module):
                def __init__(self, vocab_size=1000, hidden_size=384, num_layers=6,
                             num_heads=6, intermediate_size=1536, max_position=128):
                    super().__init__()
                    self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position)
                    self.layers = nn.ModuleList([
                        BertLayer(hidden_size, num_heads, intermediate_size)
                        for _ in range(num_layers)
                    ])
                    self.qa_outputs = nn.Linear(hidden_size, 2)

                def forward(self, input_ids, position_ids):
                    batch_size = 1
                    seq_len = 64

                    hidden_states = self.embeddings(input_ids, position_ids)
                    for layer in self.layers:
                        hidden_states = layer(hidden_states, batch_size, seq_len)
                    logits = self.qa_outputs(hidden_states)
                    return logits

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.bert = BertForQuestionAnswering()

                def forward(self, input_ids, position_ids):
                    return self.bert(input_ids, position_ids)
            """, "[(1, 64, 'i64'), (1, 64, 'i64')]");
    }

    // ==================== Tier 7: Embedding Operations ====================

    @Test
    @DisplayName("Generate: bert_squad_mini")
    void generateBertSquadMini() throws Exception {
        // Mini BERT model for SQuAD question answering
        // Architecture: embeddings -> 2 encoder layers -> linear head -> start/end logits
        // Dimensions: batch=2, seq=16, hidden=64, heads=4, ffn=256, vocab=100
        generateFixture("bert_squad_mini", """
            import torch
            import torch.nn as nn
            import math

            class TransformerEncoderLayer(nn.Module):
                def __init__(self, hidden_size, num_heads, ffn_size):
                    super().__init__()
                    self.scale = 1.0 / math.sqrt(hidden_size // num_heads)
                    self.num_heads = num_heads
                    self.head_dim = hidden_size // num_heads

                    # Attention
                    self.ln1 = nn.LayerNorm(hidden_size)
                    self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                    self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                    self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                    self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

                    # FFN
                    self.ln2 = nn.LayerNorm(hidden_size)
                    self.fc1 = nn.Linear(hidden_size, ffn_size, bias=False)
                    self.gelu = nn.GELU()
                    self.fc2 = nn.Linear(ffn_size, hidden_size, bias=False)

                def forward(self, x, batch_size, seq_len):
                    # Self-attention with residual
                    residual = x
                    x = self.ln1(x)

                    q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

                    scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                    attn = torch.softmax(scores, dim=-1)
                    out = torch.matmul(attn, v)

                    out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
                    x = residual + self.out_proj(out)

                    # FFN with residual
                    residual = x
                    x = self.ln2(x)
                    x = self.fc1(x)
                    x = self.gelu(x)
                    x = self.fc2(x)
                    return residual + x

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    hidden_size = 64
                    num_heads = 4
                    ffn_size = 256
                    vocab_size = 100
                    max_seq_len = 100

                    # Embeddings
                    self.token_embed = nn.Embedding(vocab_size, hidden_size)
                    self.pos_embed = nn.Embedding(max_seq_len, hidden_size)

                    # Encoder layers
                    self.layer1 = TransformerEncoderLayer(hidden_size, num_heads, ffn_size)
                    self.layer2 = TransformerEncoderLayer(hidden_size, num_heads, ffn_size)

                    # Final layer norm
                    self.final_ln = nn.LayerNorm(hidden_size)

                    # SQuAD head: produces start and end logits
                    self.qa_head = nn.Linear(hidden_size, 2, bias=False)

                def forward(self, token_ids, position_ids):
                    batch_size = 2
                    seq_len = 16

                    # Embeddings
                    x = self.token_embed(token_ids) + self.pos_embed(position_ids)

                    # Encoder layers
                    x = self.layer1(x, batch_size, seq_len)
                    x = self.layer2(x, batch_size, seq_len)

                    # Final layer norm
                    x = self.final_ln(x)

                    # SQuAD head: [batch, seq, 2] -> start_logits, end_logits
                    logits = self.qa_head(x)
                    return logits
            """, "[(2, 16, 'i64'), (2, 16, 'i64')]");
    }

    @Test
    @DisplayName("Generate: embedding")
    void generateEmbedding() throws Exception {
        // Simple embedding lookup (vocab_size=100, embedding_dim=32)
        // Input: token indices (int64), Output: embedded vectors (float32)
        generateFixture("embedding", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embed = nn.Embedding(100, 32)

                def forward(self, x):
                    return self.embed(x)
            """, "[(4, 8, 'i64')]");
    }

    @Test
    @DisplayName("Generate: embedding_with_position")
    void generateEmbeddingWithPosition() throws Exception {
        // Token embedding + position embedding (BERT-style)
        // Both use vocab_size=100 to match the random index range (0-99)
        generateFixture("embedding_with_position", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.token_embed = nn.Embedding(100, 32)
                    self.pos_embed = nn.Embedding(100, 32)

                def forward(self, token_ids, position_ids):
                    return self.token_embed(token_ids) + self.pos_embed(position_ids)
            """, "[(2, 8, 'i64'), (2, 8, 'i64')]");
    }

    // ==================== Helper Methods ====================

    private void generateFixture(String name, String modelSource, String inputSpecs) throws Exception {
        Path outputDir = tempDir.resolve(name);
        Files.createDirectories(outputDir);

        // Write model source to temp file
        Path modelFile = tempDir.resolve(name + "_model.py");
        Files.writeString(modelFile, modelSource);

        // Invoke snakegrinder --trace-with-values
        ProcessBuilder pb = new ProcessBuilder(
            SNAKEGRINDER_BINARY.toString(),
            "--trace-with-values",
            "--source", modelFile.toString(),
            "--class", "Model",
            "--inputs", inputSpecs,
            "--seed", String.valueOf(DEFAULT_SEED),
            "--out", outputDir.toString()
        );

        pb.redirectErrorStream(true);

        Process process = pb.start();
        StringBuilder output = new StringBuilder();

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }
        }

        boolean finished = process.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        if (!finished) {
            process.destroyForcibly();
            fail("Timeout generating fixture: " + name);
        }

        int exitCode = process.exitValue();
        if (exitCode != 0) {
            fail("Failed to generate fixture '" + name + "' (exit " + exitCode + "):\n" + output);
        }

        // Verify output files exist
        assertTrue(Files.exists(outputDir.resolve("model.mlir")),
            "model.mlir not created for " + name);
        assertTrue(Files.exists(outputDir.resolve("inputs")),
            "inputs/ directory not created for " + name);
        assertTrue(Files.exists(outputDir.resolve("outputs")),
            "outputs/ directory not created for " + name);

        // Copy to fixtures directory
        Path fixtureDir = FIXTURES_OUTPUT_DIR.resolve(name);
        copyDirectory(outputDir, fixtureDir);

        System.out.println("Generated fixture: " + name + " -> " + fixtureDir);
    }

    private void copyDirectory(Path source, Path target) throws IOException {
        Files.createDirectories(target);

        Files.walk(source).forEach(sourcePath -> {
            try {
                Path targetPath = target.resolve(source.relativize(sourcePath));
                if (Files.isDirectory(sourcePath)) {
                    Files.createDirectories(targetPath);
                } else {
                    Files.copy(sourcePath, targetPath, StandardCopyOption.REPLACE_EXISTING);
                }
            } catch (IOException e) {
                throw new RuntimeException("Failed to copy: " + sourcePath, e);
            }
        });
    }
}
