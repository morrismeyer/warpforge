package io.surfworks.warpforge.data.tokenizer;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TokenizerTest {

    @Nested
    class WhitespaceTokenizerTests {

        @Test
        void testBasicEncoding() {
            Map<String, Integer> vocab = createTestVocab();
            Tokenizer tokenizer = Tokenizer.whitespace(vocab);

            Tokenizer.EncodedInput encoded = tokenizer.encode("hello world");

            assertNotNull(encoded.inputIds());
            assertNotNull(encoded.attentionMask());
            assertTrue(encoded.length() > 0);
        }

        @Test
        void testEncodingWithSpecialTokens() {
            Map<String, Integer> vocab = createTestVocab();
            Tokenizer tokenizer = Tokenizer.whitespace(vocab);

            Tokenizer.EncodedInput encoded = tokenizer.encode("hello world");

            // Should have [CLS] hello world [SEP]
            assertEquals(4, encoded.length());

            // First token should be CLS
            assertEquals(tokenizer.specialTokens().clsTokenId(), encoded.inputIds()[0]);

            // Last token should be SEP
            assertEquals(tokenizer.specialTokens().sepTokenId(), encoded.inputIds()[encoded.length() - 1]);
        }

        @Test
        void testDecoding() {
            Map<String, Integer> vocab = createTestVocab();
            Tokenizer tokenizer = Tokenizer.whitespace(vocab);

            Tokenizer.EncodedInput encoded = tokenizer.encode("hello world");
            String decoded = tokenizer.decode(encoded.inputIds());

            assertEquals("hello world", decoded);
        }

        @Test
        void testUnknownTokens() {
            Map<String, Integer> vocab = createTestVocab();
            Tokenizer tokenizer = Tokenizer.whitespace(vocab);

            Tokenizer.EncodedInput encoded = tokenizer.encode("hello unknown");

            // "unknown" should be mapped to UNK token
            int unkId = tokenizer.specialTokens().unkTokenId();
            boolean hasUnk = false;
            for (int id : encoded.inputIds()) {
                if (id == unkId) {
                    hasUnk = true;
                    break;
                }
            }
            assertTrue(hasUnk, "Unknown word should be mapped to UNK token");
        }

        @Test
        void testEncodePair() {
            Map<String, Integer> vocab = createTestVocab();
            Tokenizer tokenizer = Tokenizer.whitespace(vocab);

            Tokenizer.EncodedInput encoded = tokenizer.encodePair("hello", "world");

            // Should have [CLS] hello [SEP] world [SEP]
            assertEquals(5, encoded.length());
        }

        @Test
        void testBatchEncoding() {
            Map<String, Integer> vocab = createTestVocab();
            Tokenizer tokenizer = Tokenizer.whitespace(vocab);

            List<Tokenizer.EncodedInput> batch = tokenizer.encodeBatch(
                    List.of("hello", "hello world", "world")
            );

            assertEquals(3, batch.size());
        }

        @Test
        void testBatchEncodingPadded() {
            Map<String, Integer> vocab = createTestVocab();
            Tokenizer tokenizer = Tokenizer.whitespace(vocab);

            Tokenizer.BatchEncodedInput batch = tokenizer.encodeBatchPadded(
                    List.of("hello", "hello world", "world"),
                    Tokenizer.EncodeOptions.forBatch()
            );

            assertEquals(3, batch.batchSize());

            // All sequences should have same length (longest)
            int maxLen = batch.maxLength();
            for (int[] ids : batch.inputIds()) {
                assertEquals(maxLen, ids.length);
            }
        }

        @Test
        void testTruncation() {
            Map<String, Integer> vocab = createTestVocab();
            Tokenizer tokenizer = Tokenizer.whitespace(vocab);

            Tokenizer.EncodeOptions options = Tokenizer.EncodeOptions.defaults()
                    .withMaxLength(3)
                    .withTruncation(true);

            Tokenizer.EncodedInput encoded = tokenizer.encode("hello world foo bar", options);

            assertEquals(3, encoded.length());
        }

        @Test
        void testPadding() {
            Map<String, Integer> vocab = createTestVocab();
            Tokenizer tokenizer = Tokenizer.whitespace(vocab);

            Tokenizer.EncodeOptions options = Tokenizer.EncodeOptions.withPadding(10);

            Tokenizer.EncodedInput encoded = tokenizer.encode("hello", options);

            assertEquals(10, encoded.length());

            // Padded positions should have attention mask = 0
            int realTokenCount = 0;
            for (int mask : encoded.attentionMask()) {
                if (mask == 1) realTokenCount++;
            }
            assertTrue(realTokenCount < 10);
        }

        @Test
        void testVocabSize() {
            Map<String, Integer> vocab = createTestVocab();
            Tokenizer tokenizer = Tokenizer.whitespace(vocab);

            assertTrue(tokenizer.vocabSize() >= vocab.size());
        }

        @Test
        void testTokenToId() {
            Map<String, Integer> vocab = createTestVocab();
            Tokenizer tokenizer = Tokenizer.whitespace(vocab);

            assertEquals(5, tokenizer.tokenToId("hello"));
            assertEquals(tokenizer.specialTokens().unkTokenId(), tokenizer.tokenToId("notinvocab"));
        }

        @Test
        void testIdToToken() {
            Map<String, Integer> vocab = createTestVocab();
            Tokenizer tokenizer = Tokenizer.whitespace(vocab);

            assertEquals("hello", tokenizer.idToToken(5));
        }

        @Test
        void testMaxLength() {
            Map<String, Integer> vocab = createTestVocab();
            Tokenizer tokenizer = Tokenizer.whitespace(vocab);

            assertEquals(512, tokenizer.maxLength());
        }

        private Map<String, Integer> createTestVocab() {
            Map<String, Integer> vocab = new HashMap<>();
            vocab.put("[PAD]", 0);
            vocab.put("[UNK]", 1);
            vocab.put("[CLS]", 2);
            vocab.put("[SEP]", 3);
            vocab.put("[MASK]", 4);
            vocab.put("hello", 5);
            vocab.put("world", 6);
            vocab.put("foo", 7);
            vocab.put("bar", 8);
            return vocab;
        }
    }

    @Nested
    class EncodeOptionsTests {

        @Test
        void testDefaultOptions() {
            Tokenizer.EncodeOptions options = Tokenizer.EncodeOptions.defaults();

            assertEquals(512, options.maxLength());
            assertTrue(options.truncation());
            assertTrue(options.addSpecialTokens());
        }

        @Test
        void testWithPadding() {
            Tokenizer.EncodeOptions options = Tokenizer.EncodeOptions.withPadding(256);

            assertEquals(256, options.maxLength());
            assertTrue(options.padding());
            assertEquals(Tokenizer.PaddingStrategy.MAX_LENGTH, options.paddingStrategy());
        }

        @Test
        void testForBatch() {
            Tokenizer.EncodeOptions options = Tokenizer.EncodeOptions.forBatch();

            assertTrue(options.padding());
            assertEquals(Tokenizer.PaddingStrategy.LONGEST, options.paddingStrategy());
        }

        @Test
        void testChainedModifications() {
            Tokenizer.EncodeOptions options = Tokenizer.EncodeOptions.defaults()
                    .withMaxLength(128)
                    .withTruncation(false)
                    .withPadding(true);

            assertEquals(128, options.maxLength());
        }
    }

    @Nested
    class SpecialTokensTests {

        @Test
        void testDefaults() {
            Tokenizer.SpecialTokens tokens = Tokenizer.SpecialTokens.defaults();

            assertEquals(0, tokens.padTokenId());
            assertEquals(101, tokens.clsTokenId());
            assertEquals(102, tokens.sepTokenId());
            assertEquals(100, tokens.unkTokenId());
        }

        @Test
        void testGptTokens() {
            Tokenizer.SpecialTokens tokens = Tokenizer.SpecialTokens.gpt();

            assertEquals(-1, tokens.clsTokenId());
            assertEquals(-1, tokens.sepTokenId());
            assertEquals(50256, tokens.bosTokenId());
            assertEquals(50256, tokens.eosTokenId());
        }
    }

    @Nested
    class EncodedInputTests {

        @Test
        void testOf() {
            int[] ids = {1, 2, 3};
            int[] mask = {1, 1, 1};

            Tokenizer.EncodedInput input = Tokenizer.EncodedInput.of(ids, mask);

            assertArrayEquals(ids, input.inputIds());
            assertArrayEquals(mask, input.attentionMask());
            assertEquals(3, input.length());
        }
    }

    @Nested
    class BatchEncodedInputTests {

        @Test
        void testOf() {
            int[][] ids = {{1, 2}, {3, 4}};
            int[][] mask = {{1, 1}, {1, 1}};

            Tokenizer.BatchEncodedInput batch = Tokenizer.BatchEncodedInput.of(ids, mask);

            assertEquals(2, batch.batchSize());
            assertEquals(2, batch.maxLength());
        }
    }

    @Nested
    @Tag("integration")
    class HuggingFaceTokenizerTests {

        @Test
        void testLoadBertTokenizer() throws Exception {
            Tokenizer tokenizer = Tokenizer.fromHuggingFace("bert-base-uncased");

            assertNotNull(tokenizer);
            assertTrue(tokenizer.vocabSize() > 0);

            Tokenizer.EncodedInput encoded = tokenizer.encode("Hello, world!");
            assertTrue(encoded.length() > 0);

            String decoded = tokenizer.decode(encoded.inputIds());
            assertTrue(decoded.contains("hello"));

            tokenizer.close();
        }

        @Test
        void testLoadGpt2Tokenizer() throws Exception {
            Tokenizer tokenizer = Tokenizer.fromHuggingFace("gpt2");

            assertNotNull(tokenizer);
            assertTrue(tokenizer.vocabSize() > 0);

            Tokenizer.EncodedInput encoded = tokenizer.encode("Hello, world!");
            assertTrue(encoded.length() > 0);

            tokenizer.close();
        }
    }
}
