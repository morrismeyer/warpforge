package io.surfworks.warpforge.runner;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for ExecutionMode enum.
 */
class ExecutionModeTest {

    @Test
    void testJvmModeExists() {
        assertEquals("JVM", ExecutionMode.JVM.name());
    }

    @Test
    void testEspressoModeExists() {
        assertEquals("ESPRESSO", ExecutionMode.ESPRESSO.name());
    }

    @Test
    void testNativeModeExists() {
        assertEquals("NATIVE", ExecutionMode.NATIVE.name());
    }

    @Test
    void testThreeModesTotal() {
        assertEquals(3, ExecutionMode.values().length);
    }

    // ========================
    // fromString() tests
    // ========================

    @ParameterizedTest
    @CsvSource({
        "jvm, JVM",
        "JVM, JVM",
        "Jvm, JVM",
        "espresso, ESPRESSO",
        "ESPRESSO, ESPRESSO",
        "Espresso, ESPRESSO",
        "native, NATIVE",
        "NATIVE, NATIVE",
        "Native, NATIVE"
    })
    void testFromStringValidModes(String input, String expectedName) {
        ExecutionMode mode = ExecutionMode.fromString(input);
        assertEquals(expectedName, mode.name());
    }

    @Test
    void testFromStringJvmLowercase() {
        assertEquals(ExecutionMode.JVM, ExecutionMode.fromString("jvm"));
    }

    @Test
    void testFromStringJvmUppercase() {
        assertEquals(ExecutionMode.JVM, ExecutionMode.fromString("JVM"));
    }

    @Test
    void testFromStringJvmMixedCase() {
        assertEquals(ExecutionMode.JVM, ExecutionMode.fromString("Jvm"));
    }

    @Test
    void testFromStringEspressoLowercase() {
        assertEquals(ExecutionMode.ESPRESSO, ExecutionMode.fromString("espresso"));
    }

    @Test
    void testFromStringEspressoUppercase() {
        assertEquals(ExecutionMode.ESPRESSO, ExecutionMode.fromString("ESPRESSO"));
    }

    @Test
    void testFromStringNativeLowercase() {
        assertEquals(ExecutionMode.NATIVE, ExecutionMode.fromString("native"));
    }

    @Test
    void testFromStringNativeUppercase() {
        assertEquals(ExecutionMode.NATIVE, ExecutionMode.fromString("NATIVE"));
    }

    @ParameterizedTest
    @ValueSource(strings = {
        "unknown",
        "jvm2",
        "espresso2",
        "native2",
        "cpu",
        "gpu",
        "",
        "  ",
        "j v m",
        "jvm ",
        " jvm"
    })
    void testFromStringInvalidThrows(String invalidInput) {
        assertThrows(IllegalArgumentException.class, () -> {
            ExecutionMode.fromString(invalidInput);
        });
    }

    @Test
    void testFromStringNullThrows() {
        assertThrows(NullPointerException.class, () -> {
            ExecutionMode.fromString(null);
        });
    }

    // ========================
    // isSupported() tests
    // ========================

    @Test
    void testJvmIsSupported() {
        assertTrue(ExecutionMode.JVM.isSupported());
    }

    @Test
    void testEspressoIsSupported() {
        assertTrue(ExecutionMode.ESPRESSO.isSupported());
    }

    @Test
    void testNativeIsSupported() {
        assertTrue(ExecutionMode.NATIVE.isSupported());
    }

    @Test
    void testAllModesAreSupported() {
        for (ExecutionMode mode : ExecutionMode.values()) {
            assertTrue(mode.isSupported(), mode + " should be supported");
        }
    }

    // ========================
    // Enum behavior tests
    // ========================

    @Test
    void testValueOf() {
        assertEquals(ExecutionMode.JVM, ExecutionMode.valueOf("JVM"));
        assertEquals(ExecutionMode.ESPRESSO, ExecutionMode.valueOf("ESPRESSO"));
        assertEquals(ExecutionMode.NATIVE, ExecutionMode.valueOf("NATIVE"));
    }

    @Test
    void testValueOfInvalidThrows() {
        assertThrows(IllegalArgumentException.class, () -> {
            ExecutionMode.valueOf("invalid");
        });
    }

    @Test
    void testOrdinalValues() {
        // Verify ordinal positions are stable
        assertEquals(0, ExecutionMode.JVM.ordinal());
        assertEquals(1, ExecutionMode.ESPRESSO.ordinal());
        assertEquals(2, ExecutionMode.NATIVE.ordinal());
    }

    @Test
    void testEnumIterationOrder() {
        ExecutionMode[] modes = ExecutionMode.values();
        assertEquals(ExecutionMode.JVM, modes[0]);
        assertEquals(ExecutionMode.ESPRESSO, modes[1]);
        assertEquals(ExecutionMode.NATIVE, modes[2]);
    }
}
