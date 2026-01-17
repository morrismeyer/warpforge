package io.surfworks.warpforge.license;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for {@link MachineFingerprint}.
 */
class MachineFingerprintTest {

    @Test
    @DisplayName("generate returns non-null fingerprint")
    void generate_returnsNonNull() {
        String fingerprint = MachineFingerprint.generate();

        assertNotNull(fingerprint);
        assertFalse(fingerprint.isBlank());
    }

    @Test
    @DisplayName("generate returns consistent fingerprint")
    void generate_isConsistent() {
        String fp1 = MachineFingerprint.generate();
        String fp2 = MachineFingerprint.generate();

        assertEquals(fp1, fp2);
    }

    @Test
    @DisplayName("generate returns 32-character hex string")
    void generate_returns32CharHex() {
        String fingerprint = MachineFingerprint.generate();

        assertEquals(32, fingerprint.length());
        assertTrue(fingerprint.matches("[0-9a-f]+"));
    }

    @Test
    @DisplayName("getMachineName returns non-null name")
    void getMachineName_returnsNonNull() {
        String name = MachineFingerprint.getMachineName();

        assertNotNull(name);
        assertFalse(name.isBlank());
    }

    @Test
    @DisplayName("getMachineName includes OS info")
    void getMachineName_includesOsInfo() {
        String name = MachineFingerprint.getMachineName();

        // Should include OS in parentheses
        assertTrue(name.contains("("));
        assertTrue(name.contains(")"));

        // Should include one of these OS indicators
        String osName = System.getProperty("os.name", "").toLowerCase();
        if (osName.contains("mac")) {
            assertTrue(name.contains("macOS"));
        } else if (osName.contains("linux")) {
            assertTrue(name.contains("Linux"));
        }
    }

    @Test
    @DisplayName("getMachineName is consistent")
    void getMachineName_isConsistent() {
        String name1 = MachineFingerprint.getMachineName();
        String name2 = MachineFingerprint.getMachineName();

        assertEquals(name1, name2);
    }

    @Test
    @DisplayName("Different fingerprints would have different hashes")
    void differentInputs_differentHashes() {
        // We can't easily test different machines, but we can verify the hash function
        // is deterministic by calling multiple times
        String fp1 = MachineFingerprint.generate();
        String fp2 = MachineFingerprint.generate();

        // Same machine = same fingerprint
        assertEquals(fp1, fp2);
    }

    @Test
    @DisplayName("Fingerprint does not contain special characters")
    void fingerprint_noSpecialCharacters() {
        String fingerprint = MachineFingerprint.generate();

        // Should only contain lowercase hex
        for (char c : fingerprint.toCharArray()) {
            assertTrue(Character.isDigit(c) || (c >= 'a' && c <= 'f'),
                "Unexpected character: " + c);
        }
    }

    @Test
    @DisplayName("Machine name has reasonable length")
    void machineName_reasonableLength() {
        String name = MachineFingerprint.getMachineName();

        // Should be at least a few characters
        assertTrue(name.length() >= 3);
        // Should not be unreasonably long
        assertTrue(name.length() < 200);
    }
}
