package io.surfworks.snakeburger.cli;

import java.lang.reflect.Method;
import java.util.Optional;

import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp;

/**
 * Babylon version and capability detection.
 *
 * <p>This class verifies that jdk.incubator.code is working correctly
 * and can be used to extract version information for codegen compatibility.
 *
 * <p>Future: Add version detection using `git describe` from Babylon repo:
 * Format: jdk-26+25-852-gfbff3d4a833 (jdk-{version}+{build}-{commits}-g{sha})
 * Store in JAR manifest as "Babylon-Version" for compatibility checks.
 */
public final class BabylonVersion {

    private BabylonVersion() {}

    @Reflect
    static double sub(double a, double b) {
        return a - b;
    }

    /**
     * Returns the code model text for a simple reflected method.
     * This verifies Babylon code reflection is working.
     */
    public static String getCodeModelText() {
        try {
            Method m = BabylonVersion.class.getDeclaredMethod("sub", double.class, double.class);
            Optional<CoreOp.FuncOp> oModel = Op.ofMethod(m);
            CoreOp.FuncOp model = oModel.orElseThrow();
            return model.toText();
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("Failed to reflect method", e);
        }
    }

    /**
     * Checks if Babylon code reflection is available and working.
     */
    public static boolean isAvailable() {
        try {
            getCodeModelText();
            return true;
        } catch (Exception | Error e) {
            return false;
        }
    }
}
