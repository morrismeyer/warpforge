package io.surfworks.snakeburger.cli;

import java.lang.reflect.Method;
import java.util.Optional;

import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp;

// TODO: Rename this class to BabylonVersion and change CLI command from -hello to -babylon
//       Add version detection using `git describe` from Babylon repo:
//         - Format: jdk-26+25-852-gfbff3d4a833 (jdk-{version}+{build}-{commits}-g{sha})
//         - Store in JAR manifest as "Babylon-Version" for codegen compatibility checks
//         - WarpForge can verify generated JARs are compatible with current Babylon
//       See: ../babylon && git describe --tags

public final class BabylonHello {

    private BabylonHello() {}

    @Reflect
    static double sub(double a, double b) {
        return a - b;
    }

    public static String helloModelText() {
        try {
            Method m = BabylonHello.class.getDeclaredMethod("sub", double.class, double.class);
            Optional<CoreOp.FuncOp> oModel = Op.ofMethod(m);
            CoreOp.FuncOp model = oModel.orElseThrow();
            return model.toText();
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("Failed to reflect method", e);
        }
    }
}
