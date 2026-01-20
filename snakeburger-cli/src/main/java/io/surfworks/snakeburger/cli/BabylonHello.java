package io.surfworks.snakeburger.cli;

import java.lang.reflect.Method;
import java.util.Optional;

import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp;

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
