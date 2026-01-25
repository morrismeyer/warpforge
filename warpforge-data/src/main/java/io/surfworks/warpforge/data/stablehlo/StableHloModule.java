package io.surfworks.warpforge.data.stablehlo;

import io.surfworks.warpforge.data.stablehlo.StableHloOps.Operation;
import io.surfworks.warpforge.data.stablehlo.StableHloTypes.FunctionType;
import io.surfworks.warpforge.data.stablehlo.StableHloTypes.TensorType;
import io.surfworks.warpforge.data.stablehlo.StableHloTypes.Type;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Represents a StableHLO module that can be loaded from or exported to MLIR text.
 *
 * <p>This is a simplified representation for benchmark model export. For full
 * parsing capabilities, use the snakeburger-core StableHloParser.
 *
 * <p>Example usage:
 * <pre>{@code
 * // Build a module programmatically
 * StableHloModule module = StableHloModule.builder("my_model")
 *     .addFunction(StableHloFunction.builder("forward")
 *         .addArgument("arg0", TensorType.of(ScalarType.F32, 1, 768))
 *         .addArgument("arg1", TensorType.of(ScalarType.F32, 768, 768))
 *         .addOperation(new DotOp(TensorType.of(ScalarType.F32, 1, 768)))
 *         .addReturn(TensorType.of(ScalarType.F32, 1, 768))
 *         .build())
 *     .build();
 *
 * // Export to MLIR
 * String mlir = module.toMlir();
 *
 * // Save to file
 * module.writeTo(Path.of("model.mlir"));
 * }</pre>
 */
public final class StableHloModule {

    private final String name;
    private final List<StableHloFunction> functions;

    private StableHloModule(String name, List<StableHloFunction> functions) {
        this.name = name;
        this.functions = List.copyOf(functions);
    }

    /**
     * Module name.
     */
    public String name() {
        return name;
    }

    /**
     * Functions in this module.
     */
    public List<StableHloFunction> functions() {
        return functions;
    }

    /**
     * Get a function by name.
     */
    public StableHloFunction function(String name) {
        return functions.stream()
                .filter(f -> f.name().equals(name))
                .findFirst()
                .orElse(null);
    }

    /**
     * Get the main/entry function (typically named "forward" or "main").
     */
    public StableHloFunction mainFunction() {
        StableHloFunction forward = function("forward");
        if (forward != null) return forward;
        StableHloFunction main = function("main");
        if (main != null) return main;
        return functions.isEmpty() ? null : functions.get(0);
    }

    /**
     * Convert to MLIR text format.
     */
    public String toMlir() {
        StringBuilder sb = new StringBuilder();
        sb.append("module @").append(name).append(" {\n");
        for (StableHloFunction func : functions) {
            sb.append(func.toMlir(2));
            sb.append("\n");
        }
        sb.append("}\n");
        return sb.toString();
    }

    /**
     * Write MLIR to file.
     */
    public void writeTo(Path path) throws IOException {
        Files.writeString(path, toMlir());
    }

    /**
     * Load module from MLIR file (basic parsing).
     *
     * <p>This provides basic parsing for simple modules. For full parsing
     * with error recovery and validation, use snakeburger-core.
     */
    public static StableHloModule loadFrom(Path path) throws IOException {
        return parse(Files.readString(path));
    }

    /**
     * Parse module from MLIR text (basic parsing).
     */
    public static StableHloModule parse(String mlir) throws IOException {
        return new MlirParser(mlir).parseModule();
    }

    /**
     * Create a new module builder.
     */
    public static Builder builder(String name) {
        return new Builder(name);
    }

    /**
     * Builder for StableHloModule.
     */
    public static final class Builder {
        private final String name;
        private final List<StableHloFunction> functions = new ArrayList<>();

        private Builder(String name) {
            this.name = name;
        }

        public Builder addFunction(StableHloFunction function) {
            this.functions.add(function);
            return this;
        }

        public Builder addFunction(StableHloFunction.Builder functionBuilder) {
            return addFunction(functionBuilder.build());
        }

        public StableHloModule build() {
            return new StableHloModule(name, functions);
        }
    }

    /**
     * Represents a function within a StableHLO module.
     */
    public static final class StableHloFunction {
        private final String name;
        private final List<Argument> arguments;
        private final List<OpEntry> operations;
        private final List<TensorType> returnTypes;
        private final boolean isPublic;

        private StableHloFunction(String name, List<Argument> arguments,
                                  List<OpEntry> operations, List<TensorType> returnTypes,
                                  boolean isPublic) {
            this.name = name;
            this.arguments = List.copyOf(arguments);
            this.operations = List.copyOf(operations);
            this.returnTypes = List.copyOf(returnTypes);
            this.isPublic = isPublic;
        }

        public String name() {
            return name;
        }

        public List<Argument> arguments() {
            return arguments;
        }

        public List<OpEntry> operations() {
            return operations;
        }

        public List<TensorType> returnTypes() {
            return returnTypes;
        }

        public boolean isPublic() {
            return isPublic;
        }

        public FunctionType type() {
            List<Type> inputTypes = arguments.stream()
                    .map(Argument::type)
                    .map(t -> (Type) t)
                    .toList();
            List<Type> resultTypes = returnTypes.stream()
                    .map(t -> (Type) t)
                    .toList();
            return new FunctionType(inputTypes, resultTypes);
        }

        public String toMlir(int indent) {
            String pad = " ".repeat(indent);
            StringBuilder sb = new StringBuilder();

            // Function signature
            sb.append(pad).append("func.func ");
            if (isPublic) sb.append("public ");
            sb.append("@").append(name).append("(");

            for (int i = 0; i < arguments.size(); i++) {
                if (i > 0) sb.append(", ");
                Argument arg = arguments.get(i);
                sb.append("%").append(arg.name).append(": ").append(arg.type.toMlir());
            }

            sb.append(") -> (");
            for (int i = 0; i < returnTypes.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append(returnTypes.get(i).toMlir());
            }
            sb.append(") {\n");

            // Operations
            for (OpEntry entry : operations) {
                sb.append(pad).append("  ");
                sb.append(entry.operation.toMlir(entry.resultNames, entry.operandNames));
                sb.append("\n");
            }

            sb.append(pad).append("}");
            return sb.toString();
        }

        public static Builder builder(String name) {
            return new Builder(name);
        }

        public record Argument(String name, TensorType type) {}

        public record OpEntry(Operation operation, List<String> resultNames, List<String> operandNames) {}

        public static final class Builder {
            private final String name;
            private final List<Argument> arguments = new ArrayList<>();
            private final List<OpEntry> operations = new ArrayList<>();
            private final List<TensorType> returnTypes = new ArrayList<>();
            private boolean isPublic = true;
            private int valueCounter = 0;

            private Builder(String name) {
                this.name = name;
            }

            public Builder setPublic(boolean isPublic) {
                this.isPublic = isPublic;
                return this;
            }

            public Builder addArgument(String name, TensorType type) {
                this.arguments.add(new Argument(name, type));
                return this;
            }

            /**
             * Add an operation with automatic result naming.
             *
             * @param operation The operation
             * @param operandNames Names of operands (arguments or previous results)
             * @return The result name(s) for use in subsequent operations
             */
            public List<String> addOperation(Operation operation, List<String> operandNames) {
                List<String> resultNames = new ArrayList<>();
                for (int i = 0; i < operation.resultTypes().size(); i++) {
                    resultNames.add(String.valueOf(valueCounter++));
                }
                this.operations.add(new OpEntry(operation, resultNames, operandNames));
                return resultNames;
            }

            public Builder addReturn(TensorType... types) {
                this.returnTypes.addAll(List.of(types));
                return this;
            }

            /**
             * Add return operation.
             *
             * @param returnOperands Names of values to return
             */
            public Builder addReturnOp(List<String> returnOperands) {
                // Add return types from the operation
                // Note: return types should be set via addReturn before calling build
                this.operations.add(new OpEntry(
                        new StableHloOps.ReturnOp(returnTypes),
                        List.of(),
                        returnOperands
                ));
                return this;
            }

            public StableHloFunction build() {
                return new StableHloFunction(name, arguments, operations, returnTypes, isPublic);
            }
        }
    }

    /**
     * Basic MLIR parser for loading StableHLO modules.
     */
    private static class MlirParser {
        private final BufferedReader reader;
        private String currentLine;
        private int lineNumber = 0;

        private static final Pattern MODULE_PATTERN = Pattern.compile("module\\s+@(\\w+)\\s*\\{");
        private static final Pattern FUNC_PATTERN = Pattern.compile(
                "func\\.func\\s+(public\\s+)?@(\\w+)\\(([^)]*)\\)\\s*->\\s*\\(([^)]*)\\)");
        private static final Pattern ARG_PATTERN = Pattern.compile("%([\\w.]+):\\s*(tensor<[^>]+>)");
        private static final Pattern TYPE_PATTERN = Pattern.compile("tensor<[^>]+>");

        MlirParser(String mlir) {
            this.reader = new BufferedReader(new StringReader(mlir));
        }

        StableHloModule parseModule() throws IOException {
            String moduleName = "main";
            List<StableHloFunction> functions = new ArrayList<>();

            while ((currentLine = reader.readLine()) != null) {
                lineNumber++;
                String trimmed = currentLine.trim();

                Matcher moduleMatcher = MODULE_PATTERN.matcher(trimmed);
                if (moduleMatcher.find()) {
                    moduleName = moduleMatcher.group(1);
                    continue;
                }

                Matcher funcMatcher = FUNC_PATTERN.matcher(trimmed);
                if (funcMatcher.find()) {
                    functions.add(parseFunction(funcMatcher));
                }
            }

            return new StableHloModule(moduleName, functions);
        }

        private StableHloFunction parseFunction(Matcher funcMatcher) throws IOException {
            boolean isPublic = funcMatcher.group(1) != null;
            String funcName = funcMatcher.group(2);
            String argsStr = funcMatcher.group(3);
            String returnStr = funcMatcher.group(4);

            // Parse arguments
            List<StableHloFunction.Argument> arguments = new ArrayList<>();
            Matcher argMatcher = ARG_PATTERN.matcher(argsStr);
            while (argMatcher.find()) {
                String argName = argMatcher.group(1);
                String typeStr = argMatcher.group(2);
                arguments.add(new StableHloFunction.Argument(argName, TensorType.fromMlir(typeStr)));
            }

            // Parse return types
            List<TensorType> returnTypes = new ArrayList<>();
            Matcher typeMatcher = TYPE_PATTERN.matcher(returnStr);
            while (typeMatcher.find()) {
                returnTypes.add(TensorType.fromMlir(typeMatcher.group()));
            }

            // Skip function body for now (basic parsing)
            // A full parser would parse operations here
            List<StableHloFunction.OpEntry> operations = new ArrayList<>();

            // Skip until closing brace
            int braceCount = 1;
            while ((currentLine = reader.readLine()) != null && braceCount > 0) {
                lineNumber++;
                for (char c : currentLine.toCharArray()) {
                    if (c == '{') braceCount++;
                    else if (c == '}') braceCount--;
                }
            }

            return new StableHloFunction(funcName, arguments, operations, returnTypes, isPublic);
        }
    }
}
