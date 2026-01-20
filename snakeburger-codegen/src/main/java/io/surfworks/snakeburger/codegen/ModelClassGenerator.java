package io.surfworks.snakeburger.codegen;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloParser;
import io.surfworks.warpforge.codegen.api.AbstractCompiledModel;
import io.surfworks.warpforge.codegen.api.ModelMetadata;

import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;

/**
 * Generates bytecode for a compiled model class.
 *
 * <p>The generated class extends {@link AbstractCompiledModel} and provides
 * data methods that return the operations, indices, and metadata.
 *
 * <p>Operations are stored by embedding the MLIR source and re-parsing at
 * class load time. This approach avoids serialization complexity with the
 * sealed type hierarchy.
 */
public final class ModelClassGenerator {

    public static final String GENERATOR_VERSION = "1.0.0";

    private static final String ABSTRACT_MODEL_INTERNAL = Type.getInternalName(AbstractCompiledModel.class);
    private static final String METADATA_INTERNAL = Type.getInternalName(ModelMetadata.class);
    private static final String METADATA_DESC = Type.getDescriptor(ModelMetadata.class);
    private static final String LIST_DESC = "Ljava/util/List;";
    private static final String PARSER_INTERNAL = Type.getInternalName(StableHloParser.class);
    private static final String MODULE_INTERNAL = Type.getInternalName(StableHloAst.Module.class);
    private static final String FUNCTION_INTERNAL = Type.getInternalName(StableHloAst.Function.class);

    private ModelClassGenerator() {} // Utility class

    /**
     * Result of generating a model class.
     */
    public record GeneratedClass(
        String className,
        byte[] bytecode,
        ModelMetadata metadata
    ) {}

    /**
     * Generates a model class from a compiled function.
     *
     * @param className     The fully qualified class name (e.g., "io.surfworks.warpforge.generated.Model")
     * @param function      The StableHLO function to compile
     * @param mlirSource    The original MLIR source (for runtime parsing)
     * @return The generated class bytecode and metadata
     * @throws CodegenException if generation fails
     */
    public static GeneratedClass generate(
            String className,
            StableHloAst.Function function,
            String mlirSource) throws CodegenException {

        // Analyze the function to extract indices
        FunctionAnalysis analysis = analyzeFunction(function);

        // Compute source hash
        String sourceHash = computeHash(mlirSource);

        // Create metadata
        ModelMetadata metadata = ModelMetadata.create(
            function.name(),
            sourceHash,
            GENERATOR_VERSION
        );

        // Generate the bytecode
        byte[] bytecode = generateBytecode(className, analysis, metadata, mlirSource, function.name());

        return new GeneratedClass(className, bytecode, metadata);
    }

    /**
     * Analyzes a function to extract operation and index information.
     */
    private static FunctionAnalysis analyzeFunction(StableHloAst.Function function) {
        CodegenContext ctx = new CodegenContext();

        // Register function arguments
        int[] inputIndices = new int[function.arguments().size()];
        for (int i = 0; i < function.arguments().size(); i++) {
            var arg = function.arguments().get(i);
            inputIndices[i] = ctx.registerValue(
                arg.name(),
                CodegenContext.tensorSpecFromType(arg.type())
            );
        }

        // Process operations
        List<int[]> opInputsList = new ArrayList<>();
        List<int[]> opOutputsList = new ArrayList<>();
        int[] outputIndices = null;

        for (var op : function.body()) {
            ctx.recordOperation(op);

            if (op instanceof StableHloAst.ReturnOp returnOp) {
                // Return op - capture output indices
                outputIndices = resolveOperandIndices(returnOp.operands(), ctx);
                opInputsList.add(outputIndices);
                opOutputsList.add(new int[0]);
            } else {
                // Regular operation
                int[] opInputs = resolveOperandIndices(op.operands(), ctx);
                int[] opOutputs = registerResults(op.results(), op.tensorResultType(), ctx);
                opInputsList.add(opInputs);
                opOutputsList.add(opOutputs);
            }
        }

        if (outputIndices == null) {
            throw new IllegalArgumentException("Function missing return statement");
        }

        int[][] opInputs = opInputsList.toArray(new int[0][]);
        int[][] opOutputs = opOutputsList.toArray(new int[0][]);

        return new FunctionAnalysis(
            inputIndices,
            outputIndices,
            ctx.tensorCount(),
            opInputs,
            opOutputs
        );
    }

    private static int[] resolveOperandIndices(List<StableHloAst.Value> operands, CodegenContext ctx) {
        int[] indices = new int[operands.size()];
        for (int i = 0; i < operands.size(); i++) {
            indices[i] = ctx.getValueIndex(operands.get(i).name());
        }
        return indices;
    }

    private static int[] registerResults(
            List<StableHloAst.Value> results,
            StableHloAst.TensorType resultType,
            CodegenContext ctx) {
        int[] indices = new int[results.size()];
        for (int i = 0; i < results.size(); i++) {
            var result = results.get(i);
            var spec = getTensorSpec(result, resultType);
            indices[i] = ctx.registerValue(result.name(), spec);
        }
        return indices;
    }

    private static io.surfworks.warpforge.core.tensor.TensorSpec getTensorSpec(
            StableHloAst.Value result,
            StableHloAst.TensorType fallbackType) {
        if (result.type() instanceof StableHloAst.TensorType tt) {
            return CodegenContext.tensorSpecFromType(tt);
        }
        if (fallbackType != null) {
            return CodegenContext.tensorSpecFromType(fallbackType);
        }
        throw new IllegalArgumentException("Cannot determine type for result: " + result.name());
    }

    /**
     * Generates the bytecode for the model class.
     */
    private static byte[] generateBytecode(
            String className,
            FunctionAnalysis analysis,
            ModelMetadata metadata,
            String mlirSource,
            String functionName) throws CodegenException {

        String internalName = className.replace('.', '/');

        ClassWriter cw = new ClassWriter(ClassWriter.COMPUTE_FRAMES | ClassWriter.COMPUTE_MAXS);

        // Class declaration: public final class <Name> extends AbstractCompiledModel
        cw.visit(
            Opcodes.V21,
            Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL | Opcodes.ACC_SUPER,
            internalName,
            null,
            ABSTRACT_MODEL_INTERNAL,
            null
        );

        // Static fields
        cw.visitField(
            Opcodes.ACC_PRIVATE | Opcodes.ACC_STATIC | Opcodes.ACC_FINAL,
            "MLIR_SOURCE",
            "Ljava/lang/String;",
            null,
            null
        ).visitEnd();

        cw.visitField(
            Opcodes.ACC_PRIVATE | Opcodes.ACC_STATIC | Opcodes.ACC_FINAL,
            "FUNCTION_NAME",
            "Ljava/lang/String;",
            null,
            null
        ).visitEnd();

        cw.visitField(
            Opcodes.ACC_PRIVATE | Opcodes.ACC_STATIC | Opcodes.ACC_FINAL,
            "operations",
            LIST_DESC,
            null,
            null
        ).visitEnd();

        cw.visitField(
            Opcodes.ACC_PRIVATE | Opcodes.ACC_STATIC | Opcodes.ACC_FINAL,
            "METADATA",
            METADATA_DESC,
            null,
            null
        ).visitEnd();

        // Generate static initializer
        generateStaticInit(cw, internalName, metadata, mlirSource, functionName);

        // Generate constructor
        generateConstructor(cw, internalName);

        // Generate abstract method implementations
        generateOperationsMethod(cw, internalName);
        generateInputIndicesMethod(cw, analysis.inputIndices);
        generateOutputIndicesMethod(cw, analysis.outputIndices);
        generateTensorCountMethod(cw, analysis.tensorCount);
        generateOperationInputIndicesMethod(cw, analysis.opInputs);
        generateOperationOutputIndicesMethod(cw, analysis.opOutputs);
        generateMetadataMethod(cw, internalName);

        cw.visitEnd();

        return cw.toByteArray();
    }

    private static void generateStaticInit(
            ClassWriter cw,
            String internalName,
            ModelMetadata metadata,
            String mlirSource,
            String functionName) {

        MethodVisitor mv = cw.visitMethod(
            Opcodes.ACC_STATIC,
            "<clinit>",
            "()V",
            null,
            null
        );
        mv.visitCode();

        // MLIR_SOURCE = "<mlir source>"
        mv.visitLdcInsn(mlirSource);
        mv.visitFieldInsn(Opcodes.PUTSTATIC, internalName, "MLIR_SOURCE", "Ljava/lang/String;");

        // FUNCTION_NAME = "<function name>"
        mv.visitLdcInsn(functionName);
        mv.visitFieldInsn(Opcodes.PUTSTATIC, internalName, "FUNCTION_NAME", "Ljava/lang/String;");

        // Parse MLIR and extract operations:
        // StableHloAst.Module module = StableHloParser.parse(MLIR_SOURCE);
        mv.visitFieldInsn(Opcodes.GETSTATIC, internalName, "MLIR_SOURCE", "Ljava/lang/String;");
        mv.visitMethodInsn(
            Opcodes.INVOKESTATIC,
            PARSER_INTERNAL,
            "parse",
            "(Ljava/lang/String;)L" + MODULE_INTERNAL + ";",
            false
        );

        // Store module in local var 0
        mv.visitVarInsn(Opcodes.ASTORE, 0);

        // StableHloAst.Function function = module.getFunction(FUNCTION_NAME).get();
        mv.visitVarInsn(Opcodes.ALOAD, 0);
        mv.visitFieldInsn(Opcodes.GETSTATIC, internalName, "FUNCTION_NAME", "Ljava/lang/String;");
        mv.visitMethodInsn(
            Opcodes.INVOKEVIRTUAL,
            MODULE_INTERNAL,
            "getFunction",
            "(Ljava/lang/String;)Ljava/util/Optional;",
            false
        );
        mv.visitMethodInsn(
            Opcodes.INVOKEVIRTUAL,
            "java/util/Optional",
            "get",
            "()Ljava/lang/Object;",
            false
        );
        mv.visitTypeInsn(Opcodes.CHECKCAST, FUNCTION_INTERNAL);

        // Store function in local var 1
        mv.visitVarInsn(Opcodes.ASTORE, 1);

        // operations = function.body();
        mv.visitVarInsn(Opcodes.ALOAD, 1);
        mv.visitMethodInsn(
            Opcodes.INVOKEVIRTUAL,
            FUNCTION_INTERNAL,
            "body",
            "()" + LIST_DESC,
            false
        );
        mv.visitFieldInsn(Opcodes.PUTSTATIC, internalName, "operations", LIST_DESC);

        // METADATA = new ModelMetadata(name, hash, timestamp, version)
        mv.visitTypeInsn(Opcodes.NEW, METADATA_INTERNAL);
        mv.visitInsn(Opcodes.DUP);
        mv.visitLdcInsn(metadata.name());
        mv.visitLdcInsn(metadata.sourceHash());
        mv.visitLdcInsn(metadata.generatedAt());
        mv.visitLdcInsn(metadata.generatorVersion());
        mv.visitMethodInsn(
            Opcodes.INVOKESPECIAL,
            METADATA_INTERNAL,
            "<init>",
            "(Ljava/lang/String;Ljava/lang/String;JLjava/lang/String;)V",
            false
        );
        mv.visitFieldInsn(Opcodes.PUTSTATIC, internalName, "METADATA", METADATA_DESC);

        mv.visitInsn(Opcodes.RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static void generateConstructor(ClassWriter cw, String internalName) {
        MethodVisitor mv = cw.visitMethod(
            Opcodes.ACC_PUBLIC,
            "<init>",
            "()V",
            null,
            null
        );
        mv.visitCode();
        mv.visitVarInsn(Opcodes.ALOAD, 0);
        mv.visitMethodInsn(
            Opcodes.INVOKESPECIAL,
            ABSTRACT_MODEL_INTERNAL,
            "<init>",
            "()V",
            false
        );
        mv.visitInsn(Opcodes.RETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static void generateOperationsMethod(ClassWriter cw, String internalName) {
        MethodVisitor mv = cw.visitMethod(
            Opcodes.ACC_PROTECTED,
            "operations",
            "()" + LIST_DESC,
            null,
            null
        );
        mv.visitCode();
        mv.visitFieldInsn(Opcodes.GETSTATIC, internalName, "operations", LIST_DESC);
        mv.visitInsn(Opcodes.ARETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static void generateInputIndicesMethod(ClassWriter cw, int[] indices) {
        generateIntArrayMethod(cw, "inputIndices", indices);
    }

    private static void generateOutputIndicesMethod(ClassWriter cw, int[] indices) {
        generateIntArrayMethod(cw, "outputIndices", indices);
    }

    private static void generateTensorCountMethod(ClassWriter cw, int count) {
        MethodVisitor mv = cw.visitMethod(
            Opcodes.ACC_PROTECTED,
            "tensorCount",
            "()I",
            null,
            null
        );
        mv.visitCode();
        pushInt(mv, count);
        mv.visitInsn(Opcodes.IRETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static void generateOperationInputIndicesMethod(ClassWriter cw, int[][] indices) {
        generateInt2DArrayMethod(cw, "operationInputIndices", indices);
    }

    private static void generateOperationOutputIndicesMethod(ClassWriter cw, int[][] indices) {
        generateInt2DArrayMethod(cw, "operationOutputIndices", indices);
    }

    private static void generateMetadataMethod(ClassWriter cw, String internalName) {
        MethodVisitor mv = cw.visitMethod(
            Opcodes.ACC_PUBLIC,
            "metadata",
            "()" + METADATA_DESC,
            null,
            null
        );
        mv.visitCode();
        mv.visitFieldInsn(Opcodes.GETSTATIC, internalName, "METADATA", METADATA_DESC);
        mv.visitInsn(Opcodes.ARETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static void generateIntArrayMethod(ClassWriter cw, String methodName, int[] values) {
        MethodVisitor mv = cw.visitMethod(
            Opcodes.ACC_PROTECTED,
            methodName,
            "()[I",
            null,
            null
        );
        mv.visitCode();

        pushInt(mv, values.length);
        mv.visitIntInsn(Opcodes.NEWARRAY, Opcodes.T_INT);

        for (int i = 0; i < values.length; i++) {
            mv.visitInsn(Opcodes.DUP);
            pushInt(mv, i);
            pushInt(mv, values[i]);
            mv.visitInsn(Opcodes.IASTORE);
        }

        mv.visitInsn(Opcodes.ARETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static void generateInt2DArrayMethod(ClassWriter cw, String methodName, int[][] values) {
        MethodVisitor mv = cw.visitMethod(
            Opcodes.ACC_PROTECTED,
            methodName,
            "()[[I",
            null,
            null
        );
        mv.visitCode();

        // Create outer array
        pushInt(mv, values.length);
        mv.visitTypeInsn(Opcodes.ANEWARRAY, "[I");

        for (int i = 0; i < values.length; i++) {
            mv.visitInsn(Opcodes.DUP);
            pushInt(mv, i);

            // Create inner array
            int[] inner = values[i];
            pushInt(mv, inner.length);
            mv.visitIntInsn(Opcodes.NEWARRAY, Opcodes.T_INT);

            for (int j = 0; j < inner.length; j++) {
                mv.visitInsn(Opcodes.DUP);
                pushInt(mv, j);
                pushInt(mv, inner[j]);
                mv.visitInsn(Opcodes.IASTORE);
            }

            mv.visitInsn(Opcodes.AASTORE);
        }

        mv.visitInsn(Opcodes.ARETURN);
        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private static void pushInt(MethodVisitor mv, int value) {
        if (value >= -1 && value <= 5) {
            mv.visitInsn(Opcodes.ICONST_0 + value);
        } else if (value >= Byte.MIN_VALUE && value <= Byte.MAX_VALUE) {
            mv.visitIntInsn(Opcodes.BIPUSH, value);
        } else if (value >= Short.MIN_VALUE && value <= Short.MAX_VALUE) {
            mv.visitIntInsn(Opcodes.SIPUSH, value);
        } else {
            mv.visitLdcInsn(value);
        }
    }

    private static String computeHash(String mlirSource) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(mlirSource.getBytes());
            StringBuilder hex = new StringBuilder();
            for (byte b : hash) {
                hex.append(String.format("%02x", b));
            }
            return hex.toString();
        } catch (NoSuchAlgorithmException e) {
            return "unknown";
        }
    }

    /**
     * Result of analyzing a function.
     */
    private record FunctionAnalysis(
        int[] inputIndices,
        int[] outputIndices,
        int tensorCount,
        int[][] opInputs,
        int[][] opOutputs
    ) {}
}
