package io.surfworks.snakegrinder.core;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Source;
import org.graalvm.polyglot.Value;

/**
 * SnakeGrinder core.
 *
 * This is intentionally tiny. It provides a single integration surface:
 * create a GraalVM Polyglot Context and evaluate a bundled Python resource.
 */
public final class SnakeGrinder {

  private SnakeGrinder() {
  }

  /**
   * Runs a no-dependency GraalPy self-test using the bundled Python bootstrap.
   *
   * @return A small JSON-like string describing the runtime.
   */
  public static String selfTest() {
    try (Context ctx = Context.newBuilder("python")
        // Keep this tight by default. Expand later as you need FFM, files, or environment.
        .allowAllAccess(false)
        .build()) {

      evalPythonResource(ctx, "/snakegrinder/bootstrap.py", "snakegrinder_bootstrap.py");

      // Execute the function defined in bootstrap.py
      Value result = ctx.eval("python", "self_test()");
      return result.toString();
    }
  }

  /**
   * Evaluates a Python source file stored on the Java classpath.
   */
  public static void evalPythonResource(Context ctx, String classpathResource, String virtualName) {
    String code = readClasspathUtf8(classpathResource);
    try {
      Source src = Source.newBuilder("python", code, virtualName).build();
      ctx.eval(src);
    } catch (IOException e) {
      throw new IllegalStateException("Failed building Source for: " + classpathResource, e);
    }
  }

  private static String readClasspathUtf8(String resourcePath) {
    InputStream in = SnakeGrinder.class.getResourceAsStream(resourcePath);
    if (in == null) {
      throw new IllegalArgumentException("Missing classpath resource: " + resourcePath);
    }
    try (in) {
      byte[] bytes = in.readAllBytes();
      return new String(bytes, StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new IllegalStateException("Failed reading resource: " + resourcePath, e);
    }
  }
}
