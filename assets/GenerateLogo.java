///usr/bin/env java --source 21 "$0" "$@"; exit $?
/*
 * Surfworks Logo Generator
 *
 * Generates SVG and PNG logo assets in light and dark modes.
 * This is the Java version - a matching Python version exists for polyglot verification.
 *
 * Usage:
 *     java GenerateLogo.java --all --output ./output
 *     java GenerateLogo.java --icon --dark --size 512
 *     java GenerateLogo.java --full --light --output ./brand
 *
 * Or make executable:
 *     chmod +x GenerateLogo.java
 *     ./GenerateLogo.java --all
 */

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

public class GenerateLogo {

    // Brand colors
    static final Map<String, Map<String, String>> COLORS = Map.of(
        "light", Map.of(
            "stroke", "#2B7A9B",
            "fill", "#2B7A9B",
            "pupil", "#1B2838",
            "eye_white", "#FFFFFF",
            "highlight", "white"
        ),
        "dark", Map.of(
            "stroke", "#E0E0E0",
            "fill", "#E0E0E0",
            "pupil", "#1B2838",
            "eye_white", "#5BA4C9",
            "highlight", "white"
        )
    );

    /**
     * Generate SVG path for a wavy circle (saw blade effect).
     */
    static String generateWavyCircle(double cx, double cy, double r, int waves, double amplitude) {
        StringBuilder path = new StringBuilder();
        int steps = waves * 8;

        for (int i = 0; i < steps; i++) {
            double angle = 2 * Math.PI * i / steps;
            double wave = amplitude * Math.sin(waves * angle);
            double currR = r + wave;
            double x = cx + currR * Math.cos(angle);
            double y = cy + currR * Math.sin(angle);

            if (i == 0) {
                path.append(String.format("M%.1f,%.1f", x, y));
            } else {
                path.append(String.format(" L%.1f,%.1f", x, y));
            }
        }
        path.append(" Z");
        return path.toString();
    }

    /**
     * Generate the icon-only SVG (eye within saw blade).
     */
    static String generateIconSvg(String mode) {
        Map<String, String> colors = COLORS.get(mode);
        String wavyPath = generateWavyCircle(100, 100, 78, 18, 10);
        String modeCapitalized = mode.substring(0, 1).toUpperCase() + mode.substring(1);

        return String.format("""
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <title>Surfworks Logo - %s Mode</title>

  <defs>
    <style>
      .blade { fill: none; stroke: %s; stroke-width: 3.5; stroke-linejoin: round; }
      .main { fill: %s; }
      .eye-white { fill: %s; }
      .pupil { fill: %s; }
    </style>
  </defs>

  <!-- Wavy circular border (saw blade) -->
  <path class="blade" d="%s"/>

  <!-- Blue crescent moon - wraps around left side of eye -->
  <path class="main" d="
    M 55,100
    C 55,60 78,35 110,35
    C 70,45 55,70 55,100
    C 55,130 70,155 110,165
    C 78,165 55,140 55,100
    Z"/>

  <!-- Arc on right side -->
  <path class="main" d="
    M 110,35
    C 150,35 165,60 165,100
    C 165,140 150,165 110,165
    C 140,155 152,130 152,100
    C 152,70 140,45 110,35
    Z"/>

  <!-- Eye white - almond shaped -->
  <path class="eye-white" d="
    M 70,100
    C 70,70 90,50 115,50
    C 140,50 155,70 155,100
    C 155,130 140,150 115,150
    C 90,150 70,130 70,100
    Z"/>

  <!-- Inner crescent (overlaps white to create moon effect) -->
  <path class="main" d="
    M 70,100
    C 70,68 88,48 115,48
    C 92,55 80,75 80,100
    C 80,125 92,145 115,152
    C 88,152 70,132 70,100
    Z"/>

  <!-- Pupil - dark ellipse -->
  <ellipse class="pupil" cx="125" cy="100" rx="18" ry="28"/>

  <!-- Small highlight on pupil -->
  <ellipse cx="132" cy="90" rx="5" ry="6" fill="%s" opacity="0.4"/>
</svg>""",
            modeCapitalized,
            colors.get("stroke"),
            colors.get("fill"),
            colors.get("eye_white"),
            colors.get("pupil"),
            wavyPath,
            colors.get("highlight")
        );
    }

    /**
     * Generate the full logo SVG (icon + 'surfworks' text).
     */
    static String generateFullSvg(String mode) {
        Map<String, String> colors = COLORS.get(mode);
        String wavyPath = generateWavyCircle(80, 80, 62, 18, 8);
        String modeCapitalized = mode.substring(0, 1).toUpperCase() + mode.substring(1);

        return String.format("""
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 160" width="380" height="160">
  <title>Surfworks Logo Full - %s Mode</title>

  <defs>
    <style>
      .blade { fill: none; stroke: %s; stroke-width: 2.8; stroke-linejoin: round; }
      .main { fill: %s; }
      .eye-white { fill: %s; }
      .pupil { fill: %s; }
      .text {
        font-family: 'VAG Rounded Std', 'Nunito', 'Arial Rounded MT Bold', sans-serif;
        font-weight: 400;
        fill: %s;
      }
    </style>
  </defs>

  <!-- Wavy circular border (saw blade) -->
  <path class="blade" d="%s"/>

  <!-- Crescent moon -->
  <path class="main" d="
    M 44,80
    C 44,52 62,32 88,32
    C 56,40 44,58 44,80
    C 44,102 56,120 88,128
    C 62,128 44,108 44,80
    Z"/>

  <!-- Arc on right side -->
  <path class="main" d="
    M 88,32
    C 120,32 132,52 132,80
    C 132,108 120,128 88,128
    C 112,120 122,102 122,80
    C 122,58 112,40 88,32
    Z"/>

  <!-- Eye white - almond shaped -->
  <path class="eye-white" d="
    M 56,80
    C 56,58 72,44 92,44
    C 112,44 124,58 124,80
    C 124,102 112,116 92,116
    C 72,116 56,102 56,80
    Z"/>

  <!-- Inner crescent -->
  <path class="main" d="
    M 56,80
    C 56,56 71,42 92,42
    C 74,48 64,62 64,80
    C 64,98 74,112 92,118
    C 71,118 56,104 56,80
    Z"/>

  <!-- Pupil -->
  <ellipse class="pupil" cx="100" cy="80" rx="14" ry="22"/>

  <!-- Highlight -->
  <ellipse cx="106" cy="72" rx="4" ry="5" fill="%s" opacity="0.4"/>

  <!-- Text: surfworks -->
  <text x="160" y="95" class="text" font-size="42">surfworks</text>
</svg>""",
            modeCapitalized,
            colors.get("stroke"),
            colors.get("fill"),
            colors.get("eye_white"),
            colors.get("pupil"),
            colors.get("fill"),
            wavyPath,
            colors.get("highlight")
        );
    }

    /**
     * Convert SVG to PNG using macOS qlmanage.
     */
    static boolean svgToPng(Path svgPath, Path pngPath, int size) {
        try {
            // Try qlmanage (macOS)
            ProcessBuilder pb = new ProcessBuilder(
                "/usr/bin/qlmanage", "-t", "-s", String.valueOf(size),
                "-o", pngPath.getParent().toString(), svgPath.toString()
            );
            pb.redirectErrorStream(true);
            Process p = pb.start();
            p.waitFor();

            if (p.exitValue() == 0) {
                // qlmanage adds .png to the filename
                Path generated = pngPath.getParent().resolve(svgPath.getFileName() + ".png");
                if (Files.exists(generated)) {
                    Files.move(generated, pngPath, StandardCopyOption.REPLACE_EXISTING);
                    return true;
                }
            }
        } catch (Exception e) {
            // Fall through to try alternative
        }

        // Try rsvg-convert (Linux)
        try {
            ProcessBuilder pb = new ProcessBuilder(
                "rsvg-convert", "-w", String.valueOf(size), "-h", String.valueOf(size),
                "-o", pngPath.toString(), svgPath.toString()
            );
            Process p = pb.start();
            p.waitFor();
            return p.exitValue() == 0;
        } catch (Exception e) {
            System.err.println("Warning: Could not convert " + svgPath + " to PNG");
            return false;
        }
    }

    /**
     * Create macOS .icns file from SVG.
     */
    static boolean createIcns(Path outputDir, Path iconSvg) {
        if (iconSvg == null || !Files.exists(iconSvg)) {
            System.err.println("Warning: Icon SVG not found for .icns generation");
            return false;
        }

        Path iconsetDir = outputDir.resolve("surfworks.iconset");
        Path icnsPath = outputDir.resolve("surfworks.icns");

        try {
            Files.createDirectories(iconsetDir);

            // Required sizes for iconset
            int[][] sizes = {
                {16, 16}, {32, 16}, {32, 32}, {64, 32},
                {128, 128}, {256, 128}, {256, 256}, {512, 256},
                {512, 512}, {1024, 512}
            };
            String[] names = {
                "icon_16x16.png", "icon_16x16@2x.png",
                "icon_32x32.png", "icon_32x32@2x.png",
                "icon_128x128.png", "icon_128x128@2x.png",
                "icon_256x256.png", "icon_256x256@2x.png",
                "icon_512x512.png", "icon_512x512@2x.png"
            };

            for (int i = 0; i < sizes.length; i++) {
                int size = sizes[i][0];
                Path pngPath = iconsetDir.resolve(names[i]);
                svgToPng(iconSvg, pngPath, size);
            }

            // Convert iconset to icns
            ProcessBuilder pb = new ProcessBuilder(
                "iconutil", "-c", "icns", iconsetDir.toString(), "-o", icnsPath.toString()
            );
            Process p = pb.start();
            p.waitFor();

            if (p.exitValue() == 0) {
                System.out.println("  Created: " + icnsPath);
                // Clean up iconset
                try (Stream<Path> walk = Files.walk(iconsetDir)) {
                    walk.sorted(Comparator.reverseOrder())
                        .map(Path::toFile)
                        .forEach(File::delete);
                }
                return true;
            }
        } catch (Exception e) {
            System.err.println("Warning: Could not create .icns file: " + e.getMessage());
        }
        return false;
    }

    /**
     * Create favicon.ico from PNG files.
     */
    static boolean createFavicon(Path outputDir, Path icon16, Path icon32) {
        Path faviconPath = outputDir.resolve("favicon.ico");

        // Try using ImageMagick convert
        try {
            ProcessBuilder pb = new ProcessBuilder(
                "convert", icon16.toString(), icon32.toString(), faviconPath.toString()
            );
            Process p = pb.start();
            p.waitFor();
            if (p.exitValue() == 0) {
                System.out.println("  Created: " + faviconPath);
                return true;
            }
        } catch (Exception e) {
            // Try fallback
        }

        // Fallback: copy 32px as basic favicon
        try {
            Files.copy(icon32, faviconPath, StandardCopyOption.REPLACE_EXISTING);
            System.out.println("  Created: " + faviconPath + " (from 32px PNG)");
            return true;
        } catch (Exception e) {
            System.err.println("Warning: Could not create favicon.ico: " + e.getMessage());
            return false;
        }
    }

    static void printUsage() {
        System.out.println("""
Surfworks Logo Generator (Java)

Usage:
    java GenerateLogo.java [options]

Options:
    --all           Generate all assets (default if no options specified)
    --icon          Generate icon only (eye/saw blade)
    --full          Generate full logo (icon + text)
    --favicon       Generate favicon.ico
    --icns          Generate macOS .icns file

    --light         Light mode (dark logo for light backgrounds)
    --dark          Dark mode (light logo for dark backgrounds)
                    (default: both modes if neither specified)

    --size SIZE     PNG size in pixels (default: generate standard sizes)
    --output DIR    Output directory (default: current directory)
    -o DIR          Same as --output
    --svg-only      Only generate SVG files, skip PNG conversion

    --help, -h      Show this help message

Examples:
    java GenerateLogo.java --all --output ./assets
    java GenerateLogo.java --icon --dark --size 512
    java GenerateLogo.java --full --light --output ./brand
    java GenerateLogo.java --favicon --icns --output ./build
""");
    }

    public static void main(String[] args) throws Exception {
        // Parse arguments
        boolean generateAll = false;
        boolean generateIcon = false;
        boolean generateFull = false;
        boolean generateFavicon = false;
        boolean generateIcns = false;
        boolean lightMode = false;
        boolean darkMode = false;
        boolean svgOnly = false;
        int size = 0;
        String outputDir = ".";

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--all" -> generateAll = true;
                case "--icon" -> generateIcon = true;
                case "--full" -> generateFull = true;
                case "--favicon" -> generateFavicon = true;
                case "--icns" -> generateIcns = true;
                case "--light" -> lightMode = true;
                case "--dark" -> darkMode = true;
                case "--svg-only" -> svgOnly = true;
                case "--size" -> size = Integer.parseInt(args[++i]);
                case "--output", "-o" -> outputDir = args[++i];
                case "--help", "-h" -> { printUsage(); return; }
                default -> {
                    if (args[i].startsWith("-")) {
                        System.err.println("Unknown option: " + args[i]);
                        printUsage();
                        System.exit(1);
                    }
                }
            }
        }

        // Defaults
        if (!generateAll && !generateIcon && !generateFull && !generateFavicon && !generateIcns) {
            generateAll = true;
        }
        if (!lightMode && !darkMode) {
            lightMode = true;
            darkMode = true;
        }

        // Setup output directory
        Path output = Path.of(outputDir);
        Files.createDirectories(output);

        List<String> modes = new ArrayList<>();
        if (lightMode) modes.add("light");
        if (darkMode) modes.add("dark");

        // Standard sizes
        int[] iconSizes = size == 0 ? new int[]{1024, 512, 256, 128, 64, 32, 16} : new int[]{size};
        int[] fullSizes = size == 0 ? new int[]{800, 400, 200} : new int[]{size};

        Map<String, Path> iconSvgs = new HashMap<>();

        System.out.println("Generating Surfworks logo assets in " + output + "/");
        System.out.println();

        for (String mode : modes) {
            String modeSuffix = mode.equals("light") ? "" : "-dark";

            // Generate icon
            if (generateAll || generateIcon) {
                String svgContent = generateIconSvg(mode);
                Path svgPath = output.resolve("surfworks-logo" + modeSuffix + ".svg");
                Files.writeString(svgPath, svgContent);
                System.out.println("  Created: " + svgPath);
                iconSvgs.put(mode, svgPath);

                if (!svgOnly) {
                    for (int s : iconSizes) {
                        Path pngPath = output.resolve("surfworks-logo" + modeSuffix + "-" + s + ".png");
                        if (svgToPng(svgPath, pngPath, s)) {
                            System.out.println("  Created: " + pngPath);
                        }
                    }
                }
            }

            // Generate full logo
            if (generateAll || generateFull) {
                String svgContent = generateFullSvg(mode);
                Path svgPath = output.resolve("surfworks-logo-full" + modeSuffix + ".svg");
                Files.writeString(svgPath, svgContent);
                System.out.println("  Created: " + svgPath);

                if (!svgOnly) {
                    for (int s : fullSizes) {
                        Path pngPath = output.resolve("surfworks-logo-full" + modeSuffix + "-" + s + ".png");
                        if (svgToPng(svgPath, pngPath, s)) {
                            System.out.println("  Created: " + pngPath);
                        }
                    }
                }
            }
        }

        // Generate favicon
        if (generateAll || generateFavicon) {
            Path icon16 = output.resolve("surfworks-logo-16.png");
            Path icon32 = output.resolve("surfworks-logo-32.png");
            if (Files.exists(icon16) && Files.exists(icon32)) {
                createFavicon(output, icon16, icon32);
            }
        }

        // Generate .icns
        if (generateAll || generateIcns) {
            Path lightSvg = iconSvgs.get("light");
            if (lightSvg != null) {
                createIcns(output, lightSvg);
            }
        }

        System.out.println();
        System.out.println("Done!");
    }
}
