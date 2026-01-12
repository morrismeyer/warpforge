#!/usr/bin/env python3
"""
Surfworks Logo Generator

Generates SVG and PNG logo assets in light and dark modes.
This is the Python version - a matching Java version exists for polyglot verification.

Usage:
    python generate-logo.py --all --output ./output
    python generate-logo.py --icon --dark --size 512
    python generate-logo.py --full --light --output ./brand
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path


# Brand colors
COLORS = {
    'light': {
        'stroke': '#2B7A9B',
        'fill': '#2B7A9B',
        'pupil': '#1B2838',
        'eye_white': '#FFFFFF',
        'highlight': 'white',
    },
    'dark': {
        'stroke': '#E0E0E0',
        'fill': '#E0E0E0',
        'pupil': '#1B2838',
        'eye_white': '#5BA4C9',
        'highlight': 'white',
    }
}


def generate_wavy_circle(cx: float, cy: float, r: float, waves: int, amplitude: float) -> str:
    """Generate SVG path for a wavy circle (saw blade effect)."""
    points = []
    steps = waves * 8
    for i in range(steps):
        angle = 2 * math.pi * i / steps
        wave = amplitude * math.sin(waves * angle)
        curr_r = r + wave
        x = cx + curr_r * math.cos(angle)
        y = cy + curr_r * math.sin(angle)
        points.append((x, y))

    path = f"M{points[0][0]:.1f},{points[0][1]:.1f}"
    for i in range(1, len(points)):
        path += f" L{points[i][0]:.1f},{points[i][1]:.1f}"
    path += " Z"
    return path


def generate_icon_svg(mode: str = 'light') -> str:
    """Generate the icon-only SVG (eye within saw blade)."""
    colors = COLORS[mode]
    wavy_path = generate_wavy_circle(100, 100, 78, 18, 10)

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <title>Surfworks Logo - {mode.capitalize()} Mode</title>

  <defs>
    <style>
      .blade {{ fill: none; stroke: {colors['stroke']}; stroke-width: 3.5; stroke-linejoin: round; }}
      .main {{ fill: {colors['fill']}; }}
      .eye-white {{ fill: {colors['eye_white']}; }}
      .pupil {{ fill: {colors['pupil']}; }}
    </style>
  </defs>

  <!-- Wavy circular border (saw blade) -->
  <path class="blade" d="{wavy_path}"/>

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
  <ellipse cx="132" cy="90" rx="5" ry="6" fill="{colors['highlight']}" opacity="0.4"/>
</svg>'''


def generate_full_svg(mode: str = 'light') -> str:
    """Generate the full logo SVG (icon + 'surfworks' text)."""
    colors = COLORS[mode]
    wavy_path = generate_wavy_circle(80, 80, 62, 18, 8)

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 160" width="380" height="160">
  <title>Surfworks Logo Full - {mode.capitalize()} Mode</title>

  <defs>
    <style>
      .blade {{ fill: none; stroke: {colors['stroke']}; stroke-width: 2.8; stroke-linejoin: round; }}
      .main {{ fill: {colors['fill']}; }}
      .eye-white {{ fill: {colors['eye_white']}; }}
      .pupil {{ fill: {colors['pupil']}; }}
      .text {{
        font-family: 'VAG Rounded Std', 'Nunito', 'Arial Rounded MT Bold', sans-serif;
        font-weight: 400;
        fill: {colors['fill']};
      }}
    </style>
  </defs>

  <!-- Wavy circular border (saw blade) -->
  <path class="blade" d="{wavy_path}"/>

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
  <ellipse cx="106" cy="72" rx="4" ry="5" fill="{colors['highlight']}" opacity="0.4"/>

  <!-- Text: surfworks -->
  <text x="160" y="95" class="text" font-size="42">surfworks</text>
</svg>'''


def svg_to_png(svg_path: Path, png_path: Path, size: int) -> bool:
    """Convert SVG to PNG using macOS qlmanage or sips."""
    try:
        # Try qlmanage first (macOS)
        result = subprocess.run(
            ['/usr/bin/qlmanage', '-t', '-s', str(size), '-o', str(png_path.parent), str(svg_path)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # qlmanage adds .png to the filename
            generated = png_path.parent / f"{svg_path.name}.png"
            if generated.exists():
                generated.rename(png_path)
                return True
    except FileNotFoundError:
        pass

    # Fallback: try rsvg-convert if available
    try:
        result = subprocess.run(
            ['rsvg-convert', '-w', str(size), '-h', str(size), '-o', str(png_path), str(svg_path)],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        pass

    print(f"Warning: Could not convert {svg_path} to PNG. Install rsvg-convert for Linux support.")
    return False


def create_favicon(output_dir: Path, icon_16: Path, icon_32: Path) -> bool:
    """Create favicon.ico from PNG files."""
    favicon_path = output_dir / 'favicon.ico'

    # Try using ImageMagick convert
    try:
        result = subprocess.run(
            ['convert', str(icon_16), str(icon_32), str(favicon_path)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  Created: {favicon_path}")
            return True
    except FileNotFoundError:
        pass

    # Try using iconutil on macOS (need to create iconset first)
    # For now, just copy the 32px as a basic favicon
    try:
        import shutil
        shutil.copy(icon_32, favicon_path)
        print(f"  Created: {favicon_path} (from 32px PNG)")
        return True
    except Exception as e:
        print(f"Warning: Could not create favicon.ico: {e}")
        return False


def create_icns(output_dir: Path, icon_svgs: dict) -> bool:
    """Create macOS .icns file from SVG."""
    iconset_dir = output_dir / 'surfworks.iconset'
    icns_path = output_dir / 'surfworks.icns'

    # Create iconset directory
    iconset_dir.mkdir(exist_ok=True)

    # Required sizes for iconset
    sizes = [
        (16, 'icon_16x16.png'),
        (32, 'icon_16x16@2x.png'),
        (32, 'icon_32x32.png'),
        (64, 'icon_32x32@2x.png'),
        (128, 'icon_128x128.png'),
        (256, 'icon_128x128@2x.png'),
        (256, 'icon_256x256.png'),
        (512, 'icon_256x256@2x.png'),
        (512, 'icon_512x512.png'),
        (1024, 'icon_512x512@2x.png'),
    ]

    svg_path = icon_svgs.get('light')
    if not svg_path or not svg_path.exists():
        print("Warning: Light mode icon SVG not found for .icns generation")
        return False

    for size, filename in sizes:
        png_path = iconset_dir / filename
        # Generate PNG at this size
        try:
            result = subprocess.run(
                ['/usr/bin/qlmanage', '-t', '-s', str(size), '-o', str(iconset_dir), str(svg_path)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                generated = iconset_dir / f"{svg_path.name}.png"
                if generated.exists():
                    generated.rename(png_path)
        except Exception:
            pass

    # Convert iconset to icns
    try:
        result = subprocess.run(
            ['iconutil', '-c', 'icns', str(iconset_dir), '-o', str(icns_path)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  Created: {icns_path}")
            # Clean up iconset
            import shutil
            shutil.rmtree(iconset_dir)
            return True
    except FileNotFoundError:
        print("Warning: iconutil not found. Cannot create .icns file.")

    return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate Surfworks logo assets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --all --output ./assets
  %(prog)s --icon --dark --size 512
  %(prog)s --full --light --size 400
  %(prog)s --favicon --icns --output ./build
        '''
    )

    # What to generate
    parser.add_argument('--all', action='store_true', help='Generate all assets')
    parser.add_argument('--icon', action='store_true', help='Generate icon only (eye/saw blade)')
    parser.add_argument('--full', action='store_true', help='Generate full logo (icon + text)')
    parser.add_argument('--favicon', action='store_true', help='Generate favicon.ico')
    parser.add_argument('--icns', action='store_true', help='Generate macOS .icns file')

    # Mode
    parser.add_argument('--light', action='store_true', help='Light mode (dark logo for light backgrounds)')
    parser.add_argument('--dark', action='store_true', help='Dark mode (light logo for dark backgrounds)')

    # Options
    parser.add_argument('--size', type=int, default=0, help='PNG size in pixels (default: generate standard sizes)')
    parser.add_argument('--output', '-o', type=str, default='.', help='Output directory')
    parser.add_argument('--svg-only', action='store_true', help='Only generate SVG files, skip PNG conversion')

    args = parser.parse_args()

    # Default: generate all if nothing specified
    if not any([args.all, args.icon, args.full, args.favicon, args.icns]):
        args.all = True

    # Default: both modes if neither specified
    if not args.light and not args.dark:
        args.light = True
        args.dark = True

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = []
    if args.light:
        modes.append('light')
    if args.dark:
        modes.append('dark')

    # Standard sizes for PNG generation
    icon_sizes = [1024, 512, 256, 128, 64, 32, 16] if args.size == 0 else [args.size]
    full_sizes = [800, 400, 200] if args.size == 0 else [args.size]

    icon_svgs = {}

    print(f"Generating Surfworks logo assets in {output_dir}/")
    print()

    for mode in modes:
        mode_suffix = '' if mode == 'light' else '-dark'

        # Generate icon
        if args.all or args.icon:
            svg_content = generate_icon_svg(mode)
            svg_path = output_dir / f'surfworks-logo{mode_suffix}.svg'
            svg_path.write_text(svg_content)
            print(f"  Created: {svg_path}")
            icon_svgs[mode] = svg_path

            if not args.svg_only:
                for size in icon_sizes:
                    png_path = output_dir / f'surfworks-logo{mode_suffix}-{size}.png'
                    if svg_to_png(svg_path, png_path, size):
                        print(f"  Created: {png_path}")

        # Generate full logo
        if args.all or args.full:
            svg_content = generate_full_svg(mode)
            svg_path = output_dir / f'surfworks-logo-full{mode_suffix}.svg'
            svg_path.write_text(svg_content)
            print(f"  Created: {svg_path}")

            if not args.svg_only:
                for size in full_sizes:
                    png_path = output_dir / f'surfworks-logo-full{mode_suffix}-{size}.png'
                    if svg_to_png(svg_path, png_path, size):
                        print(f"  Created: {png_path}")

    # Generate favicon
    if args.all or args.favicon:
        icon_16 = output_dir / 'surfworks-logo-16.png'
        icon_32 = output_dir / 'surfworks-logo-32.png'
        if icon_16.exists() and icon_32.exists():
            create_favicon(output_dir, icon_16, icon_32)

    # Generate .icns
    if args.all or args.icns:
        if 'light' in icon_svgs:
            create_icns(output_dir, icon_svgs)

    print()
    print("Done!")


if __name__ == '__main__':
    main()
