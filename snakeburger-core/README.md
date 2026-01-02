# snakeburger-core

Builds with Java toolchains set to languageVersion=26, expecting your local Babylon JDK to be
discoverable by Gradle.

babylon-runtime writes an env file that sets:
- BABYLON_JDK_HOME
- GRADLE_OPTS with -Dorg.gradle.java.installations.paths=... so Gradle can find the JDK 26 toolchain.
