package io.surfworks.warpforge.core.io;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * NumPy .npy file I/O for reading and writing tensors.
 *
 * The .npy format stores a single NumPy array with:
 * - Magic number: \x93NUMPY
 * - Version: 1.0, 2.0, or 3.0
 * - Header: Python dict with dtype, shape, fortran_order
 * - Data: Raw binary array data
 */
public final class NpyIO {

    // Magic bytes: \x93NUMPY
    private static final byte[] MAGIC = {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'};

    private NpyIO() {} // Utility class

    // ==================== Reading ====================

    /**
     * Read a tensor from a .npy file.
     */
    public static Tensor read(Path path) throws IOException {
        try (InputStream is = Files.newInputStream(path);
             BufferedInputStream bis = new BufferedInputStream(is)) {
            return read(bis);
        }
    }

    /**
     * Read a tensor from an input stream.
     */
    public static Tensor read(InputStream in) throws IOException {
        DataInputStream dis = new DataInputStream(in);

        // Read and verify magic bytes
        byte[] magic = new byte[6];
        dis.readFully(magic);
        for (int i = 0; i < MAGIC.length; i++) {
            if (magic[i] != MAGIC[i]) {
                throw new IOException("Invalid NumPy magic number");
            }
        }

        // Read version
        int majorVersion = dis.readUnsignedByte();
        int minorVersion = dis.readUnsignedByte();

        // Read header length (little-endian)
        int headerLen;
        if (majorVersion == 1) {
            // Version 1.0: 2-byte header length
            int b0 = dis.readUnsignedByte();
            int b1 = dis.readUnsignedByte();
            headerLen = b0 | (b1 << 8);
        } else {
            // Version 2.0+: 4-byte header length
            int b0 = dis.readUnsignedByte();
            int b1 = dis.readUnsignedByte();
            int b2 = dis.readUnsignedByte();
            int b3 = dis.readUnsignedByte();
            headerLen = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }

        // Read header string
        byte[] headerBytes = new byte[headerLen];
        dis.readFully(headerBytes);
        String headerStr = new String(headerBytes, StandardCharsets.US_ASCII).trim();

        // Parse header
        NpyHeader header = NpyHeader.parse(majorVersion, minorVersion, headerStr);

        // Create tensor spec
        TensorSpec spec;
        if (header.fortranOrder()) {
            // Column-major layout
            long[] strides = TensorSpec.computeColumnMajorStrides(header.shape());
            spec = TensorSpec.withStrides(header.dtype(), header.shape(), strides);
        } else {
            // Row-major layout (default)
            spec = TensorSpec.of(header.dtype(), header.shape());
        }

        // Allocate tensor
        Arena arena = Arena.ofConfined();
        MemorySegment segment = arena.allocate(spec.byteSize());

        // Read data
        byte[] dataBytes = new byte[(int) spec.byteSize()];
        dis.readFully(dataBytes);

        // Copy to memory segment, handling byte order if needed
        if (header.needsByteSwap()) {
            swapAndCopy(dataBytes, segment, header.dtype(), header.byteOrder());
        } else {
            MemorySegment.copy(dataBytes, 0, segment, ValueLayout.JAVA_BYTE, 0, dataBytes.length);
        }

        return Tensor.fromMemorySegment(segment, spec, arena);
    }

    /**
     * Swap bytes and copy to memory segment.
     */
    private static void swapAndCopy(byte[] data, MemorySegment segment, ScalarType dtype, ByteOrder sourceOrder) {
        ByteBuffer buffer = ByteBuffer.wrap(data).order(sourceOrder);
        int elementSize = dtype.byteSize();
        int elementCount = data.length / elementSize;

        switch (dtype) {
            case F32 -> {
                for (int i = 0; i < elementCount; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, buffer.getFloat(i * 4));
                }
            }
            case F64 -> {
                for (int i = 0; i < elementCount; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, buffer.getDouble(i * 8));
                }
            }
            case I16 -> {
                for (int i = 0; i < elementCount; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_SHORT, i, buffer.getShort(i * 2));
                }
            }
            case I32 -> {
                for (int i = 0; i < elementCount; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_INT, i, buffer.getInt(i * 4));
                }
            }
            case I64 -> {
                for (int i = 0; i < elementCount; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_LONG, i, buffer.getLong(i * 8));
                }
            }
            default -> {
                // Single-byte types don't need swapping
                MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);
            }
        }
    }

    // ==================== Writing ====================

    /**
     * Write a tensor to a .npy file.
     */
    public static void write(Tensor tensor, Path path) throws IOException {
        try (OutputStream os = Files.newOutputStream(path);
             BufferedOutputStream bos = new BufferedOutputStream(os)) {
            write(tensor, bos);
        }
    }

    /**
     * Write a tensor to an output stream.
     */
    public static void write(Tensor tensor, OutputStream out) throws IOException {
        DataOutputStream dos = new DataOutputStream(out);

        // Write magic
        dos.write(MAGIC);

        // Create header
        NpyHeader header = new NpyHeader(
            1, 0, // Version 1.0
            tensor.dtype(),
            ByteOrder.LITTLE_ENDIAN,
            false, // Row-major
            tensor.shape()
        );
        String headerStr = header.toHeaderString();

        // Pad header to 64-byte alignment
        // Total: 6 (magic) + 2 (version) + 2 (header len) + headerStr.length + padding + \n
        int baseLen = 6 + 2 + 2 + headerStr.length() + 1; // +1 for newline
        int padding = (64 - (baseLen % 64)) % 64;
        int totalHeaderLen = headerStr.length() + padding + 1;

        // Write version
        dos.writeByte(1); // major
        dos.writeByte(0); // minor

        // Write header length (little-endian)
        dos.writeByte(totalHeaderLen & 0xFF);
        dos.writeByte((totalHeaderLen >> 8) & 0xFF);

        // Write header string
        dos.write(headerStr.getBytes(StandardCharsets.US_ASCII));

        // Write padding spaces
        for (int i = 0; i < padding; i++) {
            dos.writeByte(' ');
        }

        // Write newline
        dos.writeByte('\n');

        // Write data
        byte[] dataBytes = new byte[(int) tensor.spec().byteSize()];
        MemorySegment.copy(tensor.data(), ValueLayout.JAVA_BYTE, 0, dataBytes, 0, dataBytes.length);
        dos.write(dataBytes);
    }

    // ==================== Header Parsing ====================

    /**
     * Read only the header from a .npy file (useful for metadata inspection).
     */
    public static NpyHeader readHeader(Path path) throws IOException {
        try (InputStream is = Files.newInputStream(path);
             BufferedInputStream bis = new BufferedInputStream(is)) {
            return readHeader(bis);
        }
    }

    /**
     * Read only the header from an input stream.
     */
    public static NpyHeader readHeader(InputStream in) throws IOException {
        DataInputStream dis = new DataInputStream(in);

        // Read and verify magic bytes
        byte[] magic = new byte[6];
        dis.readFully(magic);
        for (int i = 0; i < MAGIC.length; i++) {
            if (magic[i] != MAGIC[i]) {
                throw new IOException("Invalid NumPy magic number");
            }
        }

        // Read version
        int majorVersion = dis.readUnsignedByte();
        int minorVersion = dis.readUnsignedByte();

        // Read header length
        int headerLen;
        if (majorVersion == 1) {
            int b0 = dis.readUnsignedByte();
            int b1 = dis.readUnsignedByte();
            headerLen = b0 | (b1 << 8);
        } else {
            int b0 = dis.readUnsignedByte();
            int b1 = dis.readUnsignedByte();
            int b2 = dis.readUnsignedByte();
            int b3 = dis.readUnsignedByte();
            headerLen = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }

        // Read header string
        byte[] headerBytes = new byte[headerLen];
        dis.readFully(headerBytes);
        String headerStr = new String(headerBytes, StandardCharsets.US_ASCII).trim();

        return NpyHeader.parse(majorVersion, minorVersion, headerStr);
    }
}
