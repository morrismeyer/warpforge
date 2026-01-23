package io.surfworks.warpforge.io.collective.impl;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Level;
import java.util.logging.Logger;

import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.io.collective.AllReduceOp;
import io.surfworks.warpforge.io.collective.CollectiveApi;
import io.surfworks.warpforge.io.collective.CollectiveConfig;
import io.surfworks.warpforge.io.collective.CollectiveException;
import io.surfworks.warpforge.io.collective.CollectiveApi.CollectiveStats;
import io.surfworks.warpforge.io.collective.impl.OperationArenaPool.PooledArena;
import io.surfworks.warpforge.io.ffi.ucc.Ucc;
import io.surfworks.warpforge.io.ffi.ucc.ucc_coll_args;
import io.surfworks.warpforge.io.ffi.ucc.ucc_coll_buffer_info;
import io.surfworks.warpforge.io.ffi.ucc.ucc_context_params;
import io.surfworks.warpforge.io.ffi.ucc.ucc_lib_params;
import io.surfworks.warpforge.io.ffi.ucc.ucc_oob_coll;
import io.surfworks.warpforge.io.ffi.ucc.ucc_team_params;

/**
 * UCC-backed implementation of CollectiveApi.
 *
 * <p>This implementation uses the UCC (Unified Collective Communications)
 * library via jextract-generated FFM bindings for high-performance collective
 * operations over RDMA.
 *
 * <h2>Requirements</h2>
 * <ul>
 *   <li>Linux operating system</li>
 *   <li>UCC libraries installed (libucc.so)</li>
 *   <li>UCX libraries installed (for transport)</li>
 *   <li>jextract-generated stubs in io.surfworks.warpforge.io.ffi.ucc</li>
 * </ul>
 *
 * <h2>Implementation Status</h2>
 * <p>This class requires jextract-generated FFM bindings to function.
 * Run {@code ./gradlew :openucx-runtime:generateJextractStubs} to generate them.
 *
 * <h2>Threading Model</h2>
 * <p>Operations run synchronously on the calling thread by default, which is
 * required for UCX thread affinity. Optionally, a dedicated progress thread
 * can be enabled for true async operations.
 *
 * <h2>Performance Optimizations</h2>
 * <ul>
 *   <li>Arena pooling: Reuses pre-allocated arenas to avoid per-op allocation overhead</li>
 *   <li>Adaptive polling: Reduces FFM call overhead for large message operations</li>
 *   <li>Progress thread: Optional dedicated thread for async completion (experimental)</li>
 * </ul>
 */
public class UccCollectiveImpl implements CollectiveApi {

    private static final Logger LOG = Logger.getLogger(UccCollectiveImpl.class.getName());

    /** System property to enable arena pooling (default: true) */
    private static final String PROP_USE_ARENA_POOL = "warpforge.ucc.arenaPool";

    /** System property to enable progress thread (default: false, experimental) */
    private static final String PROP_USE_PROGRESS_THREAD = "warpforge.ucc.progressThread";

    private final CollectiveConfig config;
    private final Arena arena;

    // UCC handles (populated when FFM stubs are available)
    private MemorySegment uccLib;
    private MemorySegment uccContext;
    private MemorySegment uccTeam;

    // OOB coordinator for team formation
    private OobCoordinator oobCoordinator;

    // Performance optimization: Arena pool for operation allocations
    private final OperationArenaPool arenaPool;
    private final boolean useArenaPool;

    // Performance optimization: Dedicated progress thread (experimental)
    private final UccProgressThread progressThread;
    private final boolean useProgressThread;

    private volatile boolean closed = false;
    private volatile boolean initialized = false;

    // Statistics
    private final AtomicLong allReduceCount = new AtomicLong();
    private final AtomicLong allGatherCount = new AtomicLong();
    private final AtomicLong broadcastCount = new AtomicLong();
    private final AtomicLong reduceScatterCount = new AtomicLong();
    private final AtomicLong barrierCount = new AtomicLong();
    private final AtomicLong totalBytes = new AtomicLong();
    private final AtomicLong totalOps = new AtomicLong();

    public UccCollectiveImpl(CollectiveConfig config) {
        this.config = config;
        this.arena = Arena.ofShared();

        // Check optimization settings
        this.useArenaPool = Boolean.parseBoolean(
            System.getProperty(PROP_USE_ARENA_POOL, "true"));
        this.useProgressThread = Boolean.parseBoolean(
            System.getProperty(PROP_USE_PROGRESS_THREAD, "false"));

        // Initialize arena pool if enabled
        if (useArenaPool) {
            this.arenaPool = new OperationArenaPool();
            LOG.fine("Arena pooling enabled");
        } else {
            this.arenaPool = null;
        }

        // Initialize UCC context (must be done before progress thread)
        initializeUcc();

        // Initialize progress thread if enabled (after UCC context is ready)
        if (useProgressThread && uccContext != null) {
            this.progressThread = new UccProgressThread(uccContext);
            this.progressThread.start();
            LOG.info("Progress thread enabled (experimental)");
        } else {
            this.progressThread = null;
        }
    }

    private void initializeUcc() {
        try {
            // Check if FFM stubs are available and native library is loadable
            // Note: Loading Ucc class triggers SymbolLookup.libraryLookup() which
            // will throw if libucc.so is not found or not loadable
            Class.forName("io.surfworks.warpforge.io.ffi.ucc.Ucc");
            initializeUccReal();
            initialized = true;
        } catch (ClassNotFoundException e) {
            throw new CollectiveException(
                "UCC FFM bindings not found. Run: ./gradlew :openucx-runtime:generateJextractStubs",
                CollectiveException.ErrorCode.NOT_SUPPORTED);
        } catch (UnsatisfiedLinkError e) {
            throw new CollectiveException(
                "UCC native library (libucc.so) not found or not loadable: " + e.getMessage(),
                CollectiveException.ErrorCode.NOT_SUPPORTED);
        } catch (ExceptionInInitializerError e) {
            Throwable cause = e.getCause();
            String msg = cause != null ? cause.getMessage() : e.getMessage();
            throw new CollectiveException(
                "Failed to initialize UCC FFM bindings: " + msg,
                CollectiveException.ErrorCode.NOT_SUPPORTED);
        }
    }

    private void initializeUccReal() {
        LOG.info("Initializing UCC for rank " + config.rank() + " of " + config.worldSize());

        // 1. Read default UCC library configuration
        // This is required before ucc_init_version - passing NULL config can crash
        MemorySegment libConfigPtr = arena.allocate(ValueLayout.ADDRESS);
        int status = Ucc.ucc_lib_config_read(MemorySegment.NULL, MemorySegment.NULL, libConfigPtr);
        UccHelper.checkStatus(status, "ucc_lib_config_read");
        MemorySegment libConfig = libConfigPtr.get(ValueLayout.ADDRESS, 0);
        LOG.fine("UCC library config read successfully");

        // 2. Initialize UCC library
        // CRITICAL: Must zero-fill the struct - FFM allocate() returns uninitialized memory
        MemorySegment libParams = ucc_lib_params.allocate(arena);
        libParams.fill((byte) 0);  // Zero all fields to prevent garbage values
        ucc_lib_params.mask(libParams, 0L);  // mask=0 means no optional fields are set

        MemorySegment libHandlePtr = arena.allocate(ValueLayout.ADDRESS);
        // Use correct API version from UCC headers
        status = Ucc.ucc_init_version(
            Ucc.UCC_API_MAJOR(),
            Ucc.UCC_API_MINOR(),
            libParams,
            libConfig,  // Pass the config we read, not NULL
            libHandlePtr
        );
        UccHelper.checkStatus(status, "ucc_init_version");
        this.uccLib = libHandlePtr.get(ValueLayout.ADDRESS, 0);

        // Release the config - it's no longer needed after init
        Ucc.ucc_lib_config_release(libConfig);
        LOG.fine("UCC library initialized (API version " + Ucc.UCC_API_MAJOR() + "." + Ucc.UCC_API_MINOR() + ")");

        // 2. Create OOB coordinator for team formation (only needed for multi-rank)
        MemorySegment oobStruct = null;
        if (config.worldSize() > 1) {
            this.oobCoordinator = new OobCoordinator(config, arena);
            oobStruct = oobCoordinator.getOobStruct();
            LOG.fine("OOB coordinator created for multi-rank setup");
        } else {
            LOG.fine("Single-rank mode - skipping OOB coordinator");
        }

        // 3. Read default UCC context configuration
        MemorySegment ctxConfigPtr = arena.allocate(ValueLayout.ADDRESS);
        status = Ucc.ucc_context_config_read(uccLib, MemorySegment.NULL, ctxConfigPtr);
        UccHelper.checkStatus(status, "ucc_context_config_read");
        MemorySegment ctxConfig = ctxConfigPtr.get(ValueLayout.ADDRESS, 0);
        LOG.fine("UCC context config read successfully");

        // 4. Create UCC context
        MemorySegment ctxParams = ucc_context_params.allocate(arena);
        ctxParams.fill((byte) 0);  // Zero-fill to prevent garbage values

        // Only set OOB field for multi-rank
        if (config.worldSize() > 1 && oobStruct != null) {
            ucc_context_params.mask(ctxParams, Ucc.UCC_CONTEXT_PARAM_FIELD_OOB());
            ucc_context_params.oob(ctxParams, oobStruct);
        } else {
            ucc_context_params.mask(ctxParams, 0L);  // No OOB for single-rank
        }

        MemorySegment ctxHandlePtr = arena.allocate(ValueLayout.ADDRESS);
        status = Ucc.ucc_context_create(uccLib, ctxParams, ctxConfig, ctxHandlePtr);
        UccHelper.checkStatus(status, "ucc_context_create");
        this.uccContext = ctxHandlePtr.get(ValueLayout.ADDRESS, 0);

        // Release the context config - no longer needed after context creation
        Ucc.ucc_context_config_release(ctxConfig);
        LOG.fine("UCC context created");

        // 4. Create UCC team
        MemorySegment teamParams = ucc_team_params.allocate(arena);
        teamParams.fill((byte) 0);  // Zero-fill to prevent garbage values

        long teamMask = Ucc.UCC_TEAM_PARAM_FIELD_EP()
                      | Ucc.UCC_TEAM_PARAM_FIELD_EP_RANGE()
                      | Ucc.UCC_TEAM_PARAM_FIELD_TEAM_SIZE();

        // Only include OOB for multi-rank
        if (config.worldSize() > 1 && oobStruct != null) {
            teamMask |= Ucc.UCC_TEAM_PARAM_FIELD_OOB();
        }

        ucc_team_params.mask(teamParams, teamMask);
        ucc_team_params.ep(teamParams, config.rank());
        ucc_team_params.ep_range(teamParams, UccConstants.EP_RANGE_CONTIG);
        ucc_team_params.team_size(teamParams, config.worldSize());

        // Copy OOB structure to team params only for multi-rank
        if (config.worldSize() > 1 && oobStruct != null) {
            ucc_team_params.oob(teamParams, oobStruct);
        }

        // Allocate array of context pointers (we have 1 context)
        MemorySegment ctxArray = arena.allocate(ValueLayout.ADDRESS);
        ctxArray.set(ValueLayout.ADDRESS, 0, uccContext);

        MemorySegment teamHandlePtr = arena.allocate(ValueLayout.ADDRESS);
        status = Ucc.ucc_team_create_post(ctxArray, 1, teamParams, teamHandlePtr);
        UccHelper.checkStatus(status, "ucc_team_create_post");
        this.uccTeam = teamHandlePtr.get(ValueLayout.ADDRESS, 0);

        // 5. Poll until team creation completes
        LOG.fine("Waiting for team creation to complete...");
        int pollCount = 0;
        while (true) {
            status = Ucc.ucc_team_create_test(uccTeam);
            if (status == UccConstants.OK) {
                break;
            }
            if (status != UccConstants.INPROGRESS) {
                throw new CollectiveException(
                    "Team creation failed: " + UccConstants.statusToString(status),
                    CollectiveException.ErrorCode.COMMUNICATION_ERROR,
                    status
                );
            }
            pollCount++;
            if (pollCount % 10000 == 0) {
                LOG.fine("Team creation still in progress, poll count: " + pollCount);
            }
            Thread.onSpinWait();
        }

        LOG.info("UCC team created successfully for rank " + config.rank());
    }

    @Override
    public String backendName() {
        return "ucc";
    }

    @Override
    public CollectiveConfig config() {
        return config;
    }

    @Override
    public int worldSize() {
        return config.worldSize();
    }

    @Override
    public int rank() {
        return config.rank();
    }

    @Override
    public CompletableFuture<Tensor> allReduce(Tensor input, AllReduceOp op) {
        checkInitialized();
        checkNotClosed();

        // Single-rank optimization: no communication needed, just copy
        if (config.worldSize() == 1) {
            Tensor result = Tensor.zeros(input.dtype(), input.shape());
            MemorySegment.copy(input.data(), 0, result.data(), 0, input.spec().byteSize());
            allReduceCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());
            return CompletableFuture.completedFuture(result);
        }

        // Multi-rank: use UCC with arena pooling optimization
        PooledArena pooledArena = useArenaPool ? arenaPool.acquire() : null;
        Arena opArena = pooledArena != null ? pooledArena.arena() : Arena.ofConfined();

        try {
            Tensor result = Tensor.zeros(input.dtype(), input.shape());

            // Set up collective args (use pooled allocator if available)
            MemorySegment args = pooledArena != null
                ? pooledArena.allocateCollArgs()
                : ucc_coll_args.allocate(opArena);
            UccHelper.setupCollectiveArgsWithOp(args, UccConstants.COLL_TYPE_ALLREDUCE, allReduceOpToUcc(op));

            // Configure source buffer
            MemorySegment srcInfo = UccHelper.getSrcBufferInfo(args);
            UccHelper.setupBufferInfo(srcInfo, input);

            // Configure destination buffer
            MemorySegment dstInfo = UccHelper.getDstBufferInfo(args);
            UccHelper.setupBufferInfo(dstInfo, result);

            // Initialize and post collective
            MemorySegment requestPtr = pooledArena != null
                ? pooledArena.allocatePointer()
                : opArena.allocate(ValueLayout.ADDRESS);
            UccHelper.initAndPostCollective(args, requestPtr, uccTeam, "allreduce");

            // Wait for completion
            MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
            UccHelper.waitForCompletionWithProgress(request, uccContext);

            // Update statistics
            allReduceCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());

            return CompletableFuture.completedFuture(result);
        } finally {
            // Release pooled arena or close temporary arena
            if (pooledArena != null) {
                arenaPool.release(pooledArena);
            } else {
                opArena.close();
            }
        }
    }

    @Override
    public CompletableFuture<Void> allReduceInPlace(Tensor tensor, AllReduceOp op) {
        checkInitialized();
        checkNotClosed();

        // Single-rank optimization: no communication needed, data already in place
        if (config.worldSize() == 1) {
            allReduceCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(tensor.spec().byteSize());
            return CompletableFuture.completedFuture(null);
        }

        // Multi-rank: use UCC (must run synchronously - UCX thread affinity)
        try (Arena opArena = Arena.ofConfined()) {
            // Set up collective args with in-place flag
            MemorySegment args = ucc_coll_args.allocate(opArena);
            UccHelper.setupCollectiveArgsWithOp(args, UccConstants.COLL_TYPE_ALLREDUCE, allReduceOpToUcc(op));
            ucc_coll_args.flags(args, UccConstants.COLL_ARGS_FLAG_IN_PLACE);

            // For in-place, both src and dst point to the same buffer
            MemorySegment srcInfo = UccHelper.getSrcBufferInfo(args);
            UccHelper.setupBufferInfo(srcInfo, tensor);

            MemorySegment dstInfo = UccHelper.getDstBufferInfo(args);
            UccHelper.setupBufferInfo(dstInfo, tensor);

            // Initialize and post collective
            MemorySegment requestPtr = opArena.allocate(ValueLayout.ADDRESS);
            UccHelper.initAndPostCollective(args, requestPtr, uccTeam, "allreduce_inplace");

            // Wait for completion
            MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
            UccHelper.waitForCompletionWithProgress(request, uccContext);

            // Update statistics
            allReduceCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(tensor.spec().byteSize());
        }
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Void> allReduceRaw(MemorySegment buffer, long count, int datatype, AllReduceOp op) {
        checkInitialized();
        checkNotClosed();

        // Single-rank optimization: no communication needed, data already in place
        if (config.worldSize() == 1) {
            allReduceCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(buffer.byteSize());
            return CompletableFuture.completedFuture(null);
        }

        // Multi-rank: use UCC (must run synchronously - UCX thread affinity)
        try (Arena opArena = Arena.ofConfined()) {
            // Set up collective args with in-place flag (modifies buffer in place)
            MemorySegment args = ucc_coll_args.allocate(opArena);
            UccHelper.setupCollectiveArgsWithOp(args, UccConstants.COLL_TYPE_ALLREDUCE, allReduceOpToUcc(op));
            ucc_coll_args.flags(args, UccConstants.COLL_ARGS_FLAG_IN_PLACE);

            // Configure buffers - use raw buffer info
            long uccDatatype = uccDatatypeFromInt(datatype);
            MemorySegment srcInfo = UccHelper.getSrcBufferInfo(args);
            UccHelper.setupBufferInfo(srcInfo, buffer, count, uccDatatype, UccConstants.MEMORY_TYPE_HOST);

            MemorySegment dstInfo = UccHelper.getDstBufferInfo(args);
            UccHelper.setupBufferInfo(dstInfo, buffer, count, uccDatatype, UccConstants.MEMORY_TYPE_HOST);

            // Initialize and post collective
            MemorySegment requestPtr = opArena.allocate(ValueLayout.ADDRESS);
            UccHelper.initAndPostCollective(args, requestPtr, uccTeam, "allreduce_raw");

            // Wait for completion
            MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
            UccHelper.waitForCompletionWithProgress(request, uccContext);

            // Update statistics
            allReduceCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(buffer.byteSize());
        }
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Tensor> allGather(Tensor input) {
        checkInitialized();
        checkNotClosed();

        // Single-rank optimization: no communication needed, just copy
        if (config.worldSize() == 1) {
            Tensor result = Tensor.zeros(input.dtype(), input.shape());
            MemorySegment.copy(input.data(), 0, result.data(), 0, input.spec().byteSize());
            allGatherCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());
            return CompletableFuture.completedFuture(result);
        }

        // Multi-rank: use UCC (must run synchronously - UCX thread affinity)
        try (Arena opArena = Arena.ofConfined()) {
            // Allocate output tensor with worldSize * input size
            int[] newShape = input.shape().clone();
            newShape[0] *= config.worldSize();
            Tensor result = Tensor.zeros(input.dtype(), newShape);

            // Set up collective args
            MemorySegment args = ucc_coll_args.allocate(opArena);
            UccHelper.setupCollectiveArgs(args, UccConstants.COLL_TYPE_ALLGATHER);

            // Configure source buffer
            MemorySegment srcInfo = UccHelper.getSrcBufferInfo(args);
            UccHelper.setupBufferInfo(srcInfo, input);

            // Configure destination buffer
            MemorySegment dstInfo = UccHelper.getDstBufferInfo(args);
            UccHelper.setupBufferInfo(dstInfo, result);

            // Initialize and post collective
            MemorySegment requestPtr = opArena.allocate(ValueLayout.ADDRESS);
            UccHelper.initAndPostCollective(args, requestPtr, uccTeam, "allgather");

            // Wait for completion
            MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
            UccHelper.waitForCompletionWithProgress(request, uccContext);

            // Update statistics
            allGatherCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize() * config.worldSize());

            return CompletableFuture.completedFuture(result);
        }
    }

    @Override
    public CompletableFuture<Void> allGather(Tensor input, Tensor output) {
        checkInitialized();
        checkNotClosed();

        // Single-rank optimization: no communication needed, just copy
        if (config.worldSize() == 1) {
            MemorySegment.copy(input.data(), 0, output.data(), 0, input.spec().byteSize());
            allGatherCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());
            return CompletableFuture.completedFuture(null);
        }

        // Multi-rank: use UCC (must run synchronously - UCX thread affinity)
        try (Arena opArena = Arena.ofConfined()) {
            // Set up collective args
            MemorySegment args = ucc_coll_args.allocate(opArena);
            UccHelper.setupCollectiveArgs(args, UccConstants.COLL_TYPE_ALLGATHER);

            // Configure source buffer
            MemorySegment srcInfo = UccHelper.getSrcBufferInfo(args);
            UccHelper.setupBufferInfo(srcInfo, input);

            // Configure destination buffer
            MemorySegment dstInfo = UccHelper.getDstBufferInfo(args);
            UccHelper.setupBufferInfo(dstInfo, output);

            // Initialize and post collective
            MemorySegment requestPtr = opArena.allocate(ValueLayout.ADDRESS);
            UccHelper.initAndPostCollective(args, requestPtr, uccTeam, "allgather_into");

            // Wait for completion
            MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
            UccHelper.waitForCompletionWithProgress(request, uccContext);

            allGatherCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize() * config.worldSize());
        }
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Tensor> broadcast(Tensor tensor, int root) {
        checkInitialized();
        checkNotClosed();
        validateRank(root);

        // Single-rank optimization: no communication needed, just copy
        if (config.worldSize() == 1) {
            Tensor result = Tensor.zeros(tensor.dtype(), tensor.shape());
            MemorySegment.copy(tensor.data(), 0, result.data(), 0, tensor.spec().byteSize());
            broadcastCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(tensor.spec().byteSize());
            return CompletableFuture.completedFuture(result);
        }

        // Multi-rank: use UCC (must run synchronously - UCX thread affinity)
        try (Arena opArena = Arena.ofConfined()) {
            // Allocate output tensor
            Tensor result = Tensor.zeros(tensor.dtype(), tensor.shape());

            // PERF: Only root copies input to result - non-root will receive data via UCC
            // Avoids unnecessary 1MB+ memcpy on non-root ranks
            if (config.rank() == root) {
                MemorySegment.copy(tensor.data(), 0, result.data(), 0, tensor.spec().byteSize());
            }

            // Set up collective args
            MemorySegment args = ucc_coll_args.allocate(opArena);
            UccHelper.setupCollectiveArgsWithRoot(args, UccConstants.COLL_TYPE_BCAST, root);

            // Configure source buffer (root's data)
            MemorySegment srcInfo = UccHelper.getSrcBufferInfo(args);
            UccHelper.setupBufferInfo(srcInfo, config.rank() == root ? tensor : result);

            // Configure destination buffer
            MemorySegment dstInfo = UccHelper.getDstBufferInfo(args);
            UccHelper.setupBufferInfo(dstInfo, result);

            // Initialize and post collective
            MemorySegment requestPtr = opArena.allocate(ValueLayout.ADDRESS);
            UccHelper.initAndPostCollective(args, requestPtr, uccTeam, "broadcast");

            // Wait for completion
            MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
            UccHelper.waitForCompletionWithProgress(request, uccContext);

            // Update statistics
            broadcastCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(tensor.spec().byteSize());

            return CompletableFuture.completedFuture(result);
        }
    }

    @Override
    public CompletableFuture<Void> broadcastInPlace(Tensor tensor, int root) {
        checkInitialized();
        checkNotClosed();
        validateRank(root);

        // Single-rank optimization: no communication needed, data already in place
        if (config.worldSize() == 1) {
            broadcastCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(tensor.spec().byteSize());
            return CompletableFuture.completedFuture(null);
        }

        // Multi-rank: use UCC (must run synchronously - UCX thread affinity)
        try (Arena opArena = Arena.ofConfined()) {
            // Set up collective args
            MemorySegment args = ucc_coll_args.allocate(opArena);
            UccHelper.setupCollectiveArgsWithRoot(args, UccConstants.COLL_TYPE_BCAST, root);
            ucc_coll_args.flags(args, UccConstants.COLL_ARGS_FLAG_IN_PLACE);

            // For in-place, src and dst are the same buffer
            MemorySegment srcInfo = UccHelper.getSrcBufferInfo(args);
            UccHelper.setupBufferInfo(srcInfo, tensor);

            MemorySegment dstInfo = UccHelper.getDstBufferInfo(args);
            UccHelper.setupBufferInfo(dstInfo, tensor);

            // Initialize and post collective
            MemorySegment requestPtr = opArena.allocate(ValueLayout.ADDRESS);
            UccHelper.initAndPostCollective(args, requestPtr, uccTeam, "broadcast_inplace");

            // Wait for completion
            MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
            UccHelper.waitForCompletionWithProgress(request, uccContext);

            broadcastCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(tensor.spec().byteSize());
        }
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Tensor> reduceScatter(Tensor input, AllReduceOp op) {
        checkInitialized();
        checkNotClosed();

        // Single-rank optimization: no communication needed, just copy
        if (config.worldSize() == 1) {
            Tensor result = Tensor.zeros(input.dtype(), input.shape());
            MemorySegment.copy(input.data(), 0, result.data(), 0, input.spec().byteSize());
            reduceScatterCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());
            return CompletableFuture.completedFuture(result);
        }

        // Multi-rank: use UCC (must run synchronously - UCX thread affinity)
        try (Arena opArena = Arena.ofConfined()) {
            // Allocate output tensor with input size / worldSize
            int[] newShape = input.shape().clone();
            newShape[0] /= config.worldSize();
            Tensor result = Tensor.zeros(input.dtype(), newShape);

            // Set up collective args
            MemorySegment args = ucc_coll_args.allocate(opArena);
            UccHelper.setupCollectiveArgsWithOp(args, UccConstants.COLL_TYPE_REDUCE_SCATTER, allReduceOpToUcc(op));

            // Configure source buffer
            MemorySegment srcInfo = UccHelper.getSrcBufferInfo(args);
            UccHelper.setupBufferInfo(srcInfo, input);

            // Configure destination buffer
            MemorySegment dstInfo = UccHelper.getDstBufferInfo(args);
            UccHelper.setupBufferInfo(dstInfo, result);

            // Initialize and post collective
            MemorySegment requestPtr = opArena.allocate(ValueLayout.ADDRESS);
            UccHelper.initAndPostCollective(args, requestPtr, uccTeam, "reduce_scatter");

            // Wait for completion
            MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
            UccHelper.waitForCompletionWithProgress(request, uccContext);

            // Update statistics
            reduceScatterCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());

            return CompletableFuture.completedFuture(result);
        }
    }

    @Override
    public CompletableFuture<Void> reduceScatter(Tensor input, Tensor output, AllReduceOp op) {
        checkInitialized();
        checkNotClosed();

        // Single-rank optimization: no communication needed, just copy
        if (config.worldSize() == 1) {
            MemorySegment.copy(input.data(), 0, output.data(), 0, input.spec().byteSize());
            reduceScatterCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());
            return CompletableFuture.completedFuture(null);
        }

        // Multi-rank: use UCC (must run synchronously - UCX thread affinity)
        try (Arena opArena = Arena.ofConfined()) {
            // Set up collective args
            MemorySegment args = ucc_coll_args.allocate(opArena);
            UccHelper.setupCollectiveArgsWithOp(args, UccConstants.COLL_TYPE_REDUCE_SCATTER, allReduceOpToUcc(op));

            // Configure source buffer
            MemorySegment srcInfo = UccHelper.getSrcBufferInfo(args);
            UccHelper.setupBufferInfo(srcInfo, input);

            // Configure destination buffer
            MemorySegment dstInfo = UccHelper.getDstBufferInfo(args);
            UccHelper.setupBufferInfo(dstInfo, output);

            // Initialize and post collective
            MemorySegment requestPtr = opArena.allocate(ValueLayout.ADDRESS);
            UccHelper.initAndPostCollective(args, requestPtr, uccTeam, "reduce_scatter_into");

            // Wait for completion
            MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
            UccHelper.waitForCompletionWithProgress(request, uccContext);

            reduceScatterCount.incrementAndGet();
            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());
        }
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Tensor> allToAll(Tensor input) {
        checkInitialized();
        checkNotClosed();

        // Single-rank optimization: no communication needed, just copy
        if (config.worldSize() == 1) {
            Tensor result = Tensor.zeros(input.dtype(), input.shape());
            MemorySegment.copy(input.data(), 0, result.data(), 0, input.spec().byteSize());
            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());
            return CompletableFuture.completedFuture(result);
        }

        // Multi-rank: use UCC (must run synchronously - UCX thread affinity)
        try (Arena opArena = Arena.ofConfined()) {
            // Output has same shape as input for all-to-all
            Tensor result = Tensor.zeros(input.dtype(), input.shape());

            // Set up collective args
            MemorySegment args = ucc_coll_args.allocate(opArena);
            UccHelper.setupCollectiveArgs(args, UccConstants.COLL_TYPE_ALLTOALL);

            // Configure source buffer
            MemorySegment srcInfo = UccHelper.getSrcBufferInfo(args);
            UccHelper.setupBufferInfo(srcInfo, input);

            // Configure destination buffer
            MemorySegment dstInfo = UccHelper.getDstBufferInfo(args);
            UccHelper.setupBufferInfo(dstInfo, result);

            // Initialize and post collective
            MemorySegment requestPtr = opArena.allocate(ValueLayout.ADDRESS);
            UccHelper.initAndPostCollective(args, requestPtr, uccTeam, "alltoall");

            // Wait for completion
            MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
            UccHelper.waitForCompletionWithProgress(request, uccContext);

            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());
            return CompletableFuture.completedFuture(result);
        }
    }

    @Override
    public CompletableFuture<Void> allToAll(Tensor input, Tensor output) {
        checkInitialized();
        checkNotClosed();

        // Single-rank optimization: no communication needed, just copy
        if (config.worldSize() == 1) {
            MemorySegment.copy(input.data(), 0, output.data(), 0, input.spec().byteSize());
            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());
            return CompletableFuture.completedFuture(null);
        }

        // Multi-rank: use UCC (must run synchronously - UCX thread affinity)
        try (Arena opArena = Arena.ofConfined()) {
            // Set up collective args
            MemorySegment args = ucc_coll_args.allocate(opArena);
            UccHelper.setupCollectiveArgs(args, UccConstants.COLL_TYPE_ALLTOALL);

            // Configure source buffer
            MemorySegment srcInfo = UccHelper.getSrcBufferInfo(args);
            UccHelper.setupBufferInfo(srcInfo, input);

            // Configure destination buffer
            MemorySegment dstInfo = UccHelper.getDstBufferInfo(args);
            UccHelper.setupBufferInfo(dstInfo, output);

            // Initialize and post collective
            MemorySegment requestPtr = opArena.allocate(ValueLayout.ADDRESS);
            UccHelper.initAndPostCollective(args, requestPtr, uccTeam, "alltoall_into");

            // Wait for completion
            MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
            UccHelper.waitForCompletionWithProgress(request, uccContext);

            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());
        }
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Tensor> reduce(Tensor input, AllReduceOp op, int root) {
        checkInitialized();
        checkNotClosed();
        validateRank(root);

        // Single-rank optimization: no communication needed, just copy
        if (config.worldSize() == 1) {
            Tensor result = Tensor.zeros(input.dtype(), input.shape());
            MemorySegment.copy(input.data(), 0, result.data(), 0, input.spec().byteSize());
            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());
            return CompletableFuture.completedFuture(result);
        }

        // Multi-rank: use UCC (must run synchronously - UCX thread affinity)
        try (Arena opArena = Arena.ofConfined()) {
            // Result is only valid at root, but all ranks need output buffer
            Tensor result = Tensor.zeros(input.dtype(), input.shape());

            // Set up collective args
            MemorySegment args = ucc_coll_args.allocate(opArena);
            UccHelper.setupCollectiveArgsWithOpAndRoot(args, UccConstants.COLL_TYPE_REDUCE, allReduceOpToUcc(op), root);

            // Configure source buffer
            MemorySegment srcInfo = UccHelper.getSrcBufferInfo(args);
            UccHelper.setupBufferInfo(srcInfo, input);

            // Configure destination buffer
            MemorySegment dstInfo = UccHelper.getDstBufferInfo(args);
            UccHelper.setupBufferInfo(dstInfo, result);

            // Initialize and post collective
            MemorySegment requestPtr = opArena.allocate(ValueLayout.ADDRESS);
            UccHelper.initAndPostCollective(args, requestPtr, uccTeam, "reduce");

            // Wait for completion
            MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
            UccHelper.waitForCompletionWithProgress(request, uccContext);

            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());
            return CompletableFuture.completedFuture(result);
        }
    }

    @Override
    public CompletableFuture<Tensor> scatter(Tensor input, int root) {
        checkInitialized();
        checkNotClosed();
        validateRank(root);

        // Single-rank optimization: no communication needed, just copy
        if (config.worldSize() == 1) {
            Tensor result = Tensor.zeros(input.dtype(), input.shape());
            MemorySegment.copy(input.data(), 0, result.data(), 0, input.spec().byteSize());
            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());
            return CompletableFuture.completedFuture(result);
        }

        // Multi-rank: use UCC (must run synchronously - UCX thread affinity)
        try (Arena opArena = Arena.ofConfined()) {
            // Each rank receives input.size / worldSize elements
            int[] newShape = input.shape().clone();
            newShape[0] /= config.worldSize();
            Tensor result = Tensor.zeros(input.dtype(), newShape);

            // Set up collective args
            MemorySegment args = ucc_coll_args.allocate(opArena);
            UccHelper.setupCollectiveArgsWithRoot(args, UccConstants.COLL_TYPE_SCATTER, root);

            // Configure source buffer (root has the full data)
            MemorySegment srcInfo = UccHelper.getSrcBufferInfo(args);
            if (config.rank() == root) {
                UccHelper.setupBufferInfo(srcInfo, input);
            } else {
                // Non-root ranks don't need valid source data, but must set buffer info
                UccHelper.setupBufferInfo(srcInfo, result);
            }

            // Configure destination buffer
            MemorySegment dstInfo = UccHelper.getDstBufferInfo(args);
            UccHelper.setupBufferInfo(dstInfo, result);

            // Initialize and post collective
            MemorySegment requestPtr = opArena.allocate(ValueLayout.ADDRESS);
            UccHelper.initAndPostCollective(args, requestPtr, uccTeam, "scatter");

            // Wait for completion
            MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
            UccHelper.waitForCompletionWithProgress(request, uccContext);

            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize() / config.worldSize());
            return CompletableFuture.completedFuture(result);
        }
    }

    @Override
    public CompletableFuture<Tensor> gather(Tensor input, int root) {
        checkInitialized();
        checkNotClosed();
        validateRank(root);

        // Single-rank optimization: no communication needed, just copy
        if (config.worldSize() == 1) {
            Tensor result = Tensor.zeros(input.dtype(), input.shape());
            MemorySegment.copy(input.data(), 0, result.data(), 0, input.spec().byteSize());
            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());
            return CompletableFuture.completedFuture(result);
        }

        // Multi-rank: use UCC (must run synchronously - UCX thread affinity)
        try (Arena opArena = Arena.ofConfined()) {
            // Root receives worldSize * input.size elements, others get empty tensor
            Tensor result;
            if (config.rank() == root) {
                int[] newShape = input.shape().clone();
                newShape[0] *= config.worldSize();
                result = Tensor.zeros(input.dtype(), newShape);
            } else {
                result = Tensor.zeros(input.dtype(), new int[]{0});
            }

            // Set up collective args
            MemorySegment args = ucc_coll_args.allocate(opArena);
            UccHelper.setupCollectiveArgsWithRoot(args, UccConstants.COLL_TYPE_GATHER, root);

            // Configure source buffer
            MemorySegment srcInfo = UccHelper.getSrcBufferInfo(args);
            UccHelper.setupBufferInfo(srcInfo, input);

            // Configure destination buffer (only meaningful at root)
            MemorySegment dstInfo = UccHelper.getDstBufferInfo(args);
            if (config.rank() == root) {
                UccHelper.setupBufferInfo(dstInfo, result);
            } else {
                UccHelper.setupBufferInfo(dstInfo, input);
            }

            // Initialize and post collective
            MemorySegment requestPtr = opArena.allocate(ValueLayout.ADDRESS);
            UccHelper.initAndPostCollective(args, requestPtr, uccTeam, "gather");

            // Wait for completion
            MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
            UccHelper.waitForCompletionWithProgress(request, uccContext);

            totalOps.incrementAndGet();
            totalBytes.addAndGet(input.spec().byteSize());
            return CompletableFuture.completedFuture(result);
        }
    }

    @Override
    public CompletableFuture<Void> barrier() {
        checkInitialized();
        checkNotClosed();

        // Single-rank optimization: no communication needed, barrier is no-op
        if (config.worldSize() == 1) {
            barrierCount.incrementAndGet();
            totalOps.incrementAndGet();
            return CompletableFuture.completedFuture(null);
        }

        // Multi-rank: implement barrier via allreduce
        // UCC's native barrier (CL_HIER) requires sbgp node which isn't available
        // in our 2-node 1-process-per-node setup. Use allreduce as a barrier instead.
        // Note: Must run synchronously - UCX requires same thread as context creation.
        try (Arena opArena = Arena.ofConfined()) {
            // Allocate separate src and dst buffers for allreduce
            MemorySegment srcBuffer = opArena.allocate(ValueLayout.JAVA_INT);
            MemorySegment dstBuffer = opArena.allocate(ValueLayout.JAVA_INT);
            srcBuffer.set(ValueLayout.JAVA_INT, 0, 1);  // Dummy value

            MemorySegment args = ucc_coll_args.allocate(opArena);
            UccHelper.setupCollectiveArgsWithOp(args, UccConstants.COLL_TYPE_ALLREDUCE, UccConstants.OP_SUM);
            // Don't set IN_PLACE flag - use separate buffers

            // Configure source buffer
            MemorySegment srcInfo = UccHelper.getSrcBufferInfo(args);
            ucc_coll_buffer_info.buffer(srcInfo, srcBuffer);
            ucc_coll_buffer_info.count(srcInfo, 1L);
            ucc_coll_buffer_info.datatype(srcInfo, UccConstants.DT_INT32);
            ucc_coll_buffer_info.mem_type(srcInfo, UccConstants.MEMORY_TYPE_HOST);

            // Configure destination buffer (separate from source)
            MemorySegment dstInfo = UccHelper.getDstBufferInfo(args);
            ucc_coll_buffer_info.buffer(dstInfo, dstBuffer);
            ucc_coll_buffer_info.count(dstInfo, 1L);
            ucc_coll_buffer_info.datatype(dstInfo, UccConstants.DT_INT32);
            ucc_coll_buffer_info.mem_type(dstInfo, UccConstants.MEMORY_TYPE_HOST);

            // Use two-step init + post (some UCC builds don't implement init_and_post)
            MemorySegment requestPtr = opArena.allocate(ValueLayout.ADDRESS);
            int status = Ucc.ucc_collective_init(args, requestPtr, uccTeam);
            UccHelper.checkStatusAllowInProgress(status, "barrier (via allreduce) init");

            MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
            status = Ucc.ucc_collective_post(request);
            UccHelper.checkStatusAllowInProgress(status, "barrier (via allreduce) post");

            UccHelper.waitForCompletionWithProgress(request, uccContext);

            barrierCount.incrementAndGet();
            totalOps.incrementAndGet();
        }
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Void> send(Tensor tensor, int destRank, int tag) {
        checkInitialized();
        checkNotClosed();
        validateRank(destRank);

        // Single-rank optimization: can't send to yourself in point-to-point
        if (config.worldSize() == 1) {
            throw new CollectiveException(
                "Cannot send in single-rank mode",
                CollectiveException.ErrorCode.INVALID_STATE
            );
        }

        // Point-to-point via UCX directly (not UCC) - not yet implemented
        totalOps.incrementAndGet();
        totalBytes.addAndGet(tensor.spec().byteSize());
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Void> recv(Tensor tensor, int srcRank, int tag) {
        checkInitialized();
        checkNotClosed();
        validateRank(srcRank);

        // Single-rank optimization: can't receive from yourself in point-to-point
        if (config.worldSize() == 1) {
            throw new CollectiveException(
                "Cannot recv in single-rank mode",
                CollectiveException.ErrorCode.INVALID_STATE
            );
        }

        // Point-to-point via UCX directly (not UCC) - not yet implemented
        totalOps.incrementAndGet();
        totalBytes.addAndGet(tensor.spec().byteSize());
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CollectiveStats stats() {
        return new CollectiveStats(
                allReduceCount.get(),
                allGatherCount.get(),
                broadcastCount.get(),
                reduceScatterCount.get(),
                barrierCount.get(),
                totalBytes.get(),
                totalOps.get()
        );
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        LOG.info("Closing UCC collective for rank " + config.rank());

        // Shutdown progress thread first (before destroying UCC resources)
        if (progressThread != null) {
            try {
                progressThread.shutdown();
            } catch (Exception e) {
                LOG.log(Level.WARNING, "Error shutting down progress thread", e);
            }
        }

        // Cleanup UCC resources in reverse order
        try {
            if (uccTeam != null && !UccHelper.isNull(uccTeam)) {
                int status = Ucc.ucc_team_destroy(uccTeam);
                if (status != UccConstants.OK) {
                    LOG.warning("ucc_team_destroy returned: " + UccConstants.statusToString(status));
                }
            }
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Error destroying UCC team", e);
        }

        try {
            if (uccContext != null && !UccHelper.isNull(uccContext)) {
                int status = Ucc.ucc_context_destroy(uccContext);
                if (status != UccConstants.OK) {
                    LOG.warning("ucc_context_destroy returned: " + UccConstants.statusToString(status));
                }
            }
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Error destroying UCC context", e);
        }

        try {
            if (uccLib != null && !UccHelper.isNull(uccLib)) {
                int status = Ucc.ucc_finalize(uccLib);
                if (status != UccConstants.OK) {
                    LOG.warning("ucc_finalize returned: " + UccConstants.statusToString(status));
                }
            }
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Error finalizing UCC library", e);
        }

        // Close OOB coordinator
        if (oobCoordinator != null) {
            try {
                oobCoordinator.close();
            } catch (Exception e) {
                LOG.log(Level.WARNING, "Error closing OOB coordinator", e);
            }
        }

        // Close arena pool
        if (arenaPool != null) {
            try {
                arenaPool.close();
            } catch (Exception e) {
                LOG.log(Level.WARNING, "Error closing arena pool", e);
            }
        }

        // Close main arena
        try {
            arena.close();
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Error closing arena", e);
        }

        LOG.fine("UCC collective closed for rank " + config.rank());
    }

    private void checkInitialized() {
        if (!initialized) {
            throw new CollectiveException("UCC not initialized", CollectiveException.ErrorCode.NOT_INITIALIZED);
        }
    }

    private void checkNotClosed() {
        if (closed) {
            throw new CollectiveException("Collective context has been closed",
                    CollectiveException.ErrorCode.INVALID_STATE);
        }
    }

    private void validateRank(int rank) {
        if (rank < 0 || rank >= config.worldSize()) {
            throw CollectiveException.invalidRank(rank, config.worldSize());
        }
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /**
     * Convert AllReduceOp to UCC reduction operation constant.
     */
    private int allReduceOpToUcc(AllReduceOp op) {
        return switch (op) {
            case SUM -> UccConstants.OP_SUM;
            case PROD -> UccConstants.OP_PROD;
            case MIN -> UccConstants.OP_MIN;
            case MAX -> UccConstants.OP_MAX;
            case AVG -> UccConstants.OP_AVG;
            case LAND -> UccConstants.OP_LAND;
            case LOR -> UccConstants.OP_LOR;
            case BAND -> UccConstants.OP_BAND;
            case BOR -> UccConstants.OP_BOR;
            case BXOR -> UccConstants.OP_BXOR;
            case MINLOC -> UccConstants.OP_MINLOC;
            case MAXLOC -> UccConstants.OP_MAXLOC;
        };
    }

    /**
     * Convert raw datatype int to UCC datatype constant.
     * The raw int is expected to match CollectiveApi datatype constants.
     */
    private long uccDatatypeFromInt(int datatype) {
        // Map CollectiveApi.DTYPE_* to UCC datatypes
        return switch (datatype) {
            case 0 -> UccConstants.DT_FLOAT32;  // DTYPE_FLOAT32
            case 1 -> UccConstants.DT_FLOAT64;  // DTYPE_FLOAT64
            case 2 -> UccConstants.DT_FLOAT16;  // DTYPE_FLOAT16
            case 3 -> UccConstants.DT_BFLOAT16; // DTYPE_BFLOAT16
            case 4 -> UccConstants.DT_INT32;    // DTYPE_INT32
            case 5 -> UccConstants.DT_INT64;    // DTYPE_INT64
            case 6 -> UccConstants.DT_INT8;     // DTYPE_INT8
            case 7 -> UccConstants.DT_INT16;    // DTYPE_INT16
            default -> throw new IllegalArgumentException("Unknown datatype: " + datatype);
        };
    }
}
