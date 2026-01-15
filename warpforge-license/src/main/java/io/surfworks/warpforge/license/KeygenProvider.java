package io.surfworks.warpforge.license;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.Map;

/**
 * License provider implementation for Keygen.sh.
 *
 * <p>Keygen is a purpose-built software licensing API that supports:
 * <ul>
 *   <li>Online activation and validation via REST API</li>
 *   <li>Offline validation using Ed25519 cryptographic signatures</li>
 *   <li>Machine activation and floating licenses</li>
 *   <li>License file caching for air-gapped environments</li>
 * </ul>
 *
 * <p>Configuration is provided via environment variables:
 * <ul>
 *   <li>{@code KEYGEN_ACCOUNT_ID} - Your Keygen account ID</li>
 *   <li>{@code KEYGEN_PUBLIC_KEY} - Ed25519 public key for offline validation (optional)</li>
 * </ul>
 *
 * @see <a href="https://keygen.sh/docs/api/">Keygen API Documentation</a>
 * @see <a href="https://keygen.sh/docs/api/cryptography/">Keygen Cryptography Docs</a>
 */
public class KeygenProvider implements LicenseProvider {

    /**
     * Environment variable for Keygen account ID.
     */
    public static final String ENV_ACCOUNT_ID = "KEYGEN_ACCOUNT_ID";

    /**
     * Environment variable for Keygen Ed25519 public key (hex-encoded).
     */
    public static final String ENV_PUBLIC_KEY = "KEYGEN_PUBLIC_KEY";

    /**
     * Environment variable for Keygen product ID (optional, for validation).
     */
    public static final String ENV_PRODUCT_ID = "KEYGEN_PRODUCT_ID";

    private static final Duration TIMEOUT = Duration.ofSeconds(15);
    private static final Gson GSON = new Gson();

    private final String accountId;
    private final String publicKey;
    private final String productId;
    private final HttpClient httpClient;
    private final ProductMapper productMapper;

    /**
     * Create a Keygen provider with configuration from environment variables.
     */
    public KeygenProvider() {
        this(
            System.getenv(ENV_ACCOUNT_ID),
            System.getenv(ENV_PUBLIC_KEY),
            System.getenv(ENV_PRODUCT_ID)
        );
    }

    /**
     * Create a Keygen provider with explicit configuration.
     *
     * @param accountId the Keygen account ID
     * @param publicKey the Ed25519 public key for offline validation (may be null)
     * @param productId the expected product ID for validation (may be null)
     */
    public KeygenProvider(String accountId, String publicKey, String productId) {
        this.accountId = accountId;
        this.publicKey = publicKey;
        this.productId = productId;
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(TIMEOUT)
            .build();
        this.productMapper = ProductMapper.DEFAULT;
    }

    @Override
    public String getProviderName() {
        return "Keygen";
    }

    @Override
    public boolean supportsOfflineValidation() {
        return publicKey != null && !publicKey.isBlank();
    }

    @Override
    public ActivationResult activate(String licenseKey) {
        if (accountId == null || accountId.isBlank()) {
            return ActivationResult.failure(
                "Keygen account ID not configured. Set " + ENV_ACCOUNT_ID,
                ActivationResult.ErrorCode.UNKNOWN
            );
        }

        String fingerprint = MachineFingerprint.generate();
        String machineName = MachineFingerprint.getMachineName();

        // Keygen activation uses machine creation
        // First, validate the license key to get license ID
        String apiUrl = String.format(
            "https://api.keygen.sh/v1/accounts/%s/licenses/actions/validate-key",
            accountId
        );

        JsonObject requestBody = new JsonObject();
        JsonObject meta = new JsonObject();
        meta.addProperty("key", licenseKey);
        meta.addProperty("scope", new JsonObject().toString()); // Empty scope
        requestBody.add("meta", meta);

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(apiUrl))
            .header("Accept", "application/vnd.api+json")
            .header("Content-Type", "application/vnd.api+json")
            .POST(HttpRequest.BodyPublishers.ofString(GSON.toJson(requestBody)))
            .timeout(TIMEOUT)
            .build();

        try {
            HttpResponse<String> response = httpClient.send(
                request, HttpResponse.BodyHandlers.ofString()
            );

            return parseActivationResponse(response, licenseKey, fingerprint);
        } catch (IOException e) {
            return ActivationResult.failure(
                "Network error: " + e.getMessage(),
                ActivationResult.ErrorCode.NETWORK_ERROR
            );
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return ActivationResult.failure(
                "Request interrupted",
                ActivationResult.ErrorCode.NETWORK_ERROR
            );
        }
    }

    @Override
    public ActivationResult validate(String licenseKey, String instanceId) {
        if (accountId == null || accountId.isBlank()) {
            return ActivationResult.failure(
                "Keygen account ID not configured. Set " + ENV_ACCOUNT_ID,
                ActivationResult.ErrorCode.UNKNOWN
            );
        }

        String apiUrl = String.format(
            "https://api.keygen.sh/v1/accounts/%s/licenses/actions/validate-key",
            accountId
        );

        JsonObject requestBody = new JsonObject();
        JsonObject meta = new JsonObject();
        meta.addProperty("key", licenseKey);
        requestBody.add("meta", meta);

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(apiUrl))
            .header("Accept", "application/vnd.api+json")
            .header("Content-Type", "application/vnd.api+json")
            .POST(HttpRequest.BodyPublishers.ofString(GSON.toJson(requestBody)))
            .timeout(TIMEOUT)
            .build();

        try {
            HttpResponse<String> response = httpClient.send(
                request, HttpResponse.BodyHandlers.ofString()
            );

            return parseValidationResponse(response, licenseKey);
        } catch (IOException e) {
            return ActivationResult.failure(
                "Network error: " + e.getMessage(),
                ActivationResult.ErrorCode.NETWORK_ERROR
            );
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return ActivationResult.failure(
                "Request interrupted",
                ActivationResult.ErrorCode.NETWORK_ERROR
            );
        }
    }

    @Override
    public ActivationResult validateOffline(String licenseKey) {
        if (!supportsOfflineValidation()) {
            return ActivationResult.failure(
                "Offline validation not configured. Set " + ENV_PUBLIC_KEY,
                ActivationResult.ErrorCode.UNKNOWN
            );
        }

        // TODO: Implement Ed25519 signature verification using the public key
        // This requires parsing the signed license data and verifying against publicKey
        // For now, return a placeholder indicating this needs implementation
        return ActivationResult.failure(
            "Offline validation not yet implemented. Use online validation.",
            ActivationResult.ErrorCode.UNKNOWN
        );
    }

    @Override
    public boolean deactivate(String licenseKey, String instanceId) {
        if (accountId == null || accountId.isBlank() || instanceId == null) {
            return false;
        }

        // Keygen uses machine deletion for deactivation
        String apiUrl = String.format(
            "https://api.keygen.sh/v1/accounts/%s/machines/%s",
            accountId, instanceId
        );

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(apiUrl))
            .header("Accept", "application/vnd.api+json")
            .DELETE()
            .timeout(TIMEOUT)
            .build();

        try {
            HttpResponse<String> response = httpClient.send(
                request, HttpResponse.BodyHandlers.ofString()
            );
            return response.statusCode() == 204 || response.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    private ActivationResult parseActivationResponse(
            HttpResponse<String> response, String licenseKey, String fingerprint) {

        int status = response.statusCode();
        String body = response.body();

        if (status == 200) {
            try {
                JsonObject json = JsonParser.parseString(body).getAsJsonObject();

                // Check validation result
                JsonObject meta = json.has("meta") ? json.getAsJsonObject("meta") : null;
                if (meta == null) {
                    return ActivationResult.failure("Invalid response format");
                }

                String validationStatus = meta.has("valid")
                    ? (meta.get("valid").getAsBoolean() ? "VALID" : "INVALID")
                    : meta.has("code") ? meta.get("code").getAsString() : "UNKNOWN";

                if (!"VALID".equals(validationStatus) && !"NO_MACHINES".equals(validationStatus)) {
                    String detail = meta.has("detail") ? meta.get("detail").getAsString() : "Validation failed";
                    return mapKeygenError(validationStatus, detail);
                }

                // Extract license data
                JsonObject data = json.has("data") ? json.getAsJsonObject("data") : null;
                if (data == null) {
                    return ActivationResult.failure("No license data in response");
                }

                String licenseId = data.has("id") ? data.get("id").getAsString() : null;
                JsonObject attributes = data.has("attributes") ? data.getAsJsonObject("attributes") : null;

                Instant expiresAt = null;
                String productName = null;
                String customerEmail = null;

                if (attributes != null) {
                    if (attributes.has("expiry") && !attributes.get("expiry").isJsonNull()) {
                        expiresAt = Instant.parse(attributes.get("expiry").getAsString());
                    }
                    if (attributes.has("name") && !attributes.get("name").isJsonNull()) {
                        productName = attributes.get("name").getAsString();
                    }
                }

                WarpForgeProduct product = productMapper.mapProduct(productName, null, Map.of());

                LicenseInfo license = new LicenseInfo(
                    licenseKey,
                    licenseId, // Use license ID as instance ID for now
                    product,
                    expiresAt,
                    Instant.now(),
                    Instant.now(),
                    fingerprint,
                    customerEmail,
                    Map.of("keygen_license_id", licenseId != null ? licenseId : "")
                );

                return ActivationResult.success(license);
            } catch (Exception e) {
                return ActivationResult.failure("Failed to parse response: " + e.getMessage());
            }
        } else {
            return ActivationResult.failure("Activation failed (HTTP " + status + ")");
        }
    }

    private ActivationResult parseValidationResponse(HttpResponse<String> response, String licenseKey) {
        int status = response.statusCode();
        String body = response.body();

        if (status == 200) {
            try {
                JsonObject json = JsonParser.parseString(body).getAsJsonObject();

                JsonObject meta = json.has("meta") ? json.getAsJsonObject("meta") : null;
                if (meta == null) {
                    return ActivationResult.failure("Invalid response format");
                }

                String validationStatus = meta.has("valid")
                    ? (meta.get("valid").getAsBoolean() ? "VALID" : "INVALID")
                    : meta.has("code") ? meta.get("code").getAsString() : "UNKNOWN";

                if (!"VALID".equals(validationStatus)) {
                    String detail = meta.has("detail") ? meta.get("detail").getAsString() : "Validation failed";
                    return mapKeygenError(validationStatus, detail);
                }

                // Extract license data
                JsonObject data = json.has("data") ? json.getAsJsonObject("data") : null;
                if (data == null) {
                    return ActivationResult.failure("No license data in response");
                }

                String licenseId = data.has("id") ? data.get("id").getAsString() : null;
                JsonObject attributes = data.has("attributes") ? data.getAsJsonObject("attributes") : null;

                Instant expiresAt = null;
                String productName = null;

                if (attributes != null) {
                    if (attributes.has("expiry") && !attributes.get("expiry").isJsonNull()) {
                        expiresAt = Instant.parse(attributes.get("expiry").getAsString());
                    }
                    if (attributes.has("name") && !attributes.get("name").isJsonNull()) {
                        productName = attributes.get("name").getAsString();
                    }
                }

                WarpForgeProduct product = productMapper.mapProduct(productName, null, Map.of());

                LicenseInfo license = new LicenseInfo(
                    licenseKey,
                    licenseId,
                    product,
                    expiresAt,
                    null,
                    Instant.now(),
                    MachineFingerprint.generate(),
                    null,
                    Map.of()
                );

                return ActivationResult.success(license);
            } catch (Exception e) {
                return ActivationResult.failure("Failed to parse response: " + e.getMessage());
            }
        } else {
            return ActivationResult.failure("Validation failed (HTTP " + status + ")");
        }
    }

    private ActivationResult mapKeygenError(String code, String detail) {
        return switch (code) {
            case "NOT_FOUND", "INVALID" -> ActivationResult.failure(
                "Invalid license key", ActivationResult.ErrorCode.INVALID_KEY
            );
            case "EXPIRED" -> ActivationResult.failure(
                "License has expired", ActivationResult.ErrorCode.KEY_EXPIRED
            );
            case "SUSPENDED", "BANNED" -> ActivationResult.failure(
                "License has been suspended", ActivationResult.ErrorCode.KEY_DISABLED
            );
            case "TOO_MANY_MACHINES" -> ActivationResult.failure(
                "Activation limit reached. Deactivate another device first.",
                ActivationResult.ErrorCode.ACTIVATION_LIMIT_REACHED
            );
            default -> ActivationResult.failure(detail);
        };
    }
}
