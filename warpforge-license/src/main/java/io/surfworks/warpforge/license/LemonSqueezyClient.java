package io.surfworks.warpforge.license;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.Map;

/**
 * License provider implementation for Lemon Squeezy.
 *
 * <p>API Documentation: https://docs.lemonsqueezy.com/help/licensing/license-api
 *
 * <p><strong>Note:</strong> Lemon Squeezy rejected WarpForge's application,
 * so this provider is retained for reference but should not be used.
 * Consider using {@link KeygenProvider} instead.
 *
 * @deprecated Lemon Squeezy does not support products fulfilled outside their platform.
 *             Use {@link KeygenProvider} or another provider instead.
 */
@Deprecated
public class LemonSqueezyClient implements LicenseProvider {

    private static final String API_BASE = "https://api.lemonsqueezy.com/v1/licenses";
    private static final Duration TIMEOUT = Duration.ofSeconds(15);
    private static final Gson GSON = new Gson();

    private final HttpClient httpClient;

    public LemonSqueezyClient() {
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(TIMEOUT)
            .build();
    }

    /**
     * Activate a license key, binding it to this machine.
     *
     * @param licenseKey the license key from Lemon Squeezy
     * @return activation result with license info on success
     */
    @Override
    public ActivationResult activate(String licenseKey) {
        String fingerprint = MachineFingerprint.generate();
        String instanceName = MachineFingerprint.getMachineName();

        String body = formEncode(Map.of(
            "license_key", licenseKey,
            "instance_name", instanceName
        ));

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(API_BASE + "/activate"))
            .header("Accept", "application/json")
            .header("Content-Type", "application/x-www-form-urlencoded")
            .POST(HttpRequest.BodyPublishers.ofString(body))
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

    /**
     * Validate an existing license key.
     *
     * @param licenseKey the license key
     * @param instanceId the instance ID from activation
     * @return activation result with updated license info
     */
    @Override
    public ActivationResult validate(String licenseKey, String instanceId) {
        String body = formEncode(Map.of(
            "license_key", licenseKey,
            "instance_id", instanceId
        ));

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(API_BASE + "/validate"))
            .header("Accept", "application/json")
            .header("Content-Type", "application/x-www-form-urlencoded")
            .POST(HttpRequest.BodyPublishers.ofString(body))
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
    public String getProviderName() {
        return "Lemon Squeezy";
    }

    /**
     * Deactivate a license instance.
     *
     * @param licenseKey the license key
     * @param instanceId the instance ID to deactivate
     * @return true if deactivation succeeded
     */
    @Override
    public boolean deactivate(String licenseKey, String instanceId) {
        String body = formEncode(Map.of(
            "license_key", licenseKey,
            "instance_id", instanceId
        ));

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(API_BASE + "/deactivate"))
            .header("Accept", "application/json")
            .header("Content-Type", "application/x-www-form-urlencoded")
            .POST(HttpRequest.BodyPublishers.ofString(body))
            .timeout(TIMEOUT)
            .build();

        try {
            HttpResponse<String> response = httpClient.send(
                request, HttpResponse.BodyHandlers.ofString()
            );

            return response.statusCode() == 200;
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

                boolean activated = json.has("activated") && json.get("activated").getAsBoolean();
                if (!activated) {
                    String error = json.has("error") ? json.get("error").getAsString() : "Activation failed";
                    return ActivationResult.failure(error, ActivationResult.ErrorCode.INVALID_KEY);
                }

                // Extract instance info
                String instanceId = null;
                if (json.has("instance") && json.get("instance").isJsonObject()) {
                    JsonObject instance = json.getAsJsonObject("instance");
                    instanceId = instance.has("id") ? instance.get("id").getAsString() : null;
                }

                // Extract license key info
                Instant expiresAt = null;
                if (json.has("license_key") && json.get("license_key").isJsonObject()) {
                    JsonObject lk = json.getAsJsonObject("license_key");
                    if (lk.has("expires_at") && !lk.get("expires_at").isJsonNull()) {
                        expiresAt = Instant.parse(lk.get("expires_at").getAsString());
                    }
                }

                // Extract meta info
                String productName = null;
                String variantName = null;
                String customerEmail = null;
                if (json.has("meta") && json.get("meta").isJsonObject()) {
                    JsonObject meta = json.getAsJsonObject("meta");
                    productName = meta.has("product_name") ? meta.get("product_name").getAsString() : null;
                    variantName = meta.has("variant_name") ? meta.get("variant_name").getAsString() : null;
                    customerEmail = meta.has("customer_email") ? meta.get("customer_email").getAsString() : null;
                }

                WarpForgeProduct product = WarpForgeProduct.fromLemonSqueezy(productName, variantName);

                LicenseInfo license = new LicenseInfo(
                    licenseKey,
                    instanceId,
                    product,
                    expiresAt,
                    Instant.now(),
                    Instant.now(),
                    fingerprint,
                    customerEmail,
                    Map.of("product_name", productName != null ? productName : "",
                           "variant_name", variantName != null ? variantName : "")
                );

                return ActivationResult.success(license);
            } catch (Exception e) {
                return ActivationResult.failure("Failed to parse response: " + e.getMessage());
            }
        } else if (status == 400) {
            return ActivationResult.failure("Invalid license key", ActivationResult.ErrorCode.INVALID_KEY);
        } else if (status == 422) {
            return ActivationResult.failure(
                "Activation limit reached. Deactivate another device first.",
                ActivationResult.ErrorCode.ACTIVATION_LIMIT_REACHED
            );
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

                boolean valid = json.has("valid") && json.get("valid").getAsBoolean();
                if (!valid) {
                    String error = json.has("error") ? json.get("error").getAsString() : "License invalid";
                    return ActivationResult.failure(error, ActivationResult.ErrorCode.KEY_EXPIRED);
                }

                // Extract license key info
                Instant expiresAt = null;
                String licenseStatus = null;
                if (json.has("license_key") && json.get("license_key").isJsonObject()) {
                    JsonObject lk = json.getAsJsonObject("license_key");
                    if (lk.has("expires_at") && !lk.get("expires_at").isJsonNull()) {
                        expiresAt = Instant.parse(lk.get("expires_at").getAsString());
                    }
                    licenseStatus = lk.has("status") ? lk.get("status").getAsString() : null;
                }

                if ("disabled".equals(licenseStatus)) {
                    return ActivationResult.failure("License has been disabled", ActivationResult.ErrorCode.KEY_DISABLED);
                }
                if ("expired".equals(licenseStatus)) {
                    return ActivationResult.failure("License has expired", ActivationResult.ErrorCode.KEY_EXPIRED);
                }

                // Extract meta info
                String productName = null;
                String variantName = null;
                String customerEmail = null;
                if (json.has("meta") && json.get("meta").isJsonObject()) {
                    JsonObject meta = json.getAsJsonObject("meta");
                    productName = meta.has("product_name") ? meta.get("product_name").getAsString() : null;
                    variantName = meta.has("variant_name") ? meta.get("variant_name").getAsString() : null;
                    customerEmail = meta.has("customer_email") ? meta.get("customer_email").getAsString() : null;
                }

                WarpForgeProduct product = WarpForgeProduct.fromLemonSqueezy(productName, variantName);

                LicenseInfo license = new LicenseInfo(
                    licenseKey,
                    null, // Instance ID not returned in validation
                    product,
                    expiresAt,
                    null,
                    Instant.now(),
                    MachineFingerprint.generate(),
                    customerEmail,
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

    private String formEncode(Map<String, String> params) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String> entry : params.entrySet()) {
            if (sb.length() > 0) sb.append("&");
            sb.append(URLEncoder.encode(entry.getKey(), StandardCharsets.UTF_8));
            sb.append("=");
            sb.append(URLEncoder.encode(entry.getValue(), StandardCharsets.UTF_8));
        }
        return sb.toString();
    }
}
