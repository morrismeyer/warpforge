package io.surfworks.warpforge.launch.scheduler.ray;

import java.time.Duration;
import java.util.Objects;

/**
 * Configuration for the Ray scheduler.
 *
 * @param dashboardUrl      Ray dashboard URL (e.g., "http://localhost:8265")
 * @param connectionTimeout Timeout for establishing connections
 * @param requestTimeout    Timeout for individual requests
 */
public record RayConfig(
        String dashboardUrl,
        Duration connectionTimeout,
        Duration requestTimeout
) {

    /** Default Ray dashboard URL */
    public static final String DEFAULT_DASHBOARD_URL = "http://localhost:8265";

    /** Default connection timeout */
    public static final Duration DEFAULT_CONNECTION_TIMEOUT = Duration.ofSeconds(10);

    /** Default request timeout */
    public static final Duration DEFAULT_REQUEST_TIMEOUT = Duration.ofMinutes(5);

    public RayConfig {
        Objects.requireNonNull(dashboardUrl, "dashboardUrl cannot be null");
        Objects.requireNonNull(connectionTimeout, "connectionTimeout cannot be null");
        Objects.requireNonNull(requestTimeout, "requestTimeout cannot be null");

        if (dashboardUrl.isBlank()) {
            throw new IllegalArgumentException("dashboardUrl cannot be blank");
        }
    }

    /**
     * Creates a config for local Ray cluster with default settings.
     */
    public static RayConfig local() {
        return new RayConfig(
                DEFAULT_DASHBOARD_URL,
                DEFAULT_CONNECTION_TIMEOUT,
                DEFAULT_REQUEST_TIMEOUT
        );
    }

    /**
     * Creates a config with a custom dashboard URL and default timeouts.
     */
    public static RayConfig of(String dashboardUrl) {
        return new RayConfig(
                dashboardUrl,
                DEFAULT_CONNECTION_TIMEOUT,
                DEFAULT_REQUEST_TIMEOUT
        );
    }

    /**
     * Creates a config with a custom dashboard URL and default timeouts.
     *
     * @deprecated Use {@link #of(String)} instead.
     */
    @Deprecated
    public static RayConfig withUrl(String dashboardUrl) {
        return of(dashboardUrl);
    }

    /**
     * Returns a new config with updated connection timeout.
     */
    public RayConfig withConnectionTimeout(Duration timeout) {
        return new RayConfig(dashboardUrl, timeout, requestTimeout);
    }

    /**
     * Returns a new config with updated request timeout.
     */
    public RayConfig withRequestTimeout(Duration timeout) {
        return new RayConfig(dashboardUrl, connectionTimeout, timeout);
    }

    /**
     * Returns the base URL for the Jobs API.
     */
    public String jobsApiUrl() {
        String base = dashboardUrl.endsWith("/") ? dashboardUrl : dashboardUrl + "/";
        return base + "api/jobs/";
    }
}
