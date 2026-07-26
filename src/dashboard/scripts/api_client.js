/**
 * api_client.js
 * Centralized API client for all boundary K fetch calls.
 */

// When hosted on a different port (like 3000 during dev) or file://, point to the backend 8000
const isDevServer = window.location.port === '3000' || window.location.protocol === 'file:';
const API_URL = isDevServer ? 'http://localhost:8000/api/v1' : '/api/v1';

class ApiClient {
  /**
   * Internal wrapper for fetch with standard error handling
   */
  static async _fetch(endpoint, options = {}) {
    const url = `${API_URL}${endpoint}`;
    try {
      const response = await fetch(url, options);
      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error(`Fetch failed for ${url}:`, error);
      throw error;
    }
  }

  /**
   * Fetch paginated and filtered alerts.
   */
  static async fetchAlerts(params = {}) {
    const query = new URLSearchParams();
    if (params.page) query.append('page', params.page);
    if (params.page_size) query.append('page_size', params.page_size);
    if (params.risk_tier) query.append('risk_tier', params.risk_tier);
    if (params.attack_class) query.append('attack_class', params.attack_class);
    
    const queryString = query.toString() ? `?${query.toString()}` : '';
    return this._fetch(`/alerts${queryString}`);
  }

  /**
   * Fetch full alert details including explanation and feature attributions.
   */
  static async fetchAlertDetails(alertId) {
    return this._fetch(`/alerts/${alertId}`);
  }

  /**
   * Fetch entity profile summary (status).
   */
  static async fetchEntityStatus(entityId) {
    return this._fetch(`/entities/${entityId}/status`);
  }

  /**
   * Fetch basic event history for an entity.
   */
  static async fetchEntityHistory(entityId, limit = 50) {
    return this._fetch(`/entities/${entityId}/history?limit=${limit}`);
  }

  /**
   * Connect to the SSE streaming endpoint.
   * @param {function} onMessage - Callback for when a new alert arrives
   * @param {function} onError - Callback for errors
   */
  static connectStream(onMessage, onError) {
    const url = `${API_URL}/stream/alerts`;
    const eventSource = new EventSource(url);
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (err) {
        console.error("Failed to parse SSE data", err);
      }
    };
    
    eventSource.onerror = (error) => {
      console.error("SSE Stream Error:", error);
      if (onError) onError(error);
    };
    
    return eventSource;
  }
}

export default ApiClient;
