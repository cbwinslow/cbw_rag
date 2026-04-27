from prometheus_client import Counter, Histogram, generate_latest

# Request counters
REQ_COUNTER = Counter(
    "cbw_rag_requests_total",
    "Total number of RAG requests",
    ["endpoint"],
)

# Latency histogram (seconds)
REQ_LATENCY = Histogram(
    "cbw_rag_request_latency_seconds",
    "Latency of RAG requests",
    ["endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10),
)

def metrics_response():
    """Return the current metrics in Prometheus text format."""
    return generate_latest()
