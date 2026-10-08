"""Single-flight cadence: shorter windows must not inherit a 600-ms auto gap."""


def update_interval(requested_ms,window_ms,p95_ms):
    slow=min(600,max(300,int(window_ms)))
    return max(int(requested_ms),slow if p95_ms>240 else 300)
