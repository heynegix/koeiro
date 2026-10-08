def realtime_health(stats, callback=None, deadline_ms=5.333):
    """Session counters remain visible; a restart starts a new health session."""
    if stats.get('status') == 'Error' or stats.get('rtf',0) >= 1:
        return 'Critical'
    if stats.get('ai_overrun',0) or stats.get('dropped_chunks',0) or stats.get('ai_underrun',0):
        return 'Critical'
    if callback and callback.deadline_exceeded:
        return 'Warning'
    rtf = stats.get('rtf',0)
    high = stats.get('queue_current',0) > stats.get('queue_capacity',4)*.75
    if rtf >= .75 or high:
        return 'Warning'
    if not stats.get('inference',{}).get('count'):
        return 'Waiting'
    if rtf < .5 and (not callback or callback.p99_ms < deadline_ms*.8):
        return 'Excellent'
    return 'Good'
