"""Experimental single-flight confirmation policy, not enabled in audio runtime."""


def needs_confirmation(context, now_seconds, last_confirmation_seconds):
    if now_seconds-last_confirmation_seconds<.8: return False
    confidence=context.get('context_confidence',0.)
    if confidence>=.8: return False
    return context.get('phrase_type','UNKNOWN')=='UNKNOWN' or any(
        .35<=context.get(k,0.)<.85 for k in ('question_probability','farewell_probability'))
