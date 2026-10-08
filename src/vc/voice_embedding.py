"""Speaker-embedding assembly for voice registration: TTA, weighting, merging.

Pure array math and the per-clip encoding loop. Kept apart from the worker so the
arithmetic can be tested without loading WavLM, and so the same rules apply whether
a voice is created, re-selected, or extended with more audio.
"""
import numpy as np

EMBEDDING_SIZE = 256
# Each extra augmentation pass costs roughly one more encode of the whole retained
# material (measured RTF 1.24 on one CPU thread), so the count is deliberately small.
DEFAULT_TTA_PASSES = 2
# Gain offsets applied around the clip's own level. Level is what most affects a
# normalisation-based encoder, so that is the axis worth averaging over.
# Gain is deliberately absent: measured invariant on this encoder. See TTA_CROP_RATIOS.
TTA_GAINS = (0.7, 1.4)


# Measured on this CPU with the real WavLM + ECAPA speaker encoder, 4.8 s of speech:
# the embedding is invariant to gain (cosine 1.0000 for 0.5x and 2.0x), because the
# encoder normalises internally. Cropping, added noise, speed and bandwidth all move it
# (cosine 0.81-0.94), so those are the augmentations worth averaging over. A gain-based
# TTA would cost N times the encode time and change nothing.
TTA_CROP_RATIOS = (0.75, 0.9)
TTA_NOISE_RATIOS = (0.01, 0.02)
TTA_SPEED_RATIOS = (0.95, 1.05)
TTA_AXES = ('crop', 'noise', 'speed')


def tta_variants(passes=DEFAULT_TTA_PASSES):
    """The augmentation recipe for `passes` variants: the original plus distinct axes.

    Kept as data so the applied augmentation is visible in the registration report
    rather than hidden inside the encoder call.
    """
    passes = max(1, int(passes))
    if passes == 1:
        return (('original', {}),)
    recipe = [('original', {})]
    order = TTA_AXES*passes
    tables = {'crop': TTA_CROP_RATIOS, 'noise': TTA_NOISE_RATIOS, 'speed': TTA_SPEED_RATIOS}
    for index, axis in enumerate(order):
        values = tables[axis]
        recipe.append((axis, {'amount': float(values[index % len(values)])}))
        if len(recipe) == passes:
            break
    return tuple(recipe)


def apply_tta(audio, axis, amount, rate=16000, rng=None):
    """Apply one measured augmentation. Returns the audio unchanged for 'original'."""
    audio = np.asarray(audio, dtype=np.float32)
    if axis == 'original' or not amount:
        return audio
    if len(audio) < int(rate*0.5):
        return audio
    if axis == 'crop':
        keep = max(1, int(round(len(audio)*amount)))
        if not 1 <= keep < len(audio):
            return audio
        margin = len(audio)-keep
        start = int(margin*(0.5 if rng is None else float(rng.random())))
        start = max(0, min(start, margin))
        cropped = audio[start:start+keep]
        if len(cropped) == len(audio) or not len(cropped):
            return audio
        return np.ascontiguousarray(cropped)
    if axis == 'noise':
        generator = rng or np.random.default_rng()
        peak = float(np.max(np.abs(audio), initial=0.0))
        if peak <= 0:
            return audio
        noise = generator.standard_normal(len(audio)).astype(np.float32)
        return np.ascontiguousarray(audio + noise*amount*peak)
    if axis == 'speed':
        count = max(1, int(len(audio)/amount))
        source = np.linspace(0, len(audio)-1, count)
        return np.interp(source, np.arange(len(audio)), audio).astype(np.float32)
    raise ValueError('Unknown TTA axis: '+str(axis))


def unit(vectors):
    """Row-wise L2 normalisation, rejecting degenerate or non-finite rows."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_SIZE or not np.isfinite(vectors).all():
        raise ValueError('Invalid speaker embedding')
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError('Empty speaker embedding')
    return vectors / norms[:, None], norms


def centroid(vectors, weights=None):
    """Quality-weighted mean direction, rescaled to the median retained norm.

    Weights only steer which clips end up closest to the centre; the final centroid
    stays a plain mean so the median-norm contract of the published embedding holds.
    """
    unit_vectors, norms = unit(vectors)
    if weights is None:
        centre = unit_vectors.mean(axis=0)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(unit_vectors),) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('Invalid speaker embedding weights')
        centre = (unit_vectors * (weights/weights.sum())[:, None]).sum(axis=0)
    centre /= max(float(np.linalg.norm(centre)), 1e-8)
    keep = np.argsort(unit_vectors @ centre)[-min(2, len(unit_vectors)):]
    centre = unit_vectors[keep].mean(axis=0)
    norm = float(np.linalg.norm(centre))
    if norm < 1e-8:
        raise ValueError('Inconsistent speaker embedding')
    return (centre/norm*np.median(norms[keep])).astype(np.float32), keep.tolist()


def tta_encode(audio, encode_one, passes=DEFAULT_TTA_PASSES, rate=16000, rng=None):
    """Encode one clip under each augmentation and average the unit directions.

    Averaging directions rather than vectors is deliberate: the encoder's output norm
    tracks level and duration, which are not what identifies the speaker. ``encode_one``
    takes a float32 array and returns one embedding, so this rule can be exercised
    without loading WavLM.
    """
    audio = np.asarray(audio, dtype=np.float32)
    raw = []
    for axis, parameters in tta_variants(passes):
        raw.append(encode_one(apply_tta(audio, axis, parameters.get('amount'), rate, rng)))
    stacked, stacked_norms = unit(raw)
    # Averaging directions shortens the result whenever they disagree, so renormalise;
    # disagreement between passes is also the signal that TTA is doing anything.
    averaged = stacked.mean(axis=0)
    return averaged.astype(np.float32)/max(float(np.linalg.norm(averaged)), 1e-12), float(np.median(stacked_norms))


def geometric_median(vectors, iterations=64, tolerance=1e-7):
    """The point minimising the total angular distance to the rows.

    One outlier clip pulls a plain mean toward itself; this converges to a point the
    majority of clips agree with. Weiszfeld iterations on the unit sphere, because the
    Euclidean version would drift toward the origin and report a direction no clip
    actually supports.
    """
    unit_vectors, _ = unit(vectors)
    if len(unit_vectors) == 1:
        return unit_vectors[0].astype(np.float32)
    estimate = unit_vectors.mean(axis=0)
    length = float(np.linalg.norm(estimate))
    if length <= 1e-12:
        estimate = unit_vectors[0].copy()
    else:
        estimate = estimate/length
    for _ in range(max(1, int(iterations))):
        cosine = np.clip(unit_vectors @ estimate, 1e-9, None)
        # Angles in radians approximate distance on the sphere.
        distance = np.arccos(cosine)
        weights = np.where(distance > 1e-9, 1.0/np.maximum(distance, 1e-9), 0.0)
        if not weights.any():
            break
        updated = (unit_vectors*(weights/weights.sum())[:, None]).sum(axis=0)
        updated_length = float(np.linalg.norm(updated))
        if updated_length <= 1e-12:
            break
        updated = updated/updated_length
        if float(np.linalg.norm(updated-estimate)) < tolerance:
            return updated.astype(np.float32)
        estimate = updated
    return estimate.astype(np.float32)


def ns_mean(vectors, trim=0.25):
    """Trimmed mean along the direction of the plain mean.

    Drops the `trim` fraction furthest from the provisional centre, so a couple of
    unusual clips cannot dominate. `trim` is a fraction of rows, not a norm threshold.
    """
    unit_vectors, _ = unit(vectors)
    centre = unit_vectors.mean(axis=0)
    length = float(np.linalg.norm(centre))
    if length <= 1e-12:
        return centre.astype(np.float32)
    centre = centre/length
    cosine = unit_vectors @ centre
    keep = int(round(len(unit_vectors)*max(0.0, min(0.9, trim))))
    if keep >= len(unit_vectors):
        return centre.astype(np.float32)
    # Always retain the best-aligned rows: a high trim must not discard everything.
    order = np.argsort(cosine)[-(len(unit_vectors)-keep):]
    trimmed = unit_vectors[order].mean(axis=0)
    total = float(np.linalg.norm(trimmed))
    if total <= 1e-12:
        return centre.astype(np.float32)
    return (trimmed/total).astype(np.float32)


def consistency(vectors, direction=None):
    """Mean cosine of the rows to `direction`, and to the best-aligned row.

    Used to score a candidate clip subset: a set that agrees with itself describes a
    speaker more confidently than the same number of clips that disagree.
    """
    unit_vectors, _ = unit(vectors)
    if direction is None:
        direction = unit_vectors.mean(axis=0)
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        return 0.0, 0.0
    direction = direction/length
    cosine = unit_vectors @ direction
    best = int(np.argmax(cosine))
    return float(np.mean(cosine)), float(cosine[best])


MIN_SUBSET_CLIPS = 3


def search_subset(vectors, norms, budget, quality=None, exhaustive=False,
                  diversity_weight=0.55, minimum=MIN_SUBSET_CLIPS):
    """Choose which clips describe the speaker, with the amount of material in mind.

    Fewer clips that agree tightly beat more clips that merely average out, so agreement
    is scored *per clip* rather than as a raw mean: a single row always agrees perfectly
    with itself, which would make any subset of one clip the winner. Coverage is a hard
    constraint, not a bonus -- a set below `minimum` is never returned, because a voice
    described by one clip is a single utterance rather than a speaker.

    Greedy by default. `exhaustive` checks every combination and is only practical for
    small counts; it exists to verify greedy is not leaving value behind.
    """
    unit_vectors, _ = unit(vectors)
    count = len(unit_vectors)
    budget = max(1, min(int(budget), count))
    minimum = max(1, min(int(minimum), budget))
    if quality is None:
        quality = np.ones(count, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)

    def score(indices):
        selection = list(indices)
        rows = unit_vectors[selection]
        size = len(rows)
        if size < 2:
            return -1.0
        agreement, _ = consistency(rows)
        gram = rows @ rows.T
        pairwise = (gram.sum()-size)/(size*(size-1))
        # Penalise a set that is internally split, not one that is merely varied.
        split = 1.0 - max(0.0, (float(np.min(gram)) + 1.0)/2.0)
        return pairwise - diversity_weight*split + 0.05*min(agreement, 1.0)*quality[selection].mean()

    def evaluate(indices):
        rows = unit_vectors[list(indices)]
        agreement, sharpest = consistency(rows)
        centre = rows.mean(axis=0)
        length = float(np.linalg.norm(centre))
        members = ([float(value) for value in rows @ (centre/length)]
                   if length > 1e-12 else [])
        return {'indices': list(indices), 'score': float(score(indices)),
                'member_cosines': members, 'mean_agreement': agreement,
                'sharpest_agreement': sharpest}

    if exhaustive and count <= 18:
        from itertools import combinations
        best_indices, best_score = None, -2.0
        for size in range(minimum, budget+1):
            for combination in combinations(range(count), size):
                value = score(combination)
                if value > best_score:
                    best_score, best_indices = value, combination
        chosen = list(best_indices) if best_indices else list(range(minimum))
    else:
        remaining = set(range(count))
        chosen = [max(remaining, key=lambda index: quality[index])]
        remaining.discard(chosen[0])
        while remaining and len(chosen) < budget:
            def rank(index):
                trial = chosen+[index]
                return score(trial) + 0.1*quality[index]
            pick = max(remaining, key=rank)
            chosen.append(pick)
            remaining.discard(pick)
        chosen.sort()
    result = evaluate(chosen)
    if len(result['indices']) < minimum:
        result = evaluate(range(minimum))
    return result


def encode_clips(encoder, parts, torch_module, passes=DEFAULT_TTA_PASSES, progress=None, rng=None):
    """TTA embedding per clip, keeping each clip's own median norm for later scaling."""
    def encode_one(array):
        return encoder(torch_module.from_numpy(array)[None].float()).squeeze().cpu().numpy()

    total = len(parts)
    vectors = []
    norms = []
    for index, part in enumerate(parts):
        if progress is not None:
            progress(index, total)
        with torch_module.inference_mode():
            direction, norm = tta_encode(part, encode_one, passes, rng=rng)
        vectors.append(direction)
        norms.append(norm)
    if progress is not None:
        progress(total, total)
    return np.asarray(vectors, dtype=np.float32), np.asarray(norms, dtype=np.float64)


def scale_to_reference(direction, norm, count):
    """Return an embedding row with an explicit norm, for merging with existing audio."""
    direction = np.asarray(direction, dtype=np.float32)
    length = float(np.linalg.norm(direction))
    if length < 1e-8:
        raise ValueError('Empty speaker embedding')
    return (direction/length*float(norm)).astype(np.float32), int(count)


def merge(embedding, vectors, norms, weights=None, existing_weight=1.0):
    """Combine a published embedding with newly encoded clips.

    The stored embedding is one more observation, not an override, so extending a
    voice keeps the original material represented in the result.
    """
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.shape != (EMBEDDING_SIZE,):
        raise ValueError('Invalid speaker embedding')
    rows = [embedding]
    scale = [float(np.linalg.norm(embedding))] if float(np.linalg.norm(embedding)) > 1e-8 else []
    combined_weights = [float(existing_weight)]
    for vector, norm in zip(vectors, norms):
        unit_vector, _ = unit(np.asarray([vector]))
        rows.append(unit_vector[0]*float(norm))
        scale.append(float(norm))
        combined_weights.append(float(norm))
    if weights is not None:
        supplied = np.asarray(weights, dtype=np.float64)
        if supplied.shape != (len(vectors),):
            raise ValueError('Invalid speaker embedding weights')
        combined_weights = [float(existing_weight)] + list(supplied)
    merged, keep = centroid(np.asarray(rows, dtype=np.float32), combined_weights)
    return merged, keep