"""The statistics behind the per-cell report: Wilson intervals, pass@k and pass^k, percentiles and a seeded bootstrap.

Every function returns None where its input cannot support the figure; the report prints that as "n/a".
Floats are rounded to 4 places, so a stored figure compares equal across platforms.
"""
import math
import random

Z = 1.959964
KS = (1, 3, 5)
DEFAULT_SEED = 20260101
DEFAULT_RESAMPLES = 10_000
DECIMALS = 4


def rnd(x):
    return None if x is None else round(x, DECIMALS)


def wilson(c: int, n: int, z: float = Z):
    """The Wilson score interval for c passes in n trials, clamped to [0, 1]; None when n is 0."""
    if n <= 0:
        return None
    p, z2 = c / n, z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return rnd(max(0.0, centre - half)), rnd(min(1.0, centre + half))


def pass_at_k(n: int, c: int, k: int):
    """The chance that at least one of k trials drawn without replacement from n (c passing) passes."""
    return None if k > n else rnd(1 - math.comb(n - c, k) / math.comb(n, k))


def pass_pow_k(n: int, c: int, k: int):
    """The chance that all k trials drawn without replacement from n (c passing) pass."""
    return None if k > n else rnd(math.comb(c, k) / math.comb(n, k))


def percentile(values, pct: int):
    """Nearest rank: the sorted value at index ceil(pct/100 * n) - 1, with `pct` a whole percent so the index is exact."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, -(-pct * len(ordered) // 100) - 1)]


def _rate(vector):
    return sum(vector) / len(vector)


def _usable(pairs):
    return [(s, n) for s, n in pairs if s and n]


def delta(skill, noskill):
    """rate(skill) - rate(no-skill), or None when either side has no trial."""
    return rnd(_rate(skill) - _rate(noskill)) if skill and noskill else None


def overall_delta(pairs):
    """(mean of the per-case deltas, cases used). A case whose either side is empty is dropped."""
    use = _usable(pairs)
    if not use:
        return None, 0
    return rnd(sum(_rate(s) - _rate(n) for s, n in use) / len(use)), len(use)


def _draw(rng, vector):
    size = len(vector)
    return sum(vector[int(rng.random() * size)] for _ in range(size)) / size


def bootstrap_ci(pairs, seed: int = DEFAULT_SEED, resamples: int = DEFAULT_RESAMPLES):
    """95% interval of the mean per-case delta of `pairs` ((skill, no-skill) 0/1 lists per case); None if no case has both.

    Only `Random.random()` is used: the Python docs promise its stream stays stable for a seed across versions.
    """
    use = _usable(pairs)
    if not use:
        return None
    rng = random.Random(seed)
    deltas = sorted(sum(_draw(rng, s) - _draw(rng, n) for s, n in use) / len(use) for _ in range(resamples))
    return rnd(deltas[25 * resamples // 1000]), rnd(deltas[-(-975 * resamples // 1000) - 1])
