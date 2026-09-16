/* sdr.c - sparse distributed representations in one C file.
 *
 * Build: cc -std=c17 -O2 -Wall -Wextra sdr.c -o sdr_demo -lm
 * Test:  ./sdr_demo (exit 0 = all checks pass)
 * UBSan: cc -std=c17 -O1 -g -fsanitize=undefined sdr.c -o sdr_ubsan -lm && ./sdr_ubsan
 * Strict ISO C (no GNU __int128): cc -std=c17 -O2 -Wall -Wextra -Wpedantic -DSDR_NO_INT128 sdr.c -o sdr_demo -lm
 * Note: default __int128 path is a GNU extension and warns under -Wpedantic; use SDR_NO_INT128 for pedantic builds.
 * Define SDR_NO_MAIN to use as a library without the demo main.
 *
 * Ownership: sdr_t is caller-owned. sdr_init allocates bits, sdr_dispose
 * frees them. All other ops borrow pointers and never allocate.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef SDR_MALLOC
#define SDR_MALLOC(n) malloc(n)
#endif
#ifndef SDR_CALLOC
#define SDR_CALLOC(n, s) calloc((n), (s))
#endif
#ifndef SDR_FREE
#define SDR_FREE(p) free(p)
#endif

typedef enum {
    SDR_OK = 0,
    SDR_ERR_INVAL,
    SDR_ERR_RANGE,
    SDR_ERR_NOMEM
} sdr_err_t;

typedef struct {
    int n;
    int words;
    uint64_t *bits;
} sdr_t;

typedef struct {
    uint64_t state;
} sdr_rng_t;

typedef struct {
    int n, w, nbuckets, max_buckets;
    double resolution, offset;
    int *sets;
} sdr_rdse_t;

#define SDR_MAX_N (1 << 24)

const char *sdr_strerror(int e)
{
    switch (e) {
    case SDR_OK: return "ok";
    case SDR_ERR_INVAL: return "invalid argument";
    case SDR_ERR_RANGE: return "out of range";
    case SDR_ERR_NOMEM: return "out of memory";
    default: return "unknown";
    }
}

static uint64_t sdr_tail_mask(const sdr_t *r)
{
    unsigned b = (unsigned)r->n & 63u;
    if (b == 0)
        return ~0ull;
    return (1ull << b) - 1ull;
}

int sdr_init(sdr_t *r, int n)
{
    size_t nw;
    if (!r || n < 1 || n > SDR_MAX_N)
        return SDR_ERR_INVAL;
    nw = (size_t)((n + 63) / 64);
    r->bits = (uint64_t *)SDR_CALLOC(nw, sizeof *r->bits);
    if (!r->bits)
        return SDR_ERR_NOMEM;
    r->n = n;
    r->words = (int)nw;
    return SDR_OK;
}

void sdr_dispose(sdr_t *r)
{
    if (r && r->bits) {
        SDR_FREE(r->bits);
        r->bits = NULL;
    }
    if (r) {
        r->n = 0;
        r->words = 0;
    }
}

void sdr_clear(sdr_t *r)
{
    if (r && r->bits)
        memset(r->bits, 0, (size_t)r->words * sizeof *r->bits);
}

static int sdr_valid(const sdr_t *r)
{
    return r && r->bits && r->n >= 1 && r->n <= SDR_MAX_N &&
           r->words == (r->n + 63) / 64;
}

int sdr_set(sdr_t *r, int i)
{
    if (!sdr_valid(r) || i < 0 || i >= r->n)
        return SDR_ERR_INVAL;
    r->bits[(size_t)i >> 6] |= 1ull << ((unsigned)i & 63u);
    return SDR_OK;
}

int sdr_get(const sdr_t *r, int i)
{
    if (!sdr_valid(r) || i < 0 || i >= r->n)
        return 0;
    return (int)((r->bits[(size_t)i >> 6] >> (((unsigned)i) & 63u)) & 1u);
}

int sdr_copy(sdr_t *dst, const sdr_t *src)
{
    if (!sdr_valid(dst) || !sdr_valid(src) || dst->n != src->n)
        return SDR_ERR_INVAL;
    if (dst != src)
        memcpy(dst->bits, src->bits, (size_t)src->words * sizeof *src->bits);
    return SDR_OK;
}

int sdr_count(const sdr_t *r)
{
    int i, c = 0;
    if (!sdr_valid(r))
        return -1;
    for (i = 0; i < r->words; i++) {
#if defined(__GNUC__) || defined(__clang__)
        c += __builtin_popcountll(r->bits[i]);
#else
        uint64_t x = r->bits[i];
        for (; x; x &= x - 1)
            c++;
#endif
    }
    return c;
}

int sdr_overlap(const sdr_t *a, const sdr_t *b)
{
    int i, c = 0;
    if (!sdr_valid(a) || !sdr_valid(b) || a->n != b->n)
        return -1;
    for (i = 0; i < a->words; i++) {
#if defined(__GNUC__) || defined(__clang__)
        c += __builtin_popcountll(a->bits[i] & b->bits[i]);
#else
        uint64_t x = a->bits[i] & b->bits[i];
        for (; x; x &= x - 1)
            c++;
#endif
    }
    return c;
}

int sdr_equal(const sdr_t *a, const sdr_t *b)
{
    int i;
    if (!sdr_valid(a) || !sdr_valid(b) || a->n != b->n)
        return 0;
    for (i = 0; i < a->words; i++) {
        if (a->bits[i] != b->bits[i])
            return 0;
    }
    return 1;
}

int sdr_union_count(const sdr_t *a, const sdr_t *b)
{
    int i, c = 0;
    if (!sdr_valid(a) || !sdr_valid(b) || a->n != b->n)
        return -1;
    for (i = 0; i < a->words; i++) {
#if defined(__GNUC__) || defined(__clang__)
        c += __builtin_popcountll(a->bits[i] | b->bits[i]);
#else
        uint64_t x = a->bits[i] | b->bits[i];
        for (; x; x &= x - 1)
            c++;
#endif
    }
    return c;
}

int sdr_hamming(const sdr_t *a, const sdr_t *b)
{
    int i, c = 0;
    if (!sdr_valid(a) || !sdr_valid(b) || a->n != b->n)
        return -1;
    for (i = 0; i < a->words; i++) {
#if defined(__GNUC__) || defined(__clang__)
        c += __builtin_popcountll(a->bits[i] ^ b->bits[i]);
#else
        uint64_t x = a->bits[i] ^ b->bits[i];
        for (; x; x &= x - 1)
            c++;
#endif
    }
    return c;
}

int sdr_match(const sdr_t *a, const sdr_t *b, int theta)
{
    int ov;
    if (!sdr_valid(a) || !sdr_valid(b) || a->n != b->n)
        return 0;
    if (theta <= 0)
        return 1;
    ov = sdr_overlap(a, b);
    return ov >= theta;
}

int sdr_is_subset(const sdr_t *a, const sdr_t *b)
{
    int i;
    if (!sdr_valid(a) || !sdr_valid(b) || a->n != b->n)
        return 0;
    for (i = 0; i < a->words; i++) {
        if (a->bits[i] & ~b->bits[i])
            return 0;
    }
    return 1;
}

int sdr_or_into(sdr_t *dst, const sdr_t *src)
{
    int i;
    if (!sdr_valid(dst) || !sdr_valid(src) || dst->n != src->n)
        return SDR_ERR_INVAL;
    for (i = 0; i < dst->words; i++)
        dst->bits[i] |= src->bits[i];
    return SDR_OK;
}

int sdr_and_into(sdr_t *dst, const sdr_t *src)
{
    int i;
    if (!sdr_valid(dst) || !sdr_valid(src) || dst->n != src->n)
        return SDR_ERR_INVAL;
    for (i = 0; i < dst->words; i++)
        dst->bits[i] &= src->bits[i];
    return SDR_OK;
}

void sdr_rng_seed(sdr_rng_t *rng, uint64_t seed)
{
    if (rng)
        rng->state = seed ? seed : 0x9e3779b97f4a7c15ull;
}

static uint64_t sdr_rng_next(sdr_rng_t *rng)
{
    uint64_t z = (rng->state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

/* Lemire: reject the biased low region of the multiply-high product. */
static uint32_t sdr_rng_below(sdr_rng_t *rng, uint32_t bound)
{
    if (bound == 0)
        return 0;
    if (bound == 1)
        return 0;
#if defined(__SIZEOF_INT128__) && !defined(SDR_NO_INT128)
    {
        uint64_t x, hi, lo, t;
        x = sdr_rng_next(rng);
    {
        unsigned __int128 m = (unsigned __int128)x * bound;
        lo = (uint64_t)m;
        hi = (uint64_t)(m >> 64);
    }
    if (lo == 0) {
        t = (uint64_t)(~(uint64_t)0 % bound) + 1;
        while (lo < t) {
            unsigned __int128 m;
            x = sdr_rng_next(rng);
            m = (unsigned __int128)x * bound;
            lo = (uint64_t)m;
            hi = (uint64_t)(m >> 64);
        }
    }
    return (uint32_t)hi;
    }
#else
    {
        /* Canonical 32-bit Lemire: t is the rejection threshold.
         * 0u - bound wraps to 2^32 - bound, so t = (-bound) % bound. */
        uint32_t t = (uint32_t)(0u - bound) % bound;
        for (;;) {
            uint32_t r = (uint32_t)(sdr_rng_next(rng) >> 32);
            uint64_t m = (uint64_t)r * bound;
            if ((uint32_t)m >= t)
                return (uint32_t)(m >> 32);
        }
    }
#endif
}

/* Floyd: build a uniform w-subset of [0,n) in O(w) draws. */
static void sdr_floyd(sdr_rng_t *rng, int n, int w, int *out)
{
    int j, k;
    for (j = n - w; j < n; j++) {
        int t = (int)sdr_rng_below(rng, (uint32_t)(j + 1));
        int pos = j - (n - w);
        for (k = 0; k < pos; k++) {
            if (out[k] == t) {
                t = j;
                break;
            }
        }
        out[pos] = t;
    }
}

int sdr_random(sdr_t *r, sdr_rng_t *rng, int w)
{
    int *tmp;
    int i;
    if (!sdr_valid(r) || !rng || w < 0 || w > r->n)
        return SDR_ERR_INVAL;
    sdr_clear(r);
    if (w == 0)
        return SDR_OK;
    tmp = (int *)SDR_MALLOC((size_t)w * sizeof *tmp);
    if (!tmp)
        return SDR_ERR_NOMEM;
    sdr_floyd(rng, r->n, w, tmp);
    for (i = 0; i < w; i++)
        r->bits[(size_t)tmp[i] >> 6] |= 1ull << ((unsigned)tmp[i] & 63u);
    SDR_FREE(tmp);
    return SDR_OK;
}

static int sdr_active_list(const sdr_t *r, int *out, int cap)
{
    int n = 0;
    int wi;
    if (!r || !out || cap <= 0)
        return 0;
    for (wi = 0; wi < r->words && n < cap; wi++) {
        uint64_t w = r->bits[wi];
        while (w) {
#if defined(__GNUC__) || defined(__clang__)
            unsigned b = (unsigned)__builtin_ctzll(w);
#else
            unsigned b = 0;
            while (b < 64 && ((w >> b) & 1ull) == 0)
                b++;
            if (b >= 64)
                break;
#endif
            int bit = wi * 64 + (int)b;
            if (bit >= r->n)
                break;
            out[n++] = bit;
            if (n >= cap)
                break;
            w &= w - 1;
        }
    }
    return n;
}

int sdr_subsample(sdr_t *dst, const sdr_t *src, sdr_rng_t *rng, int w)
{
    int *on = NULL;
    int have, i, j;
    if (!sdr_valid(dst) || !sdr_valid(src) || dst->n != src->n || !rng)
        return SDR_ERR_INVAL;
    have = sdr_count(src);
    if (have < 0 || w < 0 || w > have)
        return SDR_ERR_RANGE;
    on = (int *)SDR_MALLOC((size_t)(have ? have : 1) * sizeof *on);
    if (!on)
        return SDR_ERR_NOMEM;
    sdr_active_list(src, on, have);
    for (i = have - 1; i > 0; i--) {
        j = (int)sdr_rng_below(rng, (uint32_t)(i + 1));
        { int t = on[i]; on[i] = on[j]; on[j] = t; }
    }
    sdr_clear(dst);
    for (i = 0; i < w; i++)
        dst->bits[(size_t)on[i] >> 6] |= 1ull << ((unsigned)on[i] & 63u);
    SDR_FREE(on);
    return SDR_OK;
}

int sdr_add_noise(sdr_t *dst, const sdr_t *src, sdr_rng_t *rng, double frac)
{
    int have, nflip, i, j, k, n, wi;
    int *on = NULL, *off = NULL;
    int non = 0, noff = 0;
    if (!sdr_valid(dst) || !sdr_valid(src) || dst->n != src->n || !rng)
        return SDR_ERR_INVAL;
    if (!(frac >= 0.0 && frac <= 1.0))
        return SDR_ERR_RANGE;
    n = src->n;
    have = sdr_count(src);
    if (have < 0)
        return SDR_ERR_INVAL;
    nflip = (int)(frac * have + 0.5);
    if (nflip > have)
        nflip = have;
    if (nflip > n - have)
        nflip = n - have;
    if (nflip == 0)
        return sdr_copy(dst, src);
    on = (int *)SDR_MALLOC((size_t)(have ? have : 1) * sizeof *on);
    off = (int *)SDR_MALLOC((size_t)((n - have) ? (n - have) : 1) * sizeof *off);
    if (!on || !off) {
        SDR_FREE(on);
        SDR_FREE(off);
        return SDR_ERR_NOMEM;
    }
    for (wi = 0; wi < src->words; wi++) {
        uint64_t w = src->bits[wi];
        uint64_t inv = ~w;
        int base = wi * 64;
        int bend = base + 64 <= n ? 64 : n - base;
        if (wi == src->words - 1)
            inv &= sdr_tail_mask(src);
        while (w) {
#if defined(__GNUC__) || defined(__clang__)
            unsigned b = (unsigned)__builtin_ctzll(w);
#else
            unsigned b = 0;
            while (b < 64 && ((w >> b) & 1ull) == 0)
                b++;
            if (b >= 64)
                break;
#endif
            if ((int)b >= bend)
                break;
            on[non++] = base + (int)b;
            w &= w - 1;
        }
        while (inv) {
#if defined(__GNUC__) || defined(__clang__)
            unsigned b = (unsigned)__builtin_ctzll(inv);
#else
            unsigned b = 0;
            while (b < 64 && ((inv >> b) & 1ull) == 0)
                b++;
            if (b >= 64)
                break;
#endif
            if ((int)b >= bend)
                break;
            off[noff++] = base + (int)b;
            inv &= inv - 1;
        }
    }
    for (i = non - 1; i > 0; i--) {
        j = (int)sdr_rng_below(rng, (uint32_t)(i + 1));
        { int t = on[i]; on[i] = on[j]; on[j] = t; }
    }
    for (i = noff - 1; i > 0; i--) {
        j = (int)sdr_rng_below(rng, (uint32_t)(i + 1));
        { int t = off[i]; off[i] = off[j]; off[j] = t; }
    }
    sdr_copy(dst, src);
    for (k = 0; k < nflip; k++) {
        dst->bits[(size_t)on[k] >> 6] &= ~(1ull << ((unsigned)on[k] & 63u));
        dst->bits[(size_t)off[k] >> 6] |= 1ull << ((unsigned)off[k] & 63u);
    }
    SDR_FREE(on);
    SDR_FREE(off);
    return SDR_OK;
}

int sdr_encode_scalar(sdr_t *r, double v, double vmin, double vmax,
                      int w, int periodic, int clip)
{
    double frac;
    long first;
    int i;
    if (!sdr_valid(r) || w < 1 || w > r->n)
        return SDR_ERR_INVAL;
    if (!(vmax > vmin) || !(v == v))
        return SDR_ERR_INVAL;
    if (periodic) {
        double range = vmax - vmin;
        double rel = fmod(v - vmin, range);
        long b;
        if (rel < 0)
            rel += range;
        b = (long)(rel / range * r->n);
        if (b >= r->n)
            b = r->n - 1;
        sdr_clear(r);
        for (i = 0; i < w; i++)
            sdr_set(r, (int)((b + i) % r->n));
        return SDR_OK;
    }
    if (r->n - w + 1 <= 0)
        return SDR_ERR_RANGE;
    if (v < vmin || v > vmax) {
        if (!clip)
            return SDR_ERR_RANGE;
        v = v < vmin ? vmin : vmax;
    }
    frac = (v - vmin) / (vmax - vmin);
    first = (long)(frac * (r->n - w) + 0.5);
    if (first < 0)
        first = 0;
    if (first > r->n - w)
        first = r->n - w;
    sdr_clear(r);
    for (i = 0; i < w; i++)
        sdr_set(r, (int)(first + i));
    return SDR_OK;
}

int sdr_encode_category(sdr_t *r, int cat, int ncats)
{
    int i, width;
    if (!sdr_valid(r) || ncats < 1 || cat < 0 || cat >= ncats)
        return SDR_ERR_INVAL;
    if (r->n % ncats != 0)
        return SDR_ERR_RANGE;
    width = r->n / ncats;
    sdr_clear(r);
    for (i = 0; i < width; i++)
        sdr_set(r, cat * width + i);
    return SDR_OK;
}

int sdr_rdse_init(sdr_rdse_t *e, int n, int w, double resolution,
                  double offset, int max_buckets)
{
    if (!e || n < 1 || n > SDR_MAX_N || w < 1 || w > n)
        return SDR_ERR_INVAL;
    if (!(resolution > 0.0) || !(resolution == resolution))
        return SDR_ERR_INVAL;
    if (!(offset == offset))
        return SDR_ERR_INVAL;
    if (n <= 6 * w || (w & 1) == 0)
        return SDR_ERR_RANGE;
    if (w > 256)
        return SDR_ERR_RANGE; /* build uses fixed 256 stack buffers */
    if (max_buckets <= 0)
        max_buckets = 1000;
    if (max_buckets > 65536)
        max_buckets = 65536;
    e->sets = (int *)SDR_MALLOC((size_t)max_buckets * (size_t)w * sizeof *e->sets);
    if (!e->sets)
        return SDR_ERR_NOMEM;
    {
        size_t total = (size_t)max_buckets * (size_t)w;
        size_t k;
        for (k = 0; k < total; k++)
            e->sets[k] = -1;
    }
    e->n = n;
    e->w = w;
    e->nbuckets = 0;
    e->max_buckets = max_buckets;
    e->resolution = resolution;
    e->offset = offset;
    return SDR_OK;
}

void sdr_rdse_free(sdr_rdse_t *e)
{
    if (e) {
        SDR_FREE(e->sets);
        e->sets = NULL;
        e->n = 0;
        e->w = 0;
        e->max_buckets = 0;
        e->resolution = 0;
        e->offset = 0;
    }
}

long sdr_rdse_bucket(const sdr_rdse_t *e, double x)
{
    double q;
    long long r, base, idx;
    if (!e || !(x == x))
        return 0;
    if (!(e->resolution > 0.0) || !(e->resolution == e->resolution))
        return (long)(e->max_buckets / 2);
    q = (x - e->offset) / e->resolution;
    if (!(q == q))
        return (long)(e->max_buckets / 2);
    /* Saturate huge or non-finite quotients to out-of-range sentinels
     * that callers already treat as SDR_ERR_RANGE. */
    if (!(q > -9.0e18 && q < 9.0e18))
        return q > 0 ? (long)e->max_buckets : (long)-1;
    r = llround(q);
    base = (long long)(e->max_buckets / 2);
    idx = base + r;
    if (idx < 0)
        return (long)-1;
    if (idx >= e->max_buckets)
        return (long)e->max_buckets;
    return (long)idx;
}

static int sdr_set_overlap(const int *a, const int *b, int w)
{
    int i, c = 0;
    for (i = 0; i < w; i++) {
        int j;
        for (j = 0; j < w; j++) {
            if (a[i] == b[j]) {
                c++;
                break;
            }
        }
    }
    return c;
}

static int sdr_in_set(const int *s, int w, int bit)
{
    int i;
    for (i = 0; i < w; i++) {
        if (s[i] == bit)
            return 1;
    }
    return 0;
}

/* True NuPIC rule: overlap falls linearly inside the w-window. */
static int sdr_rdse_ok(const sdr_rdse_t *e, int idx, const int *cand)
{
    int b, lo, hi, limit;
    /* Far buckets are skipped by the dist rule below, so only scan the
     * local window. Semantics are identical, cost drops from O(max_buckets)
     * to O(w) per check. */
    limit = e->w * 4;
    if (limit < 65)
        limit = 65;
    lo = idx - limit;
    hi = idx + limit;
    if (lo < 0)
        lo = 0;
    if (hi >= e->max_buckets)
        hi = e->max_buckets - 1;
    for (b = lo; b <= hi; b++) {
        const int *old = &e->sets[(size_t)b * (size_t)e->w];
        int dist, ov;
        if (old[0] < 0 || b == idx)
            continue;
        dist = b > idx ? b - idx : idx - b;
        if (dist >= e->w * 4 && dist > 64)
            continue;
        ov = sdr_set_overlap(old, cand, e->w);
        if (dist < e->w) {
            if (ov != e->w - dist)
                return 0;
        } else if (ov > 2) {
            return 0;
        }
    }
    return 1;
}

static int sdr_rdse_build(sdr_rdse_t *e, sdr_rng_t *rng, int idx)
{
    int *cand = &e->sets[(size_t)idx * (size_t)e->w];
    int tries;
    if (cand[0] >= 0)
        return 1;
    if (e->nbuckets == 0) {
        for (tries = 0; tries < 100000; tries++) {
            int tmp[256];
            sdr_floyd(rng, e->n, e->w, tmp);
            if (sdr_rdse_ok(e, idx, tmp)) {
                memcpy(cand, tmp, (size_t)e->w * sizeof cand[0]);
                e->nbuckets++;
                return 1;
            }
        }
        return 0;
    }
    if (idx > 0 && e->sets[(size_t)(idx - 1) * (size_t)e->w] >= 0) {
        const int *prev = &e->sets[(size_t)(idx - 1) * (size_t)e->w];
        int cur[256];
        int ri = idx % e->w;
        for (tries = 0; tries < 100000; tries++) {
            int bit = (int)sdr_rng_below(rng, (uint32_t)e->n);
            if (sdr_in_set(prev, e->w, bit))
                continue;
            memcpy(cur, prev, (size_t)e->w * sizeof cur[0]);
            cur[ri % e->w] = bit;
            if (sdr_rdse_ok(e, idx, cur)) {
                memcpy(cand, cur, (size_t)e->w * sizeof cand[0]);
                e->nbuckets++;
                return 1;
            }
        }
    }
    if (idx + 1 < e->max_buckets &&
        e->sets[(size_t)(idx + 1) * (size_t)e->w] >= 0) {
        const int *nxt = &e->sets[(size_t)(idx + 1) * (size_t)e->w];
        int cur[256];
        int ri = idx % e->w;
        for (tries = 0; tries < 100000; tries++) {
            int bit = (int)sdr_rng_below(rng, (uint32_t)e->n);
            if (sdr_in_set(nxt, e->w, bit))
                continue;
            memcpy(cur, nxt, (size_t)e->w * sizeof cur[0]);
            cur[ri % e->w] = bit;
            if (sdr_rdse_ok(e, idx, cur)) {
                memcpy(cand, cur, (size_t)e->w * sizeof cand[0]);
                e->nbuckets++;
                return 1;
            }
        }
    }
    for (tries = 0; tries < 100000; tries++) {
        int tmp[256];
        sdr_floyd(rng, e->n, e->w, tmp);
        if (sdr_rdse_ok(e, idx, tmp)) {
            memcpy(cand, tmp, (size_t)e->w * sizeof cand[0]);
            e->nbuckets++;
            return 1;
        }
    }
    return 0;
}

int sdr_rdse_encode(sdr_t *r, sdr_rdse_t *e, sdr_rng_t *rng, double x)
{
    long idx;
    int *cand;
    int k, lo, hi, b;
    if (!sdr_valid(r) || !e || !e->sets || !rng || r->n != e->n)
        return SDR_ERR_INVAL;
    if (!(x == x))
        return SDR_ERR_INVAL;
    if (e->w > 256)
        return SDR_ERR_RANGE;
    idx = sdr_rdse_bucket(e, x);
    if (idx < 0 || idx >= e->max_buckets)
        return SDR_ERR_RANGE;
    cand = &e->sets[(size_t)idx * (size_t)e->w];
    if (cand[0] < 0) {
        lo = hi = -1;
        for (b = 0; b < e->max_buckets; b++) {
            if (e->sets[(size_t)b * (size_t)e->w] >= 0) {
                if (lo < 0)
                    lo = b;
                hi = b;
            }
        }
        if (lo >= 0) {
            if (idx > hi) {
                for (b = hi + 1; b <= idx; b++) {
                    if (!sdr_rdse_build(e, rng, b))
                        return SDR_ERR_RANGE;
                }
            } else if (idx < lo) {
                for (b = lo - 1; b >= idx; b--) {
                    if (!sdr_rdse_build(e, rng, b))
                        return SDR_ERR_RANGE;
                }
            } else {
                if (!sdr_rdse_build(e, rng, (int)idx))
                    return SDR_ERR_RANGE;
            }
        } else {
            if (!sdr_rdse_build(e, rng, (int)idx))
                return SDR_ERR_RANGE;
        }
    }
    sdr_clear(r);
    for (k = 0; k < e->w; k++)
        sdr_set(r, cand[k]);
    return SDR_OK;
}

static double sdr_log_binom(int nn, int kk)
{
    int i;
    double s = 0.0;
    if (kk < 0 || kk > nn)
        return -1.0 / 0.0;
    if (kk > nn - kk)
        kk = nn - kk;
    for (i = 1; i <= kk; i++)
        s += log((double)(nn - kk + i) / (double)i);
    return s;
}

double sdr_fp_match(int n, int w, int theta)
{
    double tot, acc = 0.0;
    int b;
    if (n < 1 || w < 0 || w > n || theta <= 0)
        return 1.0;
    if (theta > w)
        return 0.0;
    tot = sdr_log_binom(n, w);
    for (b = theta; b <= w; b++)
        acc += exp(sdr_log_binom(w, b) + sdr_log_binom(n - w, w - b) - tot);
    return acc > 1.0 ? 1.0 : acc;
}

double sdr_union_bits(int n, int w, int m)
{
    double p;
    if (n < 1 || w < 0 || w > n || m < 0)
        return -1.0;
    if (w == 0 || m == 0)
        return 0.0;
    p = 1.0 - (double)w / (double)n;
    return (double)n * (1.0 - pow(p, m));
}

size_t sdr_hex_len(const sdr_t *r)
{
    if (!sdr_valid(r))
        return 0;
    return (size_t)r->words * 16u;
}

int sdr_to_hex(const sdr_t *r, char *out, size_t cap)
{
    static const char *hexd = "0123456789abcdef";
    size_t need;
    int i;
    if (!sdr_valid(r) || !out)
        return SDR_ERR_INVAL;
    need = sdr_hex_len(r) + 1;
    if (cap < need)
        return SDR_ERR_RANGE;
    for (i = 0; i < r->words; i++) {
        int k;
        for (k = 0; k < 16; k++)
            out[(size_t)i * 16 + (size_t)k] =
                hexd[(r->bits[i] >> ((15 - k) * 4)) & 15u];
    }
    out[need - 1] = 0;
    return SDR_OK;
}

int sdr_from_hex(sdr_t *r, const char *s)
{
    size_t L, i;
    unsigned v;
    if (!sdr_valid(r) || !s)
        return SDR_ERR_INVAL;
    L = strlen(s);
    if (L != sdr_hex_len(r))
        return SDR_ERR_RANGE;
    sdr_clear(r);
    for (i = 0; i < L; i++) {
        char c = s[i];
        if (c >= '0' && c <= '9')
            v = (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f')
            v = (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            v = (unsigned)(c - 'A' + 10);
        else
            return SDR_ERR_INVAL;
        r->bits[i / 16] |= (uint64_t)v << ((15 - (i % 16)) * 4);
    }
    if (r->bits[r->words - 1] & ~sdr_tail_mask(r))
        return SDR_ERR_RANGE;
    return SDR_OK;
}

#define SDR_T(c) do { if (!(c)) { fails++; printf("FAIL %s:%d: %s\n", __func__, __LINE__, #c); } } while (0)

int sdr_selftest(void)
{
    int fails = 0;
    sdr_t a, b, c;
    sdr_rng_t rng;
    sdr_rdse_t e;
    char *hex = NULL;
    int i, ov;

    memset(&a, 0, sizeof a);
    memset(&b, 0, sizeof b);
    memset(&c, 0, sizeof c);
    memset(&e, 0, sizeof e);
    SDR_T(sdr_init(NULL, 10) == SDR_ERR_INVAL);
    SDR_T(sdr_init(&a, 0) == SDR_ERR_INVAL);
    SDR_T(sdr_init(&a, 8) == SDR_OK);
    SDR_T(sdr_count(&a) == 0);
    SDR_T(sdr_set(&a, -1) == SDR_ERR_INVAL);
    SDR_T(sdr_set(&a, 8) == SDR_ERR_INVAL);
    SDR_T(sdr_set(&a, 3) == SDR_OK && sdr_get(&a, 3) == 1 && sdr_get(&a, 2) == 0);
    SDR_T(sdr_count(&a) == 1);
    sdr_dispose(&a);

    SDR_T(sdr_init(&a, 128) == SDR_OK);
    SDR_T(sdr_init(&b, 128) == SDR_OK);
    SDR_T(sdr_init(&c, 128) == SDR_OK);
    sdr_rng_seed(&rng, 1);
    SDR_T(sdr_random(&a, &rng, 10) == SDR_OK && sdr_count(&a) == 10);
    sdr_rng_seed(&rng, 1);
    SDR_T(sdr_random(&b, &rng, 10) == SDR_OK && sdr_overlap(&a, &b) == 10);
    SDR_T(sdr_subsample(&c, &a, &rng, 4) == SDR_OK && sdr_count(&c) == 4 &&
          sdr_is_subset(&c, &a));
    SDR_T(sdr_subsample(&c, &a, &rng, 11) == SDR_ERR_RANGE);
    SDR_T(sdr_add_noise(&c, &a, &rng, 0.0) == SDR_OK && sdr_overlap(&a, &c) == 10);
    SDR_T(sdr_add_noise(&c, &a, &rng, 2.0) == SDR_ERR_RANGE);
    SDR_T(sdr_add_noise(&c, &a, &rng, 0.3) == SDR_OK && sdr_count(&c) == 10);
    SDR_T(sdr_copy(&c, &a) == SDR_OK && sdr_overlap(&a, &c) == 10);
    SDR_T(sdr_or_into(&c, &b) == SDR_OK && sdr_is_subset(&a, &c));
    SDR_T(sdr_and_into(&c, &a) == SDR_OK && sdr_overlap(&c, &a) == 10);
    SDR_T(sdr_match(&a, &b, 5) == 1 && sdr_match(&a, &c, 11) == 0);
    sdr_dispose(&a);
    sdr_dispose(&b);
    sdr_dispose(&c);

    SDR_T(sdr_init(&a, 300) == SDR_OK);
    SDR_T(sdr_init(&b, 300) == SDR_OK);
    SDR_T(sdr_init(&c, 128) == SDR_OK);
    SDR_T(sdr_encode_scalar(&a, 0.5, 0.0, 1.0, 21, 0, 0) == SDR_OK &&
          sdr_count(&a) == 21);
    {
        sdr_t d;
        memset(&d, 0, sizeof d);
        SDR_T(sdr_init(&d, 300) == SDR_OK);
        SDR_T(sdr_encode_scalar(&d, 0.5036, 0.0, 1.0, 21, 0, 0) == SDR_OK);
        SDR_T(sdr_overlap(&a, &d) == 20);
        SDR_T(sdr_encode_scalar(&d, 5.0, 0.0, 1.0, 21, 0, 0) == SDR_ERR_RANGE);
        SDR_T(sdr_encode_scalar(&d, 5.0, 0.0, 1.0, 21, 0, 1) == SDR_OK);
        sdr_dispose(&d);
    }
    SDR_T(sdr_encode_scalar(&b, 0.99, 0.0, 1.0, 21, 1, 0) == SDR_OK);
    SDR_T(sdr_encode_category(&c, 1, 4) == SDR_OK);
    SDR_T(sdr_encode_category(&c, 4, 4) == SDR_ERR_INVAL);
    sdr_dispose(&a);
    sdr_dispose(&b);
    sdr_dispose(&c);

    SDR_T(sdr_fp_match(2048, 40, 20) < 1e-9);
    SDR_T(sdr_fp_match(100, 5, 6) == 0.0);
    SDR_T(sdr_fp_match(100, 5, 0) == 1.0);
    SDR_T(sdr_union_bits(2048, 40, 10) > 300.0 &&
          sdr_union_bits(2048, 40, 10) < 450.0);

    SDR_T(sdr_init(&a, 100) == SDR_OK);
    sdr_rng_seed(&rng, 7);
    SDR_T(sdr_random(&a, &rng, 10) == SDR_OK);
    hex = (char *)SDR_MALLOC(sdr_hex_len(&a) + 1);
    SDR_T(hex != NULL);
    if (hex) {
        SDR_T(sdr_to_hex(&a, hex, sdr_hex_len(&a) + 1) == SDR_OK);
        SDR_T(sdr_init(&b, 100) == SDR_OK);
        SDR_T(sdr_from_hex(&b, hex) == SDR_OK && sdr_overlap(&a, &b) == 10);
        SDR_T(sdr_from_hex(&b, "zz") == SDR_ERR_RANGE);
        hex[0] = 'g';
        SDR_T(sdr_from_hex(&b, hex) == SDR_ERR_INVAL);
        sdr_dispose(&b);
        SDR_FREE(hex);
    }
    sdr_dispose(&a);

    SDR_T(sdr_rdse_init(&e, 400, 21, 0.1, 0.0, 512) == SDR_OK);
    SDR_T(sdr_init(&a, 400) == SDR_OK);
    SDR_T(sdr_init(&b, 400) == SDR_OK);
    sdr_rng_seed(&rng, 42);
    SDR_T(sdr_rdse_encode(&a, &e, &rng, 1.0) == SDR_OK && sdr_count(&a) == 21);
    SDR_T(sdr_rdse_encode(&b, &e, &rng, 1.05) == SDR_OK);
    ov = sdr_overlap(&a, &b);
    SDR_T(ov == 20);
    {
        sdr_t z;
        memset(&z, 0, sizeof z);
        SDR_T(sdr_init(&z, 400) == SDR_OK);
        SDR_T(sdr_rdse_encode(&z, &e, &rng, 5.0) == SDR_OK);
        SDR_T(sdr_overlap(&a, &z) <= 2);
        sdr_dispose(&z);
    }
    for (i = 0; i < 20; i++) {
        sdr_t t;
        memset(&t, 0, sizeof t);
        SDR_T(sdr_init(&t, 400) == SDR_OK);
        SDR_T(sdr_rdse_encode(&t, &e, &rng, 2.0 + (double)i * 0.1) == SDR_OK &&
              sdr_count(&t) == 21);
        sdr_dispose(&t);
    }
    sdr_dispose(&a);
    sdr_dispose(&b);
    sdr_rdse_free(&e);
    SDR_T(sdr_rdse_init(&e, 100, 20, 0.1, 0.0, 0) == SDR_ERR_RANGE);
    SDR_T(sdr_rdse_init(&e, 400, 21, -1.0, 0.0, 0) == SDR_ERR_INVAL);

    SDR_T(sdr_init(&a, 128) == SDR_OK);
    SDR_T(sdr_init(&b, 128) == SDR_OK);
    sdr_rng_seed(&rng, 99);
    SDR_T(sdr_random(&a, &rng, 12) == SDR_OK);
    SDR_T(sdr_copy(&b, &a) == SDR_OK);
    SDR_T(sdr_equal(&a, &b) == 1);
    SDR_T(sdr_union_count(&a, &b) == 12 && sdr_hamming(&a, &b) == 0);
    SDR_T(sdr_random(&b, &rng, 12) == SDR_OK);
    {
        int ca = sdr_count(&a), cb = sdr_count(&b);
        int o = sdr_overlap(&a, &b);
        int u = sdr_union_count(&a, &b);
        int h = sdr_hamming(&a, &b);
        SDR_T(o >= 0 && u == ca + cb - o && h == ca + cb - 2 * o);
        SDR_T(sdr_equal(&a, &b) == (h == 0));
    }
    sdr_dispose(&a);
    sdr_dispose(&b);

    SDR_T(sdr_init(&a, 100) == SDR_OK);
    sdr_rng_seed(&rng, 7);
    SDR_T(sdr_random(&a, &rng, 10) == SDR_OK);
    SDR_T((a.bits[a.words - 1] & ~sdr_tail_mask(&a)) == 0);
    sdr_dispose(&a);

    SDR_T(sdr_rdse_init(&e, 400, 21, 0.1, 0.0, 512) == SDR_OK);
    SDR_T(sdr_rdse_bucket(&e, 1e30) == (long)e.max_buckets);
    SDR_T(sdr_rdse_bucket(&e, -1e30) == (long)-1);
    SDR_T(sdr_init(&a, 400) == SDR_OK);
    SDR_T(sdr_rdse_encode(&a, &e, &rng, 1e30) == SDR_ERR_RANGE);
    sdr_dispose(&a);
    sdr_rdse_free(&e);
    SDR_T(e.sets == NULL && e.n == 0 && e.w == 0 && e.max_buckets == 0);

    if (fails == 0)
        printf("sdr selftest: all pass\n");
    else
        printf("sdr selftest: %d FAIL\n", fails);
    return fails;
}

#ifndef SDR_NO_MAIN
int main(void)
{
    sdr_t a, b, u;
    sdr_rng_t rng;
    int ov, ou;
    double fp, ub;
    int rc;

    memset(&a, 0, sizeof a);
    memset(&b, 0, sizeof b);
    memset(&u, 0, sizeof u);
    if (sdr_selftest() != 0)
        return 1;
    if (sdr_init(&a, 2048) || sdr_init(&b, 2048) || sdr_init(&u, 2048)) {
        sdr_dispose(&a);
        sdr_dispose(&b);
        sdr_dispose(&u);
        return 2;
    }
    sdr_rng_seed(&rng, 12345);
    sdr_random(&a, &rng, 40);
    sdr_random(&b, &rng, 40);
    sdr_clear(&u);
    sdr_or_into(&u, &a);
    sdr_or_into(&u, &b);
    ov = sdr_overlap(&a, &b);
    ou = sdr_count(&u);
    fp = sdr_fp_match(2048, 40, 20);
    ub = sdr_union_bits(2048, 40, 2);
    printf("overlap=%d union=%d fp_match(20)=%.3g union_bits(2)=%.1f\n", ov, ou,
           fp, ub);
    sdr_dispose(&a);
    sdr_dispose(&b);
    sdr_dispose(&u);
    rc = (ou == 80 - ov) ? 0 : 1;
    return rc;
}
#endif
