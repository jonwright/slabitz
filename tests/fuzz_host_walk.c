/* ASan/UBSan fuzzer for the bounded host frame walk (doc/plan_host_walk.md).
 *
 * Reads a corpus file of inputs (a sequence of [u32le length][data] records),
 * applies a fixed-seed sequence of mutations, and calls both walk functions on
 * every mutant. A return outside {SBZ_OK, SBZ_ERR_CORRUPT, SBZ_ERR_LIMIT,
 * SBZ_ERR_UNSUPPORTED} is a bug
 * and makes the run exit nonzero (as does any sanitizer report). Prints a
 * histogram of the return codes on stdout.
 *
 * Usage: fuzz_host_walk <corpus.bin> <mutant-count>
 * Exit:  0 clean (every return in the set), 1 if any return was out of set.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host_walk.h"
#include <signal.h>
#include <sys/time.h>
#include <unistd.h>

/* Watchdog: a mutant that takes more than WATCHDOG_S seconds is saved to
 * build/fuzz_hang.bin (path from argv) and the run exits with code 3, so a
 * hang in the walk leaves its input behind instead of a bare timeout. */
#define WATCHDOG_S 10
static const uint8_t *g_cur;
static uint32_t g_cur_len;
static const char *g_hang_path = "fuzz_hang.bin";

static void on_alarm(int sig)
{
    FILE *f = fopen(g_hang_path, "wb");
    (void)sig;
    if (f) {
        fwrite(g_cur, 1, g_cur_len, f);
        fclose(f);
    }
    fprintf(stderr, "fuzz: a mutant of %u bytes ran over %d s, saved to %s\n", g_cur_len,
            WATCHDOG_S, g_hang_path);
    _exit(3);
}

static void watchdog(int on)
{
    struct itimerval t;
    memset(&t, 0, sizeof(t));
    t.it_value.tv_sec = on ? WATCHDOG_S : 0;
    setitimer(ITIMER_REAL, &t, NULL);
}

static uint32_t g_rng = 0x9E3779B9u;
static uint32_t rnd(void)
{
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

typedef struct { uint32_t len; uint8_t *data; } Seed;

/* ---- corpus ---- */

static void free_seeds(Seed *s, int n)
{
    int i;
    for (i = 0; i < n; ++i)
        free(s[i].data);
    free(s);
}

static int load_corpus(const char *path, Seed **out, int *n_out)
{
    FILE *f;
    long sz;
    uint8_t *buf;
    size_t at = 0;
    Seed *s = NULL;
    int n = 0, cap = 0;

    f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return 1;
    }
    if (fseek(f, 0, SEEK_END) != 0 || (sz = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return 1;
    }
    buf = (uint8_t *)malloc(sz ? (size_t)sz : 1u);
    if (!buf) {
        fclose(f);
        return 1;
    }
    if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return 1;
    }
    fclose(f);

    while (at + 4 <= (size_t)sz) {
        uint32_t len = (uint32_t)buf[at] | ((uint32_t)buf[at + 1] << 8) |
                       ((uint32_t)buf[at + 2] << 16) | ((uint32_t)buf[at + 3] << 24);
        at += 4;
        if (at + len > (size_t)sz) {
            fprintf(stderr, "fuzz: corpus record at %zu truncated\n", at);
            free(buf);
            free_seeds(s, n);
            return 1;
        }
        if (n == cap) {
            cap = cap ? cap * 2 : 8;
            Seed *ns = (Seed *)realloc(s, (size_t)cap * sizeof(*ns));
            if (!ns) {
                free(buf);
                free_seeds(s, n);
                return 1;
            }
            s = ns;
        }
        s[n].len = len;
        s[n].data = (uint8_t *)malloc(len ? len : 1u);
        if (!s[n].data) {
            free(buf);
            free_seeds(s, n);
            return 1;
        }
        if (len)
            memcpy(s[n].data, buf + at, len);
        ++n;
        at += len;
    }
    free(buf);
    *out = s;
    *n_out = n;
    if (n == 0)
        fprintf(stderr, "fuzz: empty corpus.\n");
    return 0;
}

/* ---- walk ---- */

static unsigned long g_ok, g_corrupt, g_limit, g_unsup, g_outset;

static void record(int rc)
{
    if (rc == SBZ_OK)
        ++g_ok;
    else if (rc == SBZ_ERR_CORRUPT)
        ++g_corrupt;
    else if (rc == SBZ_ERR_LIMIT)
        ++g_limit;
    else if (rc == SBZ_ERR_UNSUPPORTED)
        ++g_unsup;
    else {
        ++g_outset;
        fprintf(stderr, "fuzz: out-of-set return code %d\n", rc);
    }
}

/* Runs both walks on `buf` (exactly `len` bytes, all content). `frame_count` is
 * passed to Collect: the Count result when Count succeeded, else 0 (which
 * exercises Collect's "not the counted number of frames" path on corrupt
 * input). The output arrays are allocated with exactly frame_count entries
 * (at least 1 byte), so ASan catches a write past them. */
static void exercise(uint8_t *buf, uint32_t padded, uint32_t len, uint32_t frame_count)
{
    sbz_walk_error we;
    sbz_CountFramesAndBlocksInfo ci;
    sbz_OffsetAndSize *out_frames;
    sbz_FrameInfo *out_infos;
    int rc;

    memset(&we, 0, sizeof(we));
    rc = sbz_CountFramesAndBlocks(&ci, buf, padded, len, &we);
    record(rc);
    if (rc == SBZ_OK)
        frame_count = ci.frameCount;

    out_frames = (sbz_OffsetAndSize *)malloc(frame_count ? frame_count * sizeof(*out_frames) : 1u);
    out_infos = (sbz_FrameInfo *)malloc(frame_count ? frame_count * sizeof(*out_infos) : 1u);
    if (!out_frames || !out_infos) {
        fprintf(stderr, "fuzz: out of memory\n");
        exit(2);
    }
    memset(&we, 0, sizeof(we));
    rc = sbz_CollectFrames(out_frames, out_infos, frame_count, buf, padded, len, &we);
    record(rc);
    free(out_frames);
    free(out_infos);
}

/* Copies `work[0..len)` into a fresh buffer of exactly len bytes and walks it
 * with memoryBlockSizeInBytes = contentSizeInBytes = len, as the library calls
 * the walk on the caller's own buffer (no padding since the zero-copy input), so
 * ASan reports any read past the input. */
static uint32_t run_mutant(const uint8_t *work, uint32_t len, uint32_t frame_count)
{
    uint8_t *buf = (uint8_t *)malloc(len ? (size_t)len : 1u);
    if (!buf) {
        fprintf(stderr, "fuzz: out of memory\n");
        exit(2);
    }
    if (len)
        memcpy(buf, work, len);
    g_cur = buf;
    g_cur_len = len;
    watchdog(1);
    exercise(buf, len, len, frame_count);
    watchdog(0);
    free(buf);
    return len;
}

/* ---- mutations ---- */

enum { MU_FLIP, MU_TRUNC, MU_HEADER, MU_SPLICE, MU_ZERO, MU_LIMIT, MU_NUM };

static void limit_template(uint8_t *work, uint32_t *len);
static void mutate(const Seed *seeds, int n, uint8_t *work, uint32_t cap, uint32_t *len)
{
    const Seed *a = &seeds[rnd() % (uint32_t)n];
    uint32_t la = a->len;
    uint32_t p, e, i;
    uint32_t kind = rnd() % MU_NUM;

    *len = 0;
    switch (kind) {
    case MU_FLIP:
        if (la == 0)
            break;
        memcpy(work, a->data, la);
        p = rnd() % la;
        work[p] ^= (uint8_t)(1u << (rnd() & 7u));
        *len = la;
        break;
    case MU_TRUNC:
        memcpy(work, a->data, la);
        *len = rnd() % (la + 1u);
        break;
    case MU_HEADER:
        if (la == 0)
            break;
        memcpy(work, a->data, la);
        e = 1u + (rnd() % 8u);
        if (e > la)
            e = la;
        p = rnd() % (la - e + 1u);
        for (i = 0; i < e; ++i)
            work[p + i] = (uint8_t)rnd();
        *len = la;
        break;
    case MU_ZERO:
        if (la == 0)
            break;
        memcpy(work, a->data, la);
        e = 1u + (rnd() % 32u);
        if (e > la)
            e = la;
        p = rnd() % (la - e + 1u);
        memset(work + p, (uint8_t)rnd(), e);
        *len = la;
        break;
    case MU_LIMIT:
        limit_template(work, len);
        break;
    case MU_SPLICE:
    default: {
        const Seed *b = &seeds[rnd() % (uint32_t)n];
        uint32_t x = rnd() % (la + 1u);
        uint32_t y = rnd() % (b->len + 1u);
        if (x > cap)
            x = cap;
        memcpy(work, a->data, x);
        if (x + y > cap)
            y = (uint32_t)cap - x;
        memcpy(work + x, b->data, y);
        *len = x + y;
        break;
    }
    }
}

/* A tiny single-segment frame declaring a 4 GiB content size: the ParseFrame
 * content-size check rejects it as SBZ_ERR_LIMIT before any block is read. */
static void limit_template(uint8_t *work, uint32_t *len)
{
    static const uint8_t tpl[9] = {
        0x28, 0xB5, 0x2F, 0xFD, /* zstd magic (little endian) */
        0xA0,                   /* desc: single_segment | FCS flag 2 (4 bytes) */
        0xFF, 0xFF, 0xFF, 0xFF, /* content size = 0xFFFFFFFF > the 32-bit limit */
    };
    memcpy(work, tpl, sizeof(tpl));
    *len = sizeof(tpl);
}

int main(int argc, char **argv)
{
    Seed *seeds = NULL;
    int nseeds = 0;
    uint32_t count = 200000, evaluated = 0;
    int cap, smax, i;
    uint8_t *work;

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s <corpus.bin> [mutant-count]\n", argv[0]);
        return 2;
    }
    if (argc == 3)
        count = (uint32_t)strtoul(argv[2], NULL, 10);
    if (load_corpus(argv[1], &seeds, &nseeds))
        return 1;
    if (nseeds == 0)
        return 1;

    smax = seeds[0].len;
    for (i = 1; i < nseeds; ++i)
        if ((int)seeds[i].len > smax)
            smax = (int)seeds[i].len;
    cap = smax * 2 + 8;
    if (cap < 16)
        cap = 16;
    work = (uint8_t *)malloc((size_t)cap);
    if (!work)
        return 1;

    /* phase 1: every seed unmodified (guarantees OK and, for a limit bomb seed,
     * LIMIT/CORRUPT are reached). */
    for (i = 0; i < nseeds; ++i) {
        memcpy(work, seeds[i].data, seeds[i].len);
        run_mutant(work, seeds[i].len, 0);
        ++evaluated;
    }

    /* phase 2: truncation at every length for small seeds. */
    for (i = 0; i < nseeds; ++i) {
        uint32_t len;
        if (seeds[i].len > 200)
            continue;
        for (len = 0; len <= seeds[i].len; ++len) {
            memcpy(work, seeds[i].data, len);
            run_mutant(work, len, 0);
            ++evaluated;
        }
    }

    /* phase 3: fixed-seed random mutations up to `count`. */
    while (evaluated < count) {
        if ((evaluated & 0xFFFFu) == 0)
            fprintf(stderr, "fuzz: %u mutants\n", evaluated);
        uint32_t len;
        mutate(seeds, nseeds, work, (uint32_t)cap, &len);
        if (len > (uint32_t)cap)
            len = (uint32_t)cap;
        run_mutant(work, len, 0);
        ++evaluated;
    }

    printf("mutants=%u OK=%lu CORRUPT=%lu LIMIT=%lu UNSUPPORTED=%lu\n", evaluated, g_ok,
           g_corrupt, g_limit, g_unsup);
    free(work);
    free_seeds(seeds, nseeds);
    if (g_outset)
        return 1;
    return 0;
}
