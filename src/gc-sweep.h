// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef JL_GC_SWEEP_H
#define JL_GC_SWEEP_H

#include "gc.h"

#ifdef __cplusplus
extern "C" {
#endif

// GC knobs and self-measurement variables
extern int64_t last_gc_total_bytes;

// max_total_memory is a suggestion.  We try very hard to stay
// under this limit, but we will go above it rather than halting.
#ifdef _P64
typedef uint64_t memsize_t;
#define default_collect_interval (5600 * 1024 * sizeof(void *))
#else
typedef uint32_t memsize_t;
#define default_collect_interval (3200 * 1024 * sizeof(void *))
#endif
extern size_t max_collect_interval;
extern memsize_t max_total_memory;

// Full collection heuristics
extern int64_t live_bytes;
extern int64_t promoted_bytes;
extern int64_t last_live_bytes; // live_bytes at last collection
extern int64_t t_start; // Time GC starts;
extern int64_t lazy_freed_pages;
#ifdef __GLIBC__
// maxrss at last malloc_trim
extern int64_t last_trim_maxrss;
#endif

extern int prev_sweep_full;

void gc_sweep_weak_refs(void);
bigval_t **gc_sweep_big_list(int sweep_full, bigval_t **pv) JL_NOTSAFEPOINT;
void gc_sweep_big(jl_ptls_t ptls, int sweep_full) JL_NOTSAFEPOINT;
void gc_free_array(jl_array_t *a) JL_NOTSAFEPOINT;
void gc_sweep_malloced_arrays(void) JL_NOTSAFEPOINT;

extern jl_taggedvalue_t *reset_page(const jl_gc_pool_t *p, jl_gc_pagemeta_t *pg,
                                    jl_taggedvalue_t *fl) JL_NOTSAFEPOINT;

// Returns pointer to terminal pointer of list rooted at *pfl.
STATIC_INLINE jl_taggedvalue_t **sweep_page(jl_gc_pool_t *p, jl_gc_pagemeta_t *pg,
                                            jl_taggedvalue_t **pfl, int sweep_full,
                                            int osize) JL_NOTSAFEPOINT
{
    char *data = pg->data;
    uint8_t *ages = pg->ages;
    jl_taggedvalue_t *v = (jl_taggedvalue_t *)(data + GC_PAGE_OFFSET);
    char *lim = (char *)v + GC_PAGE_SZ - GC_PAGE_OFFSET - osize;
    size_t old_nfree = pg->nfree;
    size_t nfree;

    int freedall = 1;
    int pg_skpd = 1;
    if (!pg->has_marked) {
        // lazy version: (empty) if the whole page was already unused, free it (return it to
        // the pool) eager version: (freedall) free page as soon as possible the eager one
        // uses less memory.
        // FIXME - need to do accounting on a per-thread basis
        // on quick sweeps, keep a few pages empty but allocated for performance
        if (!sweep_full && lazy_freed_pages <= default_collect_interval / GC_PAGE_SZ) {
            jl_taggedvalue_t *begin = reset_page(p, pg, p->newpages);
            p->newpages = begin;
            begin->next = (jl_taggedvalue_t *)0;
            lazy_freed_pages++;
        }
        else {
            jl_gc_free_page(data);
        }
        nfree = (GC_PAGE_SZ - GC_PAGE_OFFSET) / osize;
        goto done;
    }
    // For quick sweep, we might be able to skip the page if the page doesn't
    // have any young live cell before marking.
    if (!sweep_full && !pg->has_young) {
        assert(!prev_sweep_full || pg->prev_nold >= pg->nold);
        if (!prev_sweep_full || pg->prev_nold == pg->nold) {
            // the position of the freelist begin/end in this page
            // is stored in its metadata
            if (pg->fl_begin_offset != (uint16_t)-1) {
                *pfl = page_pfl_beg(pg);
                pfl = (jl_taggedvalue_t **)page_pfl_end(pg);
            }
            freedall = 0;
            nfree = pg->nfree;
            goto done;
        }
    }

    pg_skpd = 0;
    { // scope to avoid clang goto errors
        int has_marked = 0;
        int has_young = 0;
        int16_t prev_nold = 0;
        int pg_nfree = 0;
        jl_taggedvalue_t **pfl_begin = NULL;
        uint8_t msk = 1; // mask for the age bit in the current age byte
        while ((char *)v <= lim) {
            int bits = v->bits.gc;
            if (!gc_marked(bits)) {
                *pfl = v;
                pfl = &v->next;
                pfl_begin = pfl_begin ? pfl_begin : pfl;
                pg_nfree++;
                *ages &= ~msk;
            }
            else { // marked young or old
                if (*ages & msk || bits == GC_OLD_MARKED) { // old enough
                    // `!age && bits == GC_OLD_MARKED` is possible for
                    // non-first-class objects like `jl_binding_t`
                    if (sweep_full || bits == GC_MARKED) {
                        bits = v->bits.gc = GC_OLD; // promote
                    }
                    prev_nold++;
                }
                else {
                    assert(bits == GC_MARKED);
                    bits = v->bits.gc = GC_CLEAN; // unmark
                    has_young = 1;
                }
                has_marked |= gc_marked(bits);
                *ages |= msk;
                freedall = 0;
            }
            v = (jl_taggedvalue_t *)((char *)v + osize);
            msk <<= 1;
            if (!msk) {
                msk = 1;
                ages++;
            }
        }

        assert(!freedall);
        pg->has_marked = has_marked;
        pg->has_young = has_young;
        if (pfl_begin) {
            pg->fl_begin_offset = (char *)pfl_begin - data;
            pg->fl_end_offset = (char *)pfl - data;
        }
        else {
            pg->fl_begin_offset = -1;
            pg->fl_end_offset = -1;
        }

        pg->nfree = pg_nfree;
        if (sweep_full) {
            pg->nold = 0;
            pg->prev_nold = prev_nold;
        }
    }
    nfree = pg->nfree;

done:
    gc_time_count_page(freedall, pg_skpd);
    gc_num.freed += (nfree - old_nfree) * osize;
    return pfl;
}

// the actual sweeping over all allocated pages in a memory pool
STATIC_INLINE void sweep_pool_page(jl_taggedvalue_t ***pfl, jl_gc_pagemeta_t *pg,
                                   int sweep_full) JL_NOTSAFEPOINT
{
    int p_n = pg->pool_n;
    int t_n = pg->thread_n;
    jl_ptls_t ptls2 = jl_all_tls_states[t_n];
    jl_gc_pool_t *p = &ptls2->heap.norm_pools[p_n];
    int osize = pg->osize;
    pfl[t_n * JL_GC_N_POOLS + p_n] =
        sweep_page(p, pg, pfl[t_n * JL_GC_N_POOLS + p_n], sweep_full, osize);
}

// sweep over a pagetable0 for all allocated pages
STATIC_INLINE int sweep_pool_pagetable0(jl_taggedvalue_t ***pfl, pagetable0_t *pagetable0,
                                        int sweep_full) JL_NOTSAFEPOINT
{
    unsigned ub = 0;
    unsigned alloc = 0;
    for (unsigned pg_i = 0; pg_i <= pagetable0->ub; pg_i++) {
        uint32_t line = pagetable0->allocmap[pg_i];
        unsigned j;
        if (!line)
            continue;
        ub = pg_i;
        alloc = 1;
        for (j = 0; line; j++, line >>= 1) {
            unsigned next = ffs_u32(line);
            j += next;
            line >>= next;
            jl_gc_pagemeta_t *pg = pagetable0->meta[pg_i * 32 + j];
            sweep_pool_page(pfl, pg, sweep_full);
        }
    }
    pagetable0->ub = ub;
    return alloc;
}

// sweep over pagetable1 for all pagetable0 that may contain allocated pages
STATIC_INLINE int sweep_pool_pagetable1(jl_taggedvalue_t ***pfl, pagetable1_t *pagetable1,
                                        int sweep_full) JL_NOTSAFEPOINT
{
    unsigned ub = 0;
    unsigned alloc = 0;
    for (unsigned pg_i = 0; pg_i <= pagetable1->ub; pg_i++) {
        uint32_t line = pagetable1->allocmap0[pg_i];
        unsigned j;
        for (j = 0; line; j++, line >>= 1) {
            unsigned next = ffs_u32(line);
            j += next;
            line >>= next;
            pagetable0_t *pagetable0 = pagetable1->meta0[pg_i * 32 + j];
            if (pagetable0 && !sweep_pool_pagetable0(pfl, pagetable0, sweep_full))
                pagetable1->allocmap0[pg_i] &=
                    ~(1 << j); // no allocations found, remember that for next time
        }
        if (pagetable1->allocmap0[pg_i]) {
            ub = pg_i;
            alloc = 1;
        }
    }
    pagetable1->ub = ub;
    return alloc;
}

// sweep over all memory for all pagetable1 that may contain allocated pages
void sweep_pool_pagetable(jl_taggedvalue_t ***pfl, int sweep_full) JL_NOTSAFEPOINT;

// sweep over all memory that is being used and not in a pool
STATIC_INLINE void gc_sweep_other(jl_ptls_t ptls, int sweep_full) JL_NOTSAFEPOINT
{
    gc_sweep_malloced_arrays();
    gc_sweep_big(ptls, sweep_full);
}

STATIC_INLINE void gc_pool_sync_nfree(jl_gc_pagemeta_t *pg,
                                      jl_taggedvalue_t *last) JL_NOTSAFEPOINT
{
    assert(pg->fl_begin_offset != (uint16_t)-1);
    char *cur_pg = gc_page_data(last);
    // Fast path for page that has no allocation
    jl_taggedvalue_t *fl_beg = (jl_taggedvalue_t *)(cur_pg + pg->fl_begin_offset);
    if (last == fl_beg)
        return;
    int nfree = 0;
    do {
        nfree++;
        last = last->next;
    } while (gc_page_data(last) == cur_pg);
    pg->nfree = nfree;
}


void gc_sweep_pool(int sweep_full);
void sweep_finalizer_list(arraylist_t *list);

void gc_sweep_foreign_objs_in_list(arraylist_t *objs);
void gc_sweep_foreign_objs(void);

STATIC_INLINE void gc_sweep_perm_alloc(void)
{
    uint64_t t0 = jl_hrtime();
    gc_sweep_sysimg();
    gc_time_sysimg_end(t0);
}


#ifdef __cplusplus
}
#endif

#endif