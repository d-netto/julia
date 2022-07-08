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

extern jl_taggedvalue_t *gc_reset_page(const jl_gc_pool_t *p, jl_gc_pagemeta_t *pg,
                                       jl_taggedvalue_t *fl) JL_NOTSAFEPOINT;

// Returns pointer to terminal pointer of list rooted at *pfl.
jl_taggedvalue_t **sweep_page(jl_gc_pool_t *p, jl_gc_pagemeta_t *pg, jl_taggedvalue_t **pfl,
                              int sweep_full, int osize) JL_NOTSAFEPOINT;

// the actual sweeping over all allocated pages in a memory pool
void sweep_pool_page(jl_taggedvalue_t ***pfl, jl_gc_pagemeta_t *pg,
                     int sweep_full) JL_NOTSAFEPOINT;

// sweep over a pagetable0 for all allocated pages
int sweep_pool_pagetable0(jl_taggedvalue_t ***pfl, pagetable0_t *pagetable0,
                          int sweep_full) JL_NOTSAFEPOINT;

// sweep over pagetable1 for all pagetable0 that may contain allocated pages
int sweep_pool_pagetable1(jl_taggedvalue_t ***pfl, pagetable1_t *pagetable1,
                          int sweep_full) JL_NOTSAFEPOINT;

// sweep over all memory for all pagetable1 that may contain allocated pages
void gc_sweep_pool_pagetable(jl_taggedvalue_t ***pfl, int sweep_full) JL_NOTSAFEPOINT;

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
void gc_sweep_finalizer_list(arraylist_t *list);

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