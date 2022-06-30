// This file is a part of Julia. License is MIT: https://julialang.org/license

#include "gc.h"
#include "julia_assert.h"
#include "julia_gcext.h"
#ifdef __GLIBC__
#include <malloc.h> // for malloc_trim
#endif
#include "gc-alloc.h"
#include "gc-callbacks.h"
#include "gc-finalizers.h"
#include "gc-mark.h"
#include "gc-stats.h"
#include "gc-sweep.h"

#ifdef __cplusplus
extern "C" {
#endif

extern jl_gc_callback_list_t *gc_cblist_root_scanner;
extern jl_gc_callback_list_t *gc_cblist_task_scanner;
extern jl_gc_callback_list_t *gc_cblist_pre_gc;
extern jl_gc_callback_list_t *gc_cblist_post_gc;
extern jl_gc_callback_list_t *gc_cblist_notify_external_alloc;
extern jl_gc_callback_list_t *gc_cblist_notify_external_free;

// Protect all access to `finalizer_list_marked` and `to_finalize`.
// For accessing `ptls->finalizers`, the lock is needed if a thread
// is going to realloc the buffer (of its own list) or accessing the
// list of another thread
extern jl_mutex_t finalizers_lock;
extern uv_mutex_t gc_cache_lock;

// Flag that tells us whether we need to support conservative marking
// of objects.
static _Atomic(int) support_conservative_marking = 0;

/**
 * Note about GC synchronization:
 *
 * When entering `jl_gc_collect()`, `jl_gc_running` is atomically changed from
 * `0` to `1` to make sure that only one thread can be running the GC. Other
 * threads that enters `jl_gc_collect()` at the same time (or later calling
 * from unmanaged code) will wait in `jl_gc_collect()` until the GC is finished.
 *
 * Before starting the mark phase the GC thread calls `jl_safepoint_gc_start()`
 * and `jl_gc_wait_for_the_world()`
 * to make sure all the thread are in a safe state for the GC. The function
 * activates the safepoint and wait for all the threads to get ready for the
 * GC (`gc_state != 0`). It also acquires the `finalizers` lock so that no
 * other thread will access them when the GC is running.
 *
 * During the mark and sweep phase of the GC, the threads that are not running
 * the GC should either be running unmanaged code (or code section that does
 * not have a GC critical region mainly including storing to the stack or
 * another object) or paused at a safepoint and wait for the GC to finish.
 * If a thread want to switch from running unmanaged code to running managed
 * code, it has to perform a GC safepoint check after setting the `gc_state`
 * flag (see `jl_gc_state_save_and_set()`. it is possible that the thread might
 * have `gc_state == 0` in the middle of the GC transition back before entering
 * the safepoint. This is fine since the thread won't be executing any GC
 * critical region during that time).
 *
 * The finalizers are run after the GC finishes in normal mode (the `gc_state`
 * when `jl_gc_collect` is called) with `jl_in_finalizer = 1`. (TODO:) When we
 * have proper support of GC transition in codegen, we should execute the
 * finalizers in unmanaged (GC safe) mode.
 */

jl_gc_num_t gc_num = {0};
static size_t last_long_collect_interval;

pagetable_t memory_map;

// List of marked big objects.  Not per-thread.  Accessed only by master thread.
bigval_t *big_objects_marked = NULL;

// finalization
// `ptls->finalizers` and `finalizer_list_marked` might have tagged pointers.
// If an object pointer has the lowest bit set, the next pointer is an unboxed
// c function pointer.
// `to_finalize` should not have tagged pointers.
arraylist_t finalizer_list_marked;
arraylist_t to_finalize;
JL_DLLEXPORT _Atomic(int) jl_gc_have_pending_finalizers = 0;

NOINLINE uintptr_t gc_get_stack_ptr(void)
{
    return (uintptr_t)jl_get_frame_addr();
}

#define should_timeout() 0

static void jl_gc_wait_for_the_world(void)
{
    if (jl_n_threads > 1)
        jl_wake_libuv();
    for (int i = 0; i < jl_n_threads; i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[i];
        // This acquire load pairs with the release stores
        // in the signal handler of safepoint so we are sure that
        // all the stores on those threads are visible.
        // We're currently also using atomic store release in mutator threads
        // (in jl_gc_state_set), but we may want to use signals to flush the
        // memory operations on those threads lazily instead.
        while (!jl_atomic_load_relaxed(&ptls2->gc_state) ||
               !jl_atomic_load_acquire(&ptls2->gc_state))
            jl_cpu_pause(); // yield?
    }
}

// explicitly scheduled objects for the sweepfunc callback
static void gc_sweep_foreign_objs_in_list(arraylist_t *objs)
{
    size_t p = 0;
    for (size_t i = 0; i < objs->len; i++) {
        jl_value_t *v = (jl_value_t *)(objs->items[i]);
        jl_datatype_t *t = (jl_datatype_t *)(jl_typeof(v));
        const jl_datatype_layout_t *layout = t->layout;
        jl_fielddescdyn_t *desc = (jl_fielddescdyn_t *)jl_dt_layout_fields(layout);

        int bits = jl_astaggedvalue(v)->bits.gc;
        if (!gc_marked(bits))
            desc->sweepfunc(v);
        else
            objs->items[p++] = v;
    }
    objs->len = p;
}

static void gc_sweep_foreign_objs(void)
{
    for (int i = 0; i < jl_n_threads; i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[i];
        gc_sweep_foreign_objs_in_list(&ptls2->sweep_objs);
    }
}

extern int64_t last_gc_total_bytes;

// global variables for GC stats

// Resetting the object to a young object, this is used when marking the
// finalizer list to collect them the next time because the object is very
// likely dead. This also won't break the GC invariance since these objects
// are not reachable from anywhere else.
int mark_reset_age = 0;

/*
 * The state transition looks like :
 *
 * ([(quick)sweep] means either a sweep or a quicksweep)
 *
 * <-[(quick)sweep]-
 *                 |
 *     ---->  GC_OLD  <--[(quick)sweep && age>promotion]--
 *     |     |                                           |
 *     |     |  GC_MARKED (in remset)                    |
 *     |     |     ^            |                        |
 *     |   [mark]  |          [mark]                     |
 *     |     |     |            |                        |
 *     |     |     |            |                        |
 *  [sweep]  | [write barrier]  |                        |
 *     |     v     |            v                        |
 *     ----- GC_OLD_MARKED <----                         |
 *              |               ^                        |
 *              |               |                        |
 *              --[quicksweep]---                        |
 *                                                       |
 *  ========= above this line objects are old =========  |
 *                                                       |
 *  ----[new]------> GC_CLEAN ------[mark]-----------> GC_MARKED
 *                    |    ^                                   |
 *  <-[(quick)sweep]---    |                                   |
 *                         --[(quick)sweep && age<=promotion]---
 */

// A quick sweep is a sweep where `!sweep_full`
// It means we won't touch GC_OLD_MARKED objects (old gen).

// When a reachable object has survived more than PROMOTE_AGE+1 collections
// it is tagged with GC_OLD during sweep and will be promoted on next mark
// because at that point we can know easily if it references young objects.
// Marked old objects that reference young ones are kept in the remset.

// When a write barrier triggers, the offending marked object is both queued,
// so as not to trigger the barrier again, and put in the remset.


#define PROMOTE_AGE 1
// this cannot be increased as is without changing :
// - sweep_page which is specialized for 1bit age
// - the size of the age storage in jl_gc_pagemeta_t


static int64_t scanned_bytes; // young bytes scanned while marking
static int64_t perm_scanned_bytes; // old bytes scanned while marking
static int prev_sweep_full = 1;

#define inc_sat(v, s) v = (v) >= s ? s : (v) + 1

// Full collection heuristics
int64_t live_bytes = 0;
int64_t promoted_bytes = 0;
int64_t last_live_bytes = 0; // live_bytes at last collection
int64_t t_start = 0; // Time GC starts;
#ifdef __GLIBC__
// maxrss at last malloc_trim
static int64_t last_trim_maxrss = 0;
#endif

static void gc_sync_cache_nolock(jl_ptls_t ptls,
                                 jl_gc_mark_cache_t *gc_cache) JL_NOTSAFEPOINT
{
    const int nbig = gc_cache->nbig_obj;
    for (int i = 0; i < nbig; i++) {
        void *ptr = gc_cache->big_obj[i];
        bigval_t *hdr = (bigval_t *)gc_ptr_clear_tag(ptr, 1);
        gc_big_object_unlink(hdr);
        if (gc_ptr_tag(ptr, 1)) {
            gc_big_object_link(hdr, &ptls->heap.big_objects);
        }
        else {
            // Move hdr from `big_objects` list to `big_objects_marked list`
            gc_big_object_link(hdr, &big_objects_marked);
        }
    }
    gc_cache->nbig_obj = 0;
    perm_scanned_bytes += gc_cache->perm_scanned_bytes;
    scanned_bytes += gc_cache->scanned_bytes;
    gc_cache->perm_scanned_bytes = 0;
    gc_cache->scanned_bytes = 0;
}

void gc_sync_cache(jl_ptls_t ptls) JL_NOTSAFEPOINT
{
    uv_mutex_lock(&gc_cache_lock);
    gc_sync_cache_nolock(ptls, &ptls->gc_cache);
    uv_mutex_unlock(&gc_cache_lock);
}

// No other threads can be running marking at the same time
static void gc_sync_all_caches_nolock(jl_ptls_t ptls)
{
    for (int t_i = 0; t_i < jl_n_threads; t_i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[t_i];
        gc_sync_cache_nolock(ptls, &ptls2->gc_cache);
    }
}

// weak references

JL_DLLEXPORT jl_weakref_t *jl_gc_new_weakref_th(jl_ptls_t ptls, jl_value_t *value)
{
    jl_weakref_t *wr = (jl_weakref_t *)jl_gc_alloc(ptls, sizeof(void *), jl_weakref_type);
    wr->value = value; // NOTE: wb not needed here
    arraylist_push(&ptls->heap.weak_refs, wr);
    return wr;
}

static void clear_weak_refs(void)
{
    for (int i = 0; i < jl_n_threads; i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[i];
        size_t n, l = ptls2->heap.weak_refs.len;
        void **lst = ptls2->heap.weak_refs.items;
        for (n = 0; n < l; n++) {
            jl_weakref_t *wr = (jl_weakref_t *)lst[n];
            if (!gc_marked(jl_astaggedvalue(wr->value)->bits.gc))
                wr->value = (jl_value_t *)jl_nothing;
        }
    }
}

// tracking Arrays with malloc'd storage

void jl_gc_track_malloced_array(jl_ptls_t ptls, jl_array_t *a) JL_NOTSAFEPOINT
{
    // This is **NOT** a GC safe point.
    mallocarray_t *ma;
    if (ptls->heap.mafreelist == NULL) {
        ma = (mallocarray_t *)malloc_s(sizeof(mallocarray_t));
    }
    else {
        ma = ptls->heap.mafreelist;
        ptls->heap.mafreelist = ma->next;
    }
    ma->a = a;
    ma->next = ptls->heap.mallocarrays;
    ptls->heap.mallocarrays = ma;
}

size_t jl_array_nbytes(jl_array_t *a) JL_NOTSAFEPOINT
{
    size_t sz = 0;
    int isbitsunion = jl_array_isbitsunion(a);
    if (jl_array_ndims(a) == 1)
        sz = a->elsize * a->maxsize + ((a->elsize == 1 && !isbitsunion) ? 1 : 0);
    else
        sz = a->elsize * jl_array_len(a);
    if (isbitsunion)
        // account for isbits Union array selector bytes
        sz += jl_array_len(a);
    return sz;
}

// pool allocation
jl_taggedvalue_t *reset_page(const jl_gc_pool_t *p, jl_gc_pagemeta_t *pg,
                             jl_taggedvalue_t *fl) JL_NOTSAFEPOINT
{
    assert(GC_PAGE_OFFSET >= sizeof(void *));
    pg->nfree = (GC_PAGE_SZ - GC_PAGE_OFFSET) / p->osize;
    jl_ptls_t ptls2 = jl_all_tls_states[pg->thread_n];
    pg->pool_n = p - ptls2->heap.norm_pools;
    memset(pg->ages, 0, GC_PAGE_SZ / 8 / p->osize + 1);
    jl_taggedvalue_t *beg = (jl_taggedvalue_t *)(pg->data + GC_PAGE_OFFSET);
    jl_taggedvalue_t *next = (jl_taggedvalue_t *)pg->data;
    if (fl == NULL) {
        next->next = NULL;
    }
    else {
        // Insert free page after first page.
        // This prevents unnecessary fragmentation from multiple pages
        // being allocated from at the same time. Instead, objects will
        // only ever be allocated from the first object in the list.
        // This is specifically being relied on by the implementation
        // of jl_gc_internal_obj_base_ptr() so that the function does
        // not have to traverse the entire list.
        jl_taggedvalue_t *flpage = (jl_taggedvalue_t *)gc_page_data(fl);
        next->next = flpage->next;
        flpage->next = beg;
        beg = fl;
    }
    pg->has_young = 0;
    pg->has_marked = 0;
    pg->fl_begin_offset = -1;
    pg->fl_end_offset = -1;
    return beg;
}

// Add a new page to the pool. Discards any pages in `p->newpages` before.
static NOINLINE jl_taggedvalue_t *add_page(jl_gc_pool_t *p) JL_NOTSAFEPOINT
{
    // Do not pass in `ptls` as argument. This slows down the fast path
    // in pool_alloc significantly
    jl_ptls_t ptls = jl_current_task->ptls;
    jl_gc_pagemeta_t *pg = jl_gc_alloc_page();
    pg->osize = p->osize;
    pg->ages = (uint8_t *)malloc_s(GC_PAGE_SZ / 8 / p->osize + 1);
    pg->thread_n = ptls->tid;
    jl_taggedvalue_t *fl = reset_page(p, pg, NULL);
    p->newpages = fl;
    return fl;
}

// Size includes the tag and the tag is not cleared!!
static inline jl_value_t *jl_gc_pool_alloc_inner(jl_ptls_t ptls, int pool_offset, int osize)
{
    // Use the pool offset instead of the pool address as the argument
    // to workaround a llvm bug.
    // Ref https://llvm.org/bugs/show_bug.cgi?id=27190
    jl_gc_pool_t *p = (jl_gc_pool_t *)((char *)ptls + pool_offset);
    assert(jl_atomic_load_relaxed(&ptls->gc_state) == 0);
#ifdef MEMDEBUG
    return jl_gc_big_alloc(ptls, osize);
#endif
    maybe_collect(ptls);
    jl_atomic_store_relaxed(&ptls->gc_num.allocd,
                            jl_atomic_load_relaxed(&ptls->gc_num.allocd) + osize);
    jl_atomic_store_relaxed(&ptls->gc_num.poolalloc,
                            jl_atomic_load_relaxed(&ptls->gc_num.poolalloc) + 1);
    // first try to use the freelist
    jl_taggedvalue_t *v = p->freelist;
    if (v) {
        jl_taggedvalue_t *next = v->next;
        p->freelist = next;
        if (__unlikely(gc_page_data(v) != gc_page_data(next))) {
            // we only update pg's fields when the freelist changes page
            // since pg's metadata is likely not in cache
            jl_gc_pagemeta_t *pg = jl_assume(page_metadata(v));
            assert(pg->osize == p->osize);
            pg->nfree = 0;
            pg->has_young = 1;
        }
        return jl_valueof(v);
    }
    // if the freelist is empty we reuse empty but not freed pages
    v = p->newpages;
    jl_taggedvalue_t *next = (jl_taggedvalue_t *)((char *)v + osize);
    // If there's no pages left or the current page is used up,
    // we need to use the slow path.
    char *cur_page = gc_page_data((char *)v - 1);
    if (__unlikely(!v || cur_page + GC_PAGE_SZ < (char *)next)) {
        if (v) {
            // like the freelist case,
            // but only update the page metadata when it is full
            jl_gc_pagemeta_t *pg = jl_assume(page_metadata((char *)v - 1));
            assert(pg->osize == p->osize);
            pg->nfree = 0;
            pg->has_young = 1;
            v = *(jl_taggedvalue_t **)cur_page;
        }
        // Not an else!!
        if (!v)
            v = add_page(p);
        next = (jl_taggedvalue_t *)((char *)v + osize);
    }
    p->newpages = next;
    return jl_valueof(v);
}

// Instrumented version of jl_gc_pool_alloc_inner, called into by LLVM-generated code.
JL_DLLEXPORT jl_value_t *jl_gc_pool_alloc(jl_ptls_t ptls, int pool_offset, int osize)
{
    jl_value_t *val = jl_gc_pool_alloc_inner(ptls, pool_offset, osize);
    maybe_record_alloc_to_profile(val, osize, jl_gc_unknown_type_tag);
    return val;
}

// This wrapper exists only to prevent `jl_gc_pool_alloc_inner` from being inlined into
// its callers. We provide an external-facing interface for callers, and inline
// `jl_gc_pool_alloc_inner` into this. (See https://github.com/JuliaLang/julia/pull/43868
// for more details.)
jl_value_t *jl_gc_pool_alloc_noinline(jl_ptls_t ptls, int pool_offset, int osize)
{
    return jl_gc_pool_alloc_inner(ptls, pool_offset, osize);
}

int jl_gc_classify_pools(size_t sz, int *osize)
{
    if (sz > GC_MAX_SZCLASS)
        return -1;
    size_t allocsz = sz + sizeof(jl_taggedvalue_t);
    int klass = jl_gc_szclass(allocsz);
    *osize = jl_gc_sizeclasses[klass];
    return (int)(intptr_t)(&((jl_ptls_t)0)->heap.norm_pools[klass]);
}

// sweep phase

int64_t lazy_freed_pages = 0;

// Returns pointer to terminal pointer of list rooted at *pfl.
static jl_taggedvalue_t **sweep_page(jl_gc_pool_t *p, jl_gc_pagemeta_t *pg,
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
static inline void sweep_pool_page(jl_taggedvalue_t ***pfl, jl_gc_pagemeta_t *pg,
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
static inline int sweep_pool_pagetable0(jl_taggedvalue_t ***pfl, pagetable0_t *pagetable0,
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
static inline int sweep_pool_pagetable1(jl_taggedvalue_t ***pfl, pagetable1_t *pagetable1,
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
static void sweep_pool_pagetable(jl_taggedvalue_t ***pfl, int sweep_full) JL_NOTSAFEPOINT
{
    if (REGION2_PG_COUNT == 1) { // compile-time optimization
        pagetable1_t *pagetable1 = memory_map.meta1[0];
        if (pagetable1)
            sweep_pool_pagetable1(pfl, pagetable1, sweep_full);
        return;
    }
    unsigned ub = 0;
    for (unsigned pg_i = 0; pg_i <= memory_map.ub; pg_i++) {
        uint32_t line = memory_map.allocmap1[pg_i];
        unsigned j;
        for (j = 0; line; j++, line >>= 1) {
            unsigned next = ffs_u32(line);
            j += next;
            line >>= next;
            pagetable1_t *pagetable1 = memory_map.meta1[pg_i * 32 + j];
            if (pagetable1 && !sweep_pool_pagetable1(pfl, pagetable1, sweep_full))
                memory_map.allocmap1[pg_i] &=
                    ~(1 << j); // no allocations found, remember that for next time
        }
        if (memory_map.allocmap1[pg_i]) {
            ub = pg_i;
        }
    }
    memory_map.ub = ub;
}

// sweep over all memory that is being used and not in a pool
static void gc_sweep_other(jl_ptls_t ptls, int sweep_full) JL_NOTSAFEPOINT
{
    gc_sweep_malloced_arrays();
    gc_sweep_big(ptls, sweep_full);
}

static void gc_pool_sync_nfree(jl_gc_pagemeta_t *pg, jl_taggedvalue_t *last) JL_NOTSAFEPOINT
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

// setup the data-structures for a sweep over all memory pools
static void gc_sweep_pool(int sweep_full)
{
    gc_time_pool_start();
    lazy_freed_pages = 0;

    // For the benfit of the analyzer, which doesn't know that jl_n_threads
    // doesn't change over the course of this function
    size_t n_threads = jl_n_threads;

    // allocate enough space to hold the end of the free list chain
    // for every thread and pool size
    jl_taggedvalue_t ***pfl = (jl_taggedvalue_t ***)alloca(n_threads * JL_GC_N_POOLS *
                                                           sizeof(jl_taggedvalue_t **));

    // update metadata of pages that were pointed to by freelist or newpages from a pool
    // i.e. pages being the current allocation target
    for (int t_i = 0; t_i < n_threads; t_i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[t_i];
        for (int i = 0; i < JL_GC_N_POOLS; i++) {
            jl_gc_pool_t *p = &ptls2->heap.norm_pools[i];
            jl_taggedvalue_t *last = p->freelist;
            if (last) {
                jl_gc_pagemeta_t *pg = jl_assume(page_metadata(last));
                gc_pool_sync_nfree(pg, last);
                pg->has_young = 1;
            }
            p->freelist = NULL;
            pfl[t_i * JL_GC_N_POOLS + i] = &p->freelist;

            last = p->newpages;
            if (last) {
                char *last_p = (char *)last;
                jl_gc_pagemeta_t *pg = jl_assume(page_metadata(last_p - 1));
                assert(last_p - gc_page_data(last_p - 1) >= GC_PAGE_OFFSET);
                pg->nfree = (GC_PAGE_SZ - (last_p - gc_page_data(last_p - 1))) / p->osize;
                pg->has_young = 1;
            }
            p->newpages = NULL;
        }
    }

    // the actual sweeping
    sweep_pool_pagetable(pfl, sweep_full);

    // null out terminal pointers of free lists
    for (int t_i = 0; t_i < n_threads; t_i++) {
        for (int i = 0; i < JL_GC_N_POOLS; i++) {
            *pfl[t_i * JL_GC_N_POOLS + i] = NULL;
        }
    }

    gc_time_pool_end(sweep_full);
}

static void gc_sweep_perm_alloc(void)
{
    uint64_t t0 = jl_hrtime();
    gc_sweep_sysimg();
    gc_time_sysimg_end(t0);
}


#ifdef JL_DEBUG_BUILD
static void *volatile gc_findval; // for usage from gdb, for finding the gc-root for a value
#endif

void *sysimg_base;
void *sysimg_end;
void jl_gc_set_permalloc_region(void *start, void *end)
{
    sysimg_base = start;
    sysimg_end = end;
}

// find unmarked objects that need to be finalized from the finalizer list "list".
// this must happen last in the mark phase.
static void sweep_finalizer_list(arraylist_t *list)
{
    void **items = list->items;
    size_t len = list->len;
    size_t j = 0;
    for (size_t i = 0; i < len; i += 2) {
        void *v0 = items[i];
        void *v = gc_ptr_clear_tag(v0, 1);
        if (__unlikely(!v0)) {
            // remove from this list
            continue;
        }

        void *fin = items[i + 1];
        int isfreed = !gc_marked(jl_astaggedvalue(v)->bits.gc);
        int isold = (list != &finalizer_list_marked &&
                     jl_astaggedvalue(v)->bits.gc == GC_OLD_MARKED &&
                     jl_astaggedvalue(fin)->bits.gc == GC_OLD_MARKED);
        if (isfreed || isold) {
            // remove from this list
        }
        else {
            if (j < i) {
                items[j] = items[i];
                items[j + 1] = items[i + 1];
            }
            j += 2;
        }
        if (isfreed) {
            schedule_finalization(v0, fin);
        }
        if (isold) {
            // The caller relies on the new objects to be pushed to the end of
            // the list!!
            arraylist_push(&finalizer_list_marked, v0);
            arraylist_push(&finalizer_list_marked, fin);
        }
    }
    list->len = j;
}

// collector entry point and control
static _Atomic(uint32_t) jl_gc_disable_counter = 1;

JL_DLLEXPORT int jl_gc_enable(int on)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    int prev = !ptls->disable_gc;
    ptls->disable_gc = (on == 0);
    if (on && !prev) {
        // disable -> enable
        if (jl_atomic_fetch_add(&jl_gc_disable_counter, -1) == 1) {
            gc_num.allocd += gc_num.deferred_alloc;
            gc_num.deferred_alloc = 0;
        }
    }
    else if (prev && !on) {
        // enable -> disable
        jl_atomic_fetch_add(&jl_gc_disable_counter, 1);
        // check if the GC is running and wait for it to finish
        jl_gc_safepoint_(ptls);
    }
    return prev;
}

JL_DLLEXPORT int jl_gc_is_enabled(void)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return !ptls->disable_gc;
}

size_t jl_maxrss(void);

// Only one thread should be running in this function static int
int _jl_gc_collect(jl_ptls_t ptls, jl_gc_collection_t collection)
{
    combine_thread_gc_counts(&gc_num);

    uint64_t gc_start_time = jl_hrtime();
    int64_t last_perm_scanned_bytes = perm_scanned_bytes;

    // Main mark-loop
    {
        uint64_t start_mark_time = jl_hrtime();

        JL_PROBE_GC_MARK_BEGIN();
        {
            jl_gc_markqueue_t *mq = &ptls->mark_queue;
            // Fix GC bits of objects in the remset.
            for (int t_i = 0; t_i < jl_n_threads; t_i++) {
                gc_premark(jl_all_tls_states[t_i]);
            }
            for (int t_i = 0; t_i < jl_n_threads; t_i++) {
                jl_ptls_t ptls2 = jl_all_tls_states[t_i];
                // Mark every thread local root
                gc_queue_thread_local(mq, ptls2);
                // Mark any managed objects in the backtrace buffer
                gc_queue_bt_buf(mq, ptls2);
                // Mark every object in the `last_remsets` and `rem_binding`
                gc_queue_remset(ptls, ptls2);
            }
            // Walk roots
            gc_mark_roots(mq);
            if (gc_cblist_root_scanner) {
                gc_invoke_callbacks(jl_gc_cb_root_scanner_t, gc_cblist_root_scanner,
                                    (collection));
            }
            gc_mark_loop(ptls);
        }
        JL_PROBE_GC_MARK_END(scanned_bytes, perm_scanned_bytes);

        gc_settime_premark_end();
        gc_time_mark_pause(gc_start_time, scanned_bytes, perm_scanned_bytes);

        uint64_t end_mark_time = jl_hrtime();
        uint64_t mark_time = end_mark_time - start_mark_time;
        gc_num.since_sweep += gc_num.allocd;
        gc_num.mark_time = mark_time;
        gc_num.total_mark_time += mark_time;
    }

    int64_t actual_allocd = gc_num.since_sweep;
    // 4. check for objects to finalize
    clear_weak_refs();
    // Record the length of the marked list since we need to
    // mark the object moved to the marked list from the
    // `finalizer_list` by `sweep_finalizer_list`
    size_t orig_marked_len = finalizer_list_marked.len;
    for (int i = 0; i < jl_n_threads; i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[i];
        sweep_finalizer_list(&ptls2->finalizers);
    }
    if (prev_sweep_full) {
        sweep_finalizer_list(&finalizer_list_marked);
        orig_marked_len = 0;
    }
    for (int i = 0; i < jl_n_threads; i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[i];
        gc_mark_finlist(ptls, &ptls2->finalizers, 0);
    }
    gc_mark_finlist(ptls, &finalizer_list_marked, orig_marked_len);
    // "Flush" the mark stack before flipping the reset_age bit
    // so that the objects are not incorrectly reset.
    gc_mark_loop(ptls);
    // Conservative marking relies on age to tell allocated objects
    // and freelist entries apart.
    mark_reset_age = !jl_gc_conservative_gc_support_enabled();
    // Reset the age and old bit for any unmarked objects referenced by the
    // `to_finalize` list. These objects are only reachable from this list
    // and should not be referenced by any old objects so this won't break
    // the GC invariant.
    gc_mark_finlist(ptls, &to_finalize, 0);
    gc_mark_loop(ptls);
    mark_reset_age = 0;
    gc_settime_postmark_end();

    // Flush everything in mark cache
    gc_sync_all_caches_nolock(ptls);

    int64_t live_sz_ub = live_bytes + actual_allocd;
    int64_t live_sz_est = scanned_bytes + perm_scanned_bytes;
    int64_t estimate_freed = live_sz_ub - live_sz_est;

    gc_verify(ptls);

    gc_stats_all_pool();
    gc_stats_big_obj();
    objprofile_printall();
    objprofile_reset();
    gc_num.total_allocd += gc_num.since_sweep;
    if (!prev_sweep_full)
        promoted_bytes += perm_scanned_bytes - last_perm_scanned_bytes;
    // 5. next collection decision
    int not_freed_enough =
        (collection == JL_GC_AUTO) && estimate_freed < (7 * (actual_allocd / 10));
    int nptr = 0;
    for (int i = 0; i < jl_n_threads; i++)
        nptr += jl_all_tls_states[i]->heap.remset_nptr;

    // many pointers in the intergen frontier => "quick" mark is not quick
    int large_frontier = nptr * sizeof(void *) >= default_collect_interval;
    int sweep_full = 0;
    int recollect = 0;

    // update heuristics only if this GC was automatically triggered
    if (collection == JL_GC_AUTO) {
        if (not_freed_enough) {
            gc_num.interval = gc_num.interval * 2;
        }
        if (large_frontier) {
            sweep_full = 1;
        }
        if (gc_num.interval > max_collect_interval) {
            sweep_full = 1;
            gc_num.interval = max_collect_interval;
        }
    }


    // If the live data outgrows the suggested max_total_memory
    // we keep going with minimum intervals and full gcs until
    // we either free some space or get an OOM error.
    if (live_bytes > max_total_memory) {
        sweep_full = 1;
    }
    if (gc_sweep_always_full) {
        sweep_full = 1;
    }
    if (collection == JL_GC_FULL) {
        sweep_full = 1;
        recollect = 1;
    }
    if (sweep_full) {
        // these are the difference between the number of gc-perm bytes scanned
        // on the first collection after sweep_full, and the current scan
        perm_scanned_bytes = 0;
        promoted_bytes = 0;
    }

    scanned_bytes = 0;
    uint64_t pause;

    // Sweeping
    {
        uint64_t start_sweep_time = jl_hrtime();

        JL_PROBE_GC_SWEEP_BEGIN(sweep_full);
        {
            gc_sweep_weak_refs();
            gc_sweep_stack_pools();
            gc_sweep_foreign_objs();
            gc_sweep_other(ptls, sweep_full);
            gc_scrub();
            gc_verify_tags();
            gc_sweep_pool(sweep_full);
            if (sweep_full)
                gc_sweep_perm_alloc();
        }
        JL_PROBE_GC_SWEEP_END();

        uint64_t gc_end_time = jl_hrtime();
        uint64_t sweep_time = gc_end_time - start_sweep_time;
        pause = gc_end_time - gc_start_time;
        gc_num.total_sweep_time += sweep_time;
        gc_num.sweep_time = sweep_time;
    }

    // sweeping is over
    // 7. if it is a quick sweep, put back the remembered objects in queued state
    // so that we don't trigger the barrier again on them.
    for (int t_i = 0; t_i < jl_n_threads; t_i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[t_i];
        if (!sweep_full) {
            for (int i = 0; i < ptls2->heap.remset->len; i++) {
                jl_astaggedvalue(ptls2->heap.remset->items[i])->bits.gc = GC_MARKED;
            }
            for (int i = 0; i < ptls2->heap.rem_bindings.len; i++) {
                void *ptr = ptls2->heap.rem_bindings.items[i];
                jl_astaggedvalue(ptr)->bits.gc = GC_MARKED;
            }
        }
        else {
            ptls2->heap.remset->len = 0;
            ptls2->heap.rem_bindings.len = 0;
        }
    }

#ifdef __GLIBC__
    if (sweep_full) {
        // issue #30653
        // empirically, the malloc runaway seemed to occur within a growth gap
        // of about 20-25%
        if (jl_maxrss() > (last_trim_maxrss / 4) * 5) {
            malloc_trim(0);
            last_trim_maxrss = jl_maxrss();
        }
    }
#endif


    _report_gc_finished(pause, gc_num.freed, sweep_full, recollect);

    gc_final_pause_end(gc_start_time, gc_end_time);
    gc_time_sweep_pause(gc_end_time, actual_allocd, live_bytes, estimate_freed, sweep_full);
    gc_num.full_sweep += sweep_full;
    uint64_t max_memory = last_live_bytes + gc_num.allocd;
    if (max_memory > gc_num.max_memory) {
        gc_num.max_memory = max_memory;
    }

    gc_num.allocd = 0;
    last_live_bytes = live_bytes;
    live_bytes += gc_num.since_sweep - gc_num.freed;

    if (collection == JL_GC_AUTO) {
        // If the current interval is larger than half the live data decrease the interval
        int64_t half = live_bytes / 2;
        if (gc_num.interval > half)
            gc_num.interval = half;
        // But never go below default
        if (gc_num.interval < default_collect_interval)
            gc_num.interval = default_collect_interval;
    }

    if (gc_num.interval + live_bytes > max_total_memory) {
        if (live_bytes < max_total_memory) {
            gc_num.interval = max_total_memory - live_bytes;
        }
        else {
            // We can't stay under our goal so let's go back to
            // the minimum interval and hope things get better
            gc_num.interval = default_collect_interval;
        }
    }

    gc_time_summary(sweep_full, t_start, gc_end_time, gc_num.freed, live_bytes,
                    gc_num.interval, pause, gc_num.time_to_safepoint, gc_num.mark_time,
                    gc_num.sweep_time);

    prev_sweep_full = sweep_full;
    gc_num.pause += !recollect;
    gc_num.total_time += pause;
    gc_num.since_sweep = 0;
    gc_num.freed = 0;
    if (pause > gc_num.max_pause) {
        gc_num.max_pause = pause;
    }
    reset_thread_gc_counts();

    return recollect;
}

JL_DLLEXPORT void jl_gc_collect(jl_gc_collection_t collection)
{
    JL_PROBE_GC_BEGIN(collection);

    jl_task_t *ct = jl_current_task;
    jl_ptls_t ptls = ct->ptls;
    if (jl_atomic_load_relaxed(&jl_gc_disable_counter)) {
        size_t localbytes = jl_atomic_load_relaxed(&ptls->gc_num.allocd) + gc_num.interval;
        jl_atomic_store_relaxed(&ptls->gc_num.allocd, -(int64_t)gc_num.interval);
        static_assert(sizeof(_Atomic(uint64_t)) == sizeof(gc_num.deferred_alloc), "");
        jl_atomic_fetch_add((_Atomic(uint64_t) *)&gc_num.deferred_alloc, localbytes);
        return;
    }
    jl_gc_debug_print();

    int8_t old_state = jl_atomic_load_relaxed(&ptls->gc_state);
    jl_atomic_store_release(&ptls->gc_state, JL_GC_STATE_WAITING);
    // `jl_safepoint_start_gc()` makes sure only one thread can
    // run the GC.
    uint64_t t0 = jl_hrtime();
    if (!jl_safepoint_start_gc()) {
        // Multithread only. See assertion in `safepoint.c`
        jl_gc_state_set(ptls, old_state, JL_GC_STATE_WAITING);
        return;
    }
    JL_TIMING(GC);
    int last_errno = errno;
#ifdef _OS_WINDOWS_
    DWORD last_error = GetLastError();
#endif
    // Now we are ready to wait for other threads to hit the safepoint,
    // we can do a few things that doesn't require synchronization.
    // TODO (concurrently queue objects)
    // no-op for non-threading
    jl_gc_wait_for_the_world();
    JL_PROBE_GC_STOP_THE_WORLD();

    uint64_t t1 = jl_hrtime();
    uint64_t duration = t1 - t0;
    if (duration > gc_num.max_time_to_safepoint)
        gc_num.max_time_to_safepoint = duration;
    gc_num.time_to_safepoint = duration;

    gc_invoke_callbacks(jl_gc_cb_pre_gc_t, gc_cblist_pre_gc, (collection));

    if (!jl_atomic_load_relaxed(&jl_gc_disable_counter)) {
        JL_LOCK_NOGC(&finalizers_lock);
        if (_jl_gc_collect(ptls, collection)) {
            // recollect
            int ret = _jl_gc_collect(ptls, JL_GC_AUTO);
            (void)ret;
            assert(!ret);
        }
        JL_UNLOCK_NOGC(&finalizers_lock);
    }

    // no-op for non-threading
    jl_safepoint_end_gc();
    jl_gc_state_set(ptls, old_state, JL_GC_STATE_WAITING);
    JL_PROBE_GC_END();

    // Only disable finalizers on current thread
    // Doing this on all threads is racy (it's impossible to check
    // or wait for finalizers on other threads without dead lock).
    if (!ptls->finalizers_inhibited && ptls->locks.len == 0) {
        int8_t was_in_finalizer = ptls->in_finalizer;
        ptls->in_finalizer = 1;
        run_finalizers(ct);
        ptls->in_finalizer = was_in_finalizer;
    }
    JL_PROBE_GC_FINALIZER();

    gc_invoke_callbacks(jl_gc_cb_post_gc_t, gc_cblist_post_gc, (collection));
#ifdef _OS_WINDOWS_
    SetLastError(last_error);
#endif
    errno = last_errno;
}

// Per-thread initialization
void jl_init_thread_heap(jl_ptls_t ptls)
{
    if (ptls->tid == 0)
        ptls->disable_gc = 1;
    jl_thread_heap_t *heap = &ptls->heap;
    jl_gc_pool_t *p = heap->norm_pools;
    for (int i = 0; i < JL_GC_N_POOLS; i++) {
        p[i].osize = jl_gc_sizeclasses[i];
        p[i].freelist = NULL;
        p[i].newpages = NULL;
    }
    arraylist_new(&heap->weak_refs, 0);
    arraylist_new(&heap->live_tasks, 0);
    heap->mallocarrays = NULL;
    heap->mafreelist = NULL;
    heap->big_objects = NULL;
    arraylist_new(&heap->rem_bindings, 0);
    heap->remset = &heap->_remset[0];
    heap->last_remset = &heap->_remset[1];
    arraylist_new(heap->remset, 0);
    arraylist_new(heap->last_remset, 0);
    arraylist_new(&ptls->finalizers, 0);
    arraylist_new(&ptls->sweep_objs, 0);

    jl_gc_mark_cache_t *gc_cache = &ptls->gc_cache;
    gc_cache->perm_scanned_bytes = 0;
    gc_cache->scanned_bytes = 0;
    gc_cache->nbig_obj = 0;
    size_t init_size = 1024;

    jl_gc_markqueue_t *mq = &ptls->mark_queue;

    mq->current = mq->start = (jl_value_t **)malloc_s(init_size * sizeof(jl_value_t *));
    mq->end = mq->start + init_size;

    memset(&ptls->gc_num, 0, sizeof(ptls->gc_num));
    assert(gc_num.interval == default_collect_interval);
    jl_atomic_store_relaxed(&ptls->gc_num.allocd, -(int64_t)gc_num.interval);
}

// System-wide initializations
void jl_gc_init(void)
{
    JL_MUTEX_INIT(&finalizers_lock);
    uv_mutex_init(&gc_cache_lock);
    uv_mutex_init(&gc_perm_lock);

    jl_gc_init_page();
    jl_gc_debug_init();

    arraylist_new(&finalizer_list_marked, 0);
    arraylist_new(&to_finalize, 0);

    gc_num.interval = default_collect_interval;
    last_long_collect_interval = default_collect_interval;
    gc_num.allocd = 0;
    gc_num.max_pause = 0;
    gc_num.max_memory = 0;

#ifdef _P64
    // on a big memory machine, set max_collect_interval to totalmem / ncores / 2
    uint64_t total_mem = uv_get_total_memory();
    uint64_t constrained_mem = uv_get_constrained_memory();
    if (constrained_mem > 0 && constrained_mem < total_mem)
        total_mem = constrained_mem;
    size_t maxmem = total_mem / jl_cpu_threads() / 2;
    if (maxmem > max_collect_interval)
        max_collect_interval = maxmem;
#endif

    // We allocate with abandon until we get close to the free memory on the machine.
    uint64_t free_mem = uv_get_free_memory();
    uint64_t high_water_mark = free_mem / 10 * 7; // 70% high water mark

    if (high_water_mark < max_total_memory)
        max_total_memory = high_water_mark;

    t_start = jl_hrtime();
}

void jl_gc_set_max_memory(uint64_t max_mem)
{
    if (max_mem > 0 && max_mem < (uint64_t)1 << (sizeof(memsize_t) * 8 - 1)) {
        max_total_memory = max_mem;
    }
}

// callback for passing OOM errors from gmp
JL_DLLEXPORT void jl_throw_out_of_memory_error(void)
{
    jl_throw(jl_memory_exception);
}

JL_DLLEXPORT jl_weakref_t *jl_gc_new_weakref(jl_value_t *value)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return jl_gc_new_weakref_th(ptls, value);
}

JL_DLLEXPORT int jl_gc_enable_conservative_gc_support(void)
{
    static_assert(jl_buff_tag % GC_PAGE_SZ == 0,
                  "jl_buff_tag must be a multiple of GC_PAGE_SZ");
    if (jl_is_initialized()) {
        int result = jl_atomic_fetch_or(&support_conservative_marking, 1);
        if (!result) {
            // Do a full collection to ensure that age bits are updated
            // properly. We don't have to worry about race conditions
            // for this part, as allocation itself is unproblematic and
            // a collection will wait for safepoints.
            jl_gc_collect(JL_GC_FULL);
        }
        return result;
    }
    else {
        int result = jl_atomic_load(&support_conservative_marking);
        jl_atomic_store(&support_conservative_marking, 1);
        return result;
    }
}

JL_DLLEXPORT int jl_gc_conservative_gc_support_enabled(void)
{
    return jl_atomic_load(&support_conservative_marking);
}

JL_DLLEXPORT jl_value_t *jl_gc_internal_obj_base_ptr(void *p)
{
    p = (char *)p - 1;
    jl_gc_pagemeta_t *meta = page_metadata(p);
    if (meta && meta->ages) {
        char *page = gc_page_data(p);
        // offset within page.
        size_t off = (char *)p - page;
        if (off < GC_PAGE_OFFSET)
            return NULL;
        // offset within object
        size_t off2 = (off - GC_PAGE_OFFSET);
        size_t osize = meta->osize;
        off2 %= osize;
        if (off - off2 + osize > GC_PAGE_SZ)
            return NULL;
        jl_taggedvalue_t *cell = (jl_taggedvalue_t *)((char *)p - off2);
        // We have to distinguish between three cases:
        // 1. We are on a page where every cell is allocated.
        // 2. We are on a page where objects are currently bump-allocated
        //    from the corresponding pool->newpages list.
        // 3. We are on a page with a freelist that is used for object
        //    allocation.
        if (meta->nfree == 0) {
            // case 1: full page; `cell` must be an object
            goto valid_object;
        }
        jl_gc_pool_t *pool =
            jl_all_tls_states[meta->thread_n]->heap.norm_pools + meta->pool_n;
        if (meta->fl_begin_offset == (uint16_t)-1) {
            // case 2: this is a page on the newpages list
            jl_taggedvalue_t *newpages = pool->newpages;
            // Check if the page is being allocated from via newpages
            if (!newpages)
                return NULL;
            char *data = gc_page_data(newpages);
            if (data != meta->data) {
                // Pages on newpages form a linked list where only the
                // first one is allocated from (see reset_page()).
                // All other pages are empty.
                return NULL;
            }
            // This is the first page on the newpages list, where objects
            // are allocated from.
            if ((char *)cell >= (char *)newpages) // past allocation pointer
                return NULL;
            goto valid_object;
        }
        // case 3: this is a page with a freelist
        // marked or old objects can't be on the freelist
        if (cell->bits.gc)
            goto valid_object;
        // When allocating from a freelist, three subcases are possible:
        // * The freelist of a page has been exhausted; this was handled
        //   under case 1, as nfree == 0.
        // * The freelist of the page has not been used, and the age bits
        //   reflect whether a cell is on the freelist or an object.
        // * The freelist is currently being allocated from. In this case,
        //   pool->freelist will point to the current page; any cell with
        //   a lower address will be an allocated object, and for cells
        //   with the same or a higher address, the corresponding age
        //   bit will reflect whether it's on the freelist.
        // Age bits are set in sweep_page() and are 0 for freelist
        // entries and 1 for live objects. The above subcases arise
        // because allocating a cell will not update the age bit, so we
        // need extra logic for pages that have been allocated from.
        unsigned obj_id = (off - off2) / osize;
        // We now distinguish between the second and third subcase.
        // Freelist entries are consumed in ascending order. Anything
        // before the freelist pointer was either live during the last
        // sweep or has been allocated since.
        if (gc_page_data(cell) == gc_page_data(pool->freelist) &&
            (char *)cell < (char *)pool->freelist)
            goto valid_object;
        // We know now that the age bit reflects liveness status during
        // the last sweep and that the cell has not been reused since.
        if (!(meta->ages[obj_id / 8] & (1 << (obj_id % 8)))) {
            return NULL;
        }
        // Not a freelist entry, therefore a valid object.
valid_object:
        // We have to treat objects with type `jl_buff_tag` differently,
        // as they must not be passed to the usual marking functions.
        // Note that jl_buff_tag is a multiple of GC_PAGE_SZ, thus it
        // cannot be a type reference.
        if ((cell->header & ~(uintptr_t)3) == jl_buff_tag)
            return NULL;
        return jl_valueof(cell);
    }
    return NULL;
}

JL_DLLEXPORT size_t jl_gc_max_internal_obj_size(void)
{
    return GC_MAX_SZCLASS;
}

JL_DLLEXPORT size_t jl_gc_external_obj_hdr_size(void)
{
    return sizeof(bigval_t);
}

JL_DLLEXPORT void jl_gc_schedule_foreign_sweepfunc(jl_ptls_t ptls, jl_value_t *obj)
{
    arraylist_push(&ptls->sweep_objs, obj);
}

#ifdef __cplusplus
}
#endif
