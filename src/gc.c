// This file is a part of Julia. License is MIT: https://julialang.org/license

#include "gc.h"
#include "gc-markqueue.h"
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

extern int64_t last_gc_total_bytes;

// global variables for GC stats

int64_t scanned_bytes; // young bytes scanned while marking
int64_t perm_scanned_bytes; // old bytes scanned while marking

STATIC_INLINE void gc_sync_cache_nolock(jl_ptls_t ptls,
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

STATIC_INLINE void gc_sync_all_caches_nolock(jl_ptls_t ptls)
{
    for (int t_i = 0; t_i < jl_n_threads; t_i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[t_i];
        gc_sync_cache_nolock(ptls, &ptls2->gc_cache);
    }
}

void gc_sync_cache(jl_ptls_t ptls) JL_NOTSAFEPOINT
{
    uv_mutex_lock(&gc_cache_lock);
    gc_sync_cache_nolock(ptls, &ptls->gc_cache);
    uv_mutex_unlock(&gc_cache_lock);
}

// Weak references

JL_DLLEXPORT jl_weakref_t *jl_gc_new_weakref_th(jl_ptls_t ptls, jl_value_t *value)
{
    jl_weakref_t *wr = (jl_weakref_t *)jl_gc_alloc(ptls, sizeof(void *), jl_weakref_type);
    wr->value = value; // NOTE: wb not needed here
    arraylist_push(&ptls->heap.weak_refs, wr);
    return wr;
}

JL_DLLEXPORT jl_weakref_t *jl_gc_new_weakref(jl_value_t *value)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return jl_gc_new_weakref_th(ptls, value);
}

STATIC_INLINE void gc_clear_weak_refs(void)
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

// Tracking arrays with malloc'd storage

void jl_gc_track_malloced_array(jl_ptls_t ptls, jl_array_t *a) JL_NOTSAFEPOINT
{
    // This is **NOT** a GC safe point.
    mallocarray_t *ma;
    if (!ptls->heap.mafreelist) {
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
    gc_combine_thread_counts(&gc_num);

    uint64_t gc_start_time = jl_hrtime();
    int64_t last_perm_scanned_bytes = perm_scanned_bytes;

    // Main mark-loop
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

    int64_t actual_allocd = gc_num.since_sweep;

    // Check for objects to finalize
    gc_clear_weak_refs();
    // Record the length of the marked list since we need to
    // mark the object moved to the marked list from the
    // `finalizer_list` by `sweep_finalizer_list`
    size_t orig_marked_len = finalizer_list_marked.len;
    for (int i = 0; i < jl_n_threads; i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[i];
        gc_sweep_finalizer_list(&ptls2->finalizers);
    }
    if (prev_sweep_full) {
        gc_sweep_finalizer_list(&finalizer_list_marked);
        orig_marked_len = 0;
    }
    for (int i = 0; i < jl_n_threads; i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[i];
        gc_mark_finlist(ptls, &ptls2->finalizers, 0);
    }
    gc_mark_finlist(ptls, &finalizer_list_marked, orig_marked_len);
    gc_mark_finlist(ptls, &to_finalize, 0);
    gc_mark_loop(ptls);
    gc_settime_postmark_end();

    // Flush everything in mark cache
    gc_sync_all_caches_nolock(ptls);

    int64_t live_sz_ub = live_bytes + actual_allocd;
    int64_t live_sz_est = scanned_bytes + perm_scanned_bytes;
    int64_t estimate_freed = live_sz_ub - live_sz_est;

    // Verification and stats
    gc_verify(ptls);
    gc_stats_all_pool();
    gc_stats_big_obj();
    objprofile_printall();
    objprofile_reset();

    gc_num.total_allocd += gc_num.since_sweep;
    if (!prev_sweep_full)
        promoted_bytes += perm_scanned_bytes - last_perm_scanned_bytes;

    // Next collection decision
    int not_freed_enough =
        (collection == JL_GC_AUTO) && estimate_freed < (7 * (actual_allocd / 10));
    int nptr = 0;
    for (int i = 0; i < jl_n_threads; i++) {
        nptr += jl_all_tls_states[i]->heap.remset_nptr;
    }
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

    uint64_t pause;
    scanned_bytes = 0;

    // Sweeping
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

    // if it is a quick sweep, put back the remembered objects in queued state
    // so that we don't trigger the barrier again on them.
    for (int t_i = 0; t_i < jl_n_threads; t_i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[t_i];
        if (!sweep_full) {
            for (int i = 0; i < ptls2->heap.remset.len; i++) {
                jl_astaggedvalue(ptls2->heap.remset.items[i])->bits.gc = GC_MARKED;
            }
            for (int i = 0; i < ptls2->heap.rem_bindings.len; i++) {
                void *ptr = ptls2->heap.rem_bindings.items[i];
                jl_astaggedvalue(ptr)->bits.gc = GC_MARKED;
            }
        }
        else {
            ptls2->heap.remset.len = 0;
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

    gc_num.full_sweep += sweep_full;

    _report_gc_finished(pause, gc_num.freed, sweep_full, recollect);
    gc_final_pause_end(gc_start_time, gc_end_time);
    gc_time_sweep_pause(gc_end_time, actual_allocd, live_bytes, estimate_freed, sweep_full);

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
    gc_reset_thread_counts();

    return recollect;
}

JL_DLLEXPORT void jl_gc_collect(jl_gc_collection_t collection)
{
    int last_errno;
    jl_task_t *ct = jl_current_task;
    jl_ptls_t ptls = ct->ptls;

    // Collection
    JL_PROBE_GC_BEGIN(collection);
    {
        if (jl_atomic_load_relaxed(&jl_gc_disable_counter)) {
            size_t localbytes =
                jl_atomic_load_relaxed(&ptls->gc_num.allocd) + gc_num.interval;
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

        last_errno = errno;

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
    }
    JL_PROBE_GC_END();

    // Finalizers
    {
        // Only disable finalizers on current thread
        // Doing this on all threads is racy (it's impossible to check
        // or wait for finalizers on other threads without dead lock).
        if (!ptls->finalizers_inhibited && ptls->locks.len == 0) {
            int8_t was_in_finalizer = ptls->in_finalizer;
            ptls->in_finalizer = 1;
            run_finalizers(ct);
            ptls->in_finalizer = was_in_finalizer;
        }
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
    arraylist_new(&heap->remset, 0);
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
