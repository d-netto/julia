// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef JL_GC_STATS_H
#define JL_GC_STATS_H

// Full collection heuristics
extern int64_t live_bytes;
extern int64_t promoted_bytes;
extern int64_t last_live_bytes; // live_bytes at last collection
extern int64_t t_start; // Time GC starts;

void jl_gc_count_allocd(size_t sz) JL_NOTSAFEPOINT
{
    jl_ptls_t ptls = jl_current_task->ptls;
    jl_atomic_store_relaxed(&ptls->gc_num.allocd,
                            jl_atomic_load_relaxed(&ptls->gc_num.allocd) + sz);
}

void combine_thread_gc_counts(jl_gc_num_t *dest) JL_NOTSAFEPOINT
{
    for (int i = 0; i < jl_n_threads; i++) {
        jl_ptls_t ptls = jl_all_tls_states[i];
        if (ptls) {
            dest->allocd +=
                (jl_atomic_load_relaxed(&ptls->gc_num.allocd) + gc_num.interval);
            dest->freed += jl_atomic_load_relaxed(&ptls->gc_num.freed);
            dest->malloc += jl_atomic_load_relaxed(&ptls->gc_num.malloc);
            dest->realloc += jl_atomic_load_relaxed(&ptls->gc_num.realloc);
            dest->poolalloc += jl_atomic_load_relaxed(&ptls->gc_num.poolalloc);
            dest->bigalloc += jl_atomic_load_relaxed(&ptls->gc_num.bigalloc);
            dest->freecall += jl_atomic_load_relaxed(&ptls->gc_num.freecall);
        }
    }
}

void reset_thread_gc_counts(void) JL_NOTSAFEPOINT
{
    for (int i = 0; i < jl_n_threads; i++) {
        jl_ptls_t ptls = jl_all_tls_states[i];
        if (ptls) {
            memset(&ptls->gc_num, 0, sizeof(ptls->gc_num));
            jl_atomic_store_relaxed(&ptls->gc_num.allocd, -(int64_t)gc_num.interval);
        }
    }
}

void jl_gc_reset_alloc_count(void) JL_NOTSAFEPOINT
{
    combine_thread_gc_counts(&gc_num);
    live_bytes += (gc_num.deferred_alloc + gc_num.allocd);
    gc_num.allocd = 0;
    gc_num.deferred_alloc = 0;
    reset_thread_gc_counts();
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

JL_DLLEXPORT void jl_gc_get_total_bytes(int64_t *bytes) JL_NOTSAFEPOINT
{
    jl_gc_num_t num = gc_num;
    combine_thread_gc_counts(&num);
    // Sync this logic with `base/util.jl:GC_Diff`
    *bytes = (num.total_allocd + num.deferred_alloc + num.allocd);
}

JL_DLLEXPORT uint64_t jl_gc_total_hrtime(void)
{
    return gc_num.total_time;
}

JL_DLLEXPORT jl_gc_num_t jl_gc_num(void)
{
    jl_gc_num_t num = gc_num;
    combine_thread_gc_counts(&num);
    return num;
}

JL_DLLEXPORT void jl_gc_reset_stats(void)
{
    gc_num.max_pause = 0;
    gc_num.max_memory = 0;
    gc_num.max_time_to_safepoint = 0;
}

// TODO: these were supposed to be thread local
JL_DLLEXPORT int64_t jl_gc_diff_total_bytes(void) JL_NOTSAFEPOINT
{
    int64_t oldtb = last_gc_total_bytes;
    int64_t newtb;
    jl_gc_get_total_bytes(&newtb);
    last_gc_total_bytes = newtb;
    return newtb - oldtb;
}

JL_DLLEXPORT int64_t jl_gc_sync_total_bytes(int64_t offset) JL_NOTSAFEPOINT
{
    int64_t oldtb = last_gc_total_bytes;
    int64_t newtb;
    jl_gc_get_total_bytes(&newtb);
    last_gc_total_bytes = newtb - offset;
    return newtb - oldtb;
}

JL_DLLEXPORT int64_t jl_gc_live_bytes(void)
{
    return live_bytes;
}

#ifdef __cplusplus
}
#endif

#endif