// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef JL_GC_SWEEP_H
#define JL_GC_SWEEP_H

#include "gc.h"

#ifdef __cplusplus
extern "C" {
#endif

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

void gc_sweep_weak_refs(void);
bigval_t **gc_sweep_big_list(int sweep_full, bigval_t **pv) JL_NOTSAFEPOINT;
void gc_sweep_big(jl_ptls_t ptls, int sweep_full) JL_NOTSAFEPOINT;
void gc_free_array(jl_array_t *a) JL_NOTSAFEPOINT;
void gc_sweep_malloced_arrays(void) JL_NOTSAFEPOINT;

#ifdef __cplusplus
}
#endif

#endif