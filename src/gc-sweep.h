// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef JL_GC_SWEEP_H
#define JL_GC_SWEEP_H

#include "gc.h"

#ifdef __cplusplus
extern "C" {
#endif

void gc_sweep_weak_refs(void);
bigval_t **gc_sweep_big_list(int sweep_full, bigval_t **pv) JL_NOTSAFEPOINT;
void gc_sweep_big(jl_ptls_t ptls, int sweep_full) JL_NOTSAFEPOINT;
void gc_free_array(jl_array_t *a) JL_NOTSAFEPOINT;
void gc_sweep_malloced_arrays(void) JL_NOTSAFEPOINT;

#ifdef __cplusplus
}
#endif

#endif