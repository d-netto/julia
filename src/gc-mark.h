// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef JL_GC_MARK_H
#define JL_GC_MARK_H

#include "gc.h"

#ifdef __cplusplus
extern "C" {
#endif

void gc_premark(jl_ptls_t ptls);
void gc_queue_thread_local(jl_gc_markqueue_t *mq, jl_ptls_t ptls);
void gc_queue_bt_buf(jl_gc_markqueue_t *mq, jl_ptls_t ptls);
void gc_queue_remset(jl_ptls_t ptls, jl_ptls_t ptls2);
void gc_mark_roots(jl_gc_markqueue_t *mq);
void gc_mark_queue_all_roots(jl_ptls_t ptls, jl_gc_markqueue_t *mq);
void gc_mark_finlist(jl_ptls_t ptls, arraylist_t *list, size_t start) JL_NOTSAFEPOINT;

JL_EXTENSION NOINLINE void gc_mark_loop(jl_ptls_t ptls);

#ifdef __cplusplus
}
#endif

#endif