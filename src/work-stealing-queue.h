// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef WORK_STEALING_QUEUE_H
#define WORK_STEALING_QUEUE_H

#include "julia_atomics.h"

#ifdef __cplusplus
extern "C" {
#endif

// =======
// Idempotent work-stealing stack
//
// * Michael M. M. et al. Idempotent Work Stealing
// =======

typedef struct {
    char *buffer;
    int32_t capacity;
    int32_t eltsz;
} ws_array_t;

static inline ws_array_t *create_ws_array(size_t capacity, int32_t eltsz) JL_NOTSAFEPOINT
{
    ws_array_t *a = (ws_array_t *)malloc_s(sizeof(ws_array_t));
    a->buffer = (char *)malloc_s(capacity * eltsz);
    a->capacity = capacity;
    a->eltsz = eltsz;
    return a;
}

typedef struct {
    int32_t tail;
    int32_t tag;
} ws_anchor_t;

typedef struct {
    _Atomic(ws_anchor_t) anchor;
    _Atomic(ws_array_t *) array;
} ws_queue_t;

static inline ws_array_t *ws_queue_push(ws_queue_t *q, void *elt) JL_NOTSAFEPOINT
{
    ws_anchor_t anc = jl_atomic_load_acquire(&q->anchor);
    ws_array_t *ary = jl_atomic_load_relaxed(&q->array);
    ws_array_t *old_ary = NULL;
    if (anc.tail == ary->capacity) {
        // Resize queue
        ws_array_t *new_ary = create_ws_array(2 * ary->capacity, ary->eltsz);
        memcpy(new_ary->buffer, ary->buffer, anc.tail * ary->eltsz);
        jl_atomic_store_relaxed(&q->array, new_ary);
        old_ary = ary;
        ary = new_ary;
    }
    memcpy(ary->buffer + anc.tail * ary->eltsz, elt, ary->eltsz);
    anc.tail++;
    anc.tag++;
    jl_atomic_store_release(&q->anchor, anc);
    return old_ary;
}

static inline void ws_queue_pop(ws_queue_t *q, void *dest) JL_NOTSAFEPOINT
{
    ws_anchor_t anc = jl_atomic_load_acquire(&q->anchor);
    ws_array_t *ary = jl_atomic_load_relaxed(&q->array);
    if (anc.tail == 0)
        // Empty queue
        return;
    anc.tail--;
    memcpy(dest, ary->buffer + anc.tail * ary->eltsz, ary->eltsz);
    jl_atomic_store_release(&q->anchor, anc);
}

static inline void ws_queue_steal_from(ws_queue_t *q, void *dest) JL_NOTSAFEPOINT
{
    ws_anchor_t anc = jl_atomic_load_acquire(&q->anchor);
    ws_array_t *ary = jl_atomic_load_acquire(&q->array);
    if (anc.tail == 0)
        // Empty queue
        return;
    memcpy(dest, ary->buffer + (anc.tail - 1) * ary->eltsz, ary->eltsz);
    ws_anchor_t anc2 = {anc.tail - 1, anc.tag};
    if (!jl_atomic_cmpswap(&q->anchor, &anc, anc2)) {
        // Steal failed
        memset(dest, 0, ary->eltsz);
        return;
    }
}

#ifdef __cplusplus
}
#endif

#endif
