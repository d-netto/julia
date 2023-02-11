// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef WORK_STEALING_QUEUE_H
#define WORK_STEALING_QUEUE_H

#include "julia_atomics.h"

#ifdef __cplusplus
extern "C" {
#endif

// =======
// Idempotent work-stealing deque
//
// * Michael M. M. et al. Idempotent Work Stealing
// =======

typedef struct {
    void **buffer;
    int64_t capacity;
} ws_array_t;

static inline ws_array_t *create_ws_array(size_t capacity, size_t eltsz) JL_NOTSAFEPOINT
{
    ws_array_t *a = (ws_array_t *)malloc_s(sizeof(ws_array_t));
    a->buffer = (void **)malloc_s(capacity * eltsz);
    a->capacity = capacity;
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
        ws_array_t *new_ary = create_ws_array(2 * ary->capacity, sizeof(void *));
        memcpy(new_ary->buffer, ary->buffer, anc.tail * sizeof(void *));
        jl_atomic_store_relaxed(&q->array, new_ary);
        old_ary = ary;
        ary = new_ary;
    }
    ary->buffer[anc.tail] = elt;
    anc.tail++;
    anc.tag++;
    jl_atomic_store_release(&q->anchor, anc);
    return old_ary;
}

static inline void *ws_queue_pop(ws_queue_t *q) JL_NOTSAFEPOINT
{
    ws_anchor_t anc = jl_atomic_load_acquire(&q->anchor);
    ws_array_t *ary = jl_atomic_load_relaxed(&q->array);
    if (anc.tail == 0)
        // Empty queue
        return NULL;
    anc.tail--;
    void *elt = ary->buffer[anc.tail];
    jl_atomic_store_release(&q->anchor, anc);
    return elt;
}

static inline void *ws_queue_steal_from(ws_queue_t *q) JL_NOTSAFEPOINT
{
    ws_anchor_t anc = jl_atomic_load_acquire(&q->anchor);
    ws_array_t *ary = jl_atomic_load_acquire(&q->array);
    if (anc.tail == 0)
        // Empty queue
        return NULL;
    void *elt = ary->buffer[anc.tail - 1];
    ws_anchor_t anc2 = {anc.tail - 1, anc.tag};
    if (!jl_atomic_cmpswap(&q->anchor, &anc, anc2))
        // Steal failed
        return NULL;
    return elt;
}

#ifdef __cplusplus
}
#endif

#endif
