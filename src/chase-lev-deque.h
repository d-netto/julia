// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef CHASE_LEV_DEQUE_H
#define CHASE_LEV_DEQUE_H

#include "julia_atomics.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void **buffer;
    int64_t capacity;
} ws_array_t;

STATIC_INLINE ws_array_t *create_ws_array(size_t capacity) JL_NOTSAFEPOINT
{
    ws_array_t *a = (ws_array_t *)malloc_s(sizeof(ws_array_t));
    a->buffer = (void **)malloc_s(capacity * sizeof(void *));
    a->capacity = capacity;
    return a;
}

// =======
// Chase and Lev's work-stealing queue, optimized for
// weak memory models by Le et al.
//
// * Chase D., Lev Y. Dynamic Circular Work-Stealing queue
// * Le N. M. et al. Correct and Efficient Work-Stealing for
//   Weak Memory Models
// =======

typedef struct {
    _Atomic(int64_t) top;
    _Atomic(int64_t) bottom;
    _Atomic(ws_array_t *) array;
} ws_queue_t;

STATIC_INLINE int ws_queue_push(ws_queue_t *q, void *v) JL_NOTSAFEPOINT
{
    int64_t b = jl_atomic_load_relaxed(&q->bottom);
    int64_t t = jl_atomic_load_acquire(&q->top);
    ws_array_t *a = jl_atomic_load_relaxed(&q->array);
    if (__unlikely(b - t > a->capacity - 1)) {
        // Queue is full
        return 0;
    }
    jl_atomic_store_relaxed((_Atomic(void *) *)&a->buffer[b % a->capacity], v);
    jl_fence_release();
    jl_atomic_store_relaxed(&q->bottom, b + 1);
    return 1;
}

STATIC_INLINE void *ws_queue_pop(ws_queue_t *q) JL_NOTSAFEPOINT
{
        int64_t b = jl_atomic_load_relaxed(&q->bottom) - 1;
    ws_array_t *a = jl_atomic_load_relaxed(&q->array);
    jl_atomic_store_relaxed(&q->bottom, b);
#if defined(_CPU_X86_64_)
    __asm__ volatile ("lock orq $0, (%rsp)");
#else
    jl_fence();
#endif
    int64_t t = jl_atomic_load_relaxed(&q->top);
    void *v;
    if (__likely(t <= b)) {
        v = jl_atomic_load_relaxed((_Atomic(void *) *)&a->buffer[b % a->capacity]);
        if (t == b) {
            if (!jl_atomic_cmpswap(&q->top, &t, t + 1))
                v = NULL;
            jl_atomic_store_relaxed(&q->bottom, b + 1);
        }
    }
    else {
        v = NULL;
        jl_atomic_store_relaxed(&q->bottom, b + 1);
    }
    return v;
}

STATIC_INLINE void *ws_queue_steal_from(ws_queue_t *q) JL_NOTSAFEPOINT
{
        int64_t t = jl_atomic_load_acquire(&q->top);
#if defined(_CPU_X86_64_)
    __asm__ volatile ("lock orq $0, (%rsp)");
#else
    jl_fence();
#endif
    int64_t b = jl_atomic_load_acquire(&q->bottom);
    void *v = NULL;
    if (t < b) {
        ws_array_t *a = jl_atomic_load_relaxed(&q->array);
        v = jl_atomic_load_relaxed((_Atomic(void *) *)&a->buffer[t % a->capacity]);
        if (!jl_atomic_cmpswap(&q->top, &t, t + 1))
            return NULL;
    }
    return v;
}

#ifdef __cplusplus
}
#endif

#endif