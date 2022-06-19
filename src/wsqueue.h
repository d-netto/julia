// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef WS_QUEUE_H
#define WS_QUEUE_H

#include "julia_atomics.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void **buffer;
    size_t capacity;
} ws_array_t;

ws_array_t *create_ws_array(size_t capacity);

// =======
// Chase and Lev's work-stealing queue, optimized for
// weak memory models by Le et al.
//
// * Chase D., Lev. Y. Dynamic Circular Work-Stealing queue
// * Le N. M. et al. Correct and Efficient Work-Stealing for
//   Weak Memory Models
// =======

typedef struct {
    _Atomic(int64_t) top;
    _Atomic(int64_t) bottom;
    _Atomic(ws_array_t *) array;
} ws_queue_t;

void ws_queue_push(ws_queue_t *dq, void *elt);

ws_array_t *ws_queue_resize(ws_queue_t *dq);

void *ws_queue_pop(ws_queue_t *dq);

void *ws_queue_steal_from(ws_queue_t *dq);

#ifdef __cplusplus
}
#endif

#endif

