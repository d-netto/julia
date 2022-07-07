// This file is a part of Julia. License is MIT: https://julialang.org/license

#include "gc-markqueue.h"
#include "gc.h"

#ifdef __cplusplus
extern "C" {
#endif

void gc_markqueue_resize(jl_gc_markqueue_t *mq)
{
    jl_value_t **old_start = mq->start;
    size_t old_queue_size = (mq->end - mq->start);
    size_t offset = (mq->current - old_start);
    mq->start =
        (jl_value_t **)realloc_s(old_start, 2 * old_queue_size * sizeof(jl_value_t *));
    mq->current = (mq->start + offset);
    mq->end = (mq->start + 2 * old_queue_size);
}

void gc_markqueue_push(jl_gc_markqueue_t *mq, jl_value_t *obj)
{
    if (__unlikely(mq->current == mq->end))
        gc_markqueue_resize(mq);
    *mq->current = obj;
    mq->current++;
}

jl_value_t *gc_markqueue_pop(jl_gc_markqueue_t *mq)
{
    if (mq->current == mq->start)
        return NULL;
    mq->current--;
    jl_value_t *obj = *mq->current;
    return obj;
}

#ifdef __cplusplus
}
#endif