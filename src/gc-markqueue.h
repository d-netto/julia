// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef JL_GC_MARKQUEUE_H
#define JL_GC_MARKQUEUE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    struct _jl_value_t **start;
    struct _jl_value_t **current;
    struct _jl_value_t **end;
} jl_gc_markqueue_t;

// Double the mark queue
void gc_markqueue_resize(jl_gc_markqueue_t *mq);
// Push a work item to the queue
void gc_markqueue_push(jl_gc_markqueue_t *mq, struct _jl_value_t *obj);
// Pop from the mark queue
struct _jl_value_t *gc_markqueue_pop(jl_gc_markqueue_t *mq);

#ifdef __cplusplus
}
#endif

#endif