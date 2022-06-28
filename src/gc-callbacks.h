// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef JL_GC_CB_H
#define JL_GC_CB_H

#include "gc.h"

#ifdef __cplusplus
extern "C" {
#endif

#define gc_invoke_callbacks(ty, list, args)                                 \
    do {                                                                    \
        for (jl_gc_callback_list_t *cb = list; cb != NULL; cb = cb->next) { \
            ((ty)(cb->func)) args;                                          \
        }                                                                   \
    } while (0)

typedef void (*jl_gc_cb_func_t)(void);

// Linked list of callback functions
typedef struct jl_gc_callback_list_t {
    struct jl_gc_callback_list_t *next;
    jl_gc_cb_func_t func;
} jl_gc_callback_list_t;

void jl_gc_register_callback(jl_gc_callback_list_t **list, jl_gc_cb_func_t func);
void jl_gc_deregister_callback(jl_gc_callback_list_t **list, jl_gc_cb_func_t func);

#ifdef __cplusplus
}
#endif

#endif