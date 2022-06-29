// This file is a part of Julia. License is MIT: https://julialang.org/license

#include "gc-callbacks.h"
#include "gc.h"
#include "julia_assert.h"
#include "julia_gcext.h"

#ifdef __cplusplus
extern "C" {
#endif

void schedule_finalization(void *o, void *f) JL_NOTSAFEPOINT;
void run_finalizer(jl_task_t *ct, jl_value_t *o, jl_value_t *ff);
void finalize_object(arraylist_t *list, jl_value_t *o, arraylist_t *copied_list,
                     int need_sync) JL_NOTSAFEPOINT;
void jl_gc_push_arraylist(jl_task_t *ct, arraylist_t *list);
void jl_gc_run_finalizers_in_list(jl_task_t *ct, arraylist_t *list);
void run_finalizers(jl_task_t *ct);

JL_DLLEXPORT void jl_gc_run_pending_finalizers(jl_task_t *ct);
JL_DLLEXPORT int jl_gc_get_finalizers_inhibited(jl_ptls_t ptls);
JL_DLLEXPORT void jl_gc_disable_finalizers_internal(void);
JL_DLLEXPORT void jl_gc_enable_finalizers_internal(void);
JL_DLLEXPORT void jl_gc_enable_finalizers(jl_task_t *ct, int on);
JL_DLLEXPORT void jl_gc_add_ptr_finalizer(jl_ptls_t ptls, jl_value_t *v,
                                          void *f) JL_NOTSAFEPOINT;
JL_DLLEXPORT void jl_gc_add_finalizer_th(jl_ptls_t ptls, jl_value_t *v,
                                         jl_function_t *f) JL_NOTSAFEPOINT;
JL_DLLEXPORT void jl_finalize_th(jl_task_t *ct, jl_value_t *o);

void schedule_all_finalizers(arraylist_t *flist) JL_NOTSAFEPOINT;
void jl_gc_run_all_finalizers(jl_task_t *ct);
void jl_gc_add_finalizer_(jl_ptls_t ptls, void *v, void *f) JL_NOTSAFEPOINT;

JL_DLLEXPORT void jl_gc_add_finalizer(jl_value_t *v, jl_function_t *f);
JL_DLLEXPORT void jl_finalize(jl_value_t *o);

#ifdef __cplusplus
}
#endif