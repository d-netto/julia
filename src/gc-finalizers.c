// This file is a part of Julia. License is MIT: https://julialang.org/license

#include "gc-callbacks.h"
#include "gc.h"
#include "julia_assert.h"
#include "julia_gcext.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 Finalizers
*/

jl_mutex_t finalizers_lock;
uv_mutex_t gc_cache_lock;

void schedule_finalization(void *o, void *f) JL_NOTSAFEPOINT
{
    arraylist_push(&to_finalize, o);
    arraylist_push(&to_finalize, f);
    // doesn't need release, since we'll keep checking (on the reader) until we see the work
    // and release our lock, and that will have a release barrier by then
    jl_atomic_store_relaxed(&jl_gc_have_pending_finalizers, 1);
}

void run_finalizer(jl_task_t *ct, jl_value_t *o, jl_value_t *ff)
{
    if (gc_ptr_tag(o, 1)) {
        ((void (*)(void *))ff)(gc_ptr_clear_tag(o, 1));
        return;
    }
    jl_value_t *args[2] = {ff, o};
    JL_TRY {
        size_t last_age = ct->world_age;
        ct->world_age = jl_atomic_load_acquire(&jl_world_counter);
        jl_apply(args, 2);
        ct->world_age = last_age;
    }
    JL_CATCH {
        jl_printf((JL_STREAM *)STDERR_FILENO, "error in running finalizer: ");
        jl_static_show((JL_STREAM *)STDERR_FILENO, jl_current_exception());
        jl_printf((JL_STREAM *)STDERR_FILENO, "\n");
        jlbacktrace(); // written to STDERR_FILENO
    }
}

// if `need_sync` is true, the `list` is the `finalizers` list of another
// thread and we need additional synchronizations
void finalize_object(arraylist_t *list, jl_value_t *o, arraylist_t *copied_list,
                     int need_sync) JL_NOTSAFEPOINT
{
    // The acquire load makes sure that the first `len` objects are valid.
    // If `need_sync` is true, all mutations of the content should be limited
    // to the first `oldlen` elements and no mutation is allowed after the
    // new length is published with the `cmpxchg` at the end of the function.
    // This way, the mutation should not conflict with the owning thread,
    // which only writes to locations later than `len`
    // and will not resize the buffer without acquiring the lock.
    size_t len =
        need_sync ? jl_atomic_load_acquire((_Atomic(size_t) *)&list->len) : list->len;
    size_t oldlen = len;
    void **items = list->items;
    size_t j = 0;
    for (size_t i = 0; i < len; i += 2) {
        void *v = items[i];
        int move = 0;
        if (o == (jl_value_t *)gc_ptr_clear_tag(v, 1)) {
            void *f = items[i + 1];
            move = 1;
            arraylist_push(copied_list, v);
            arraylist_push(copied_list, f);
        }
        if (move || __unlikely(!v)) {
            // remove item
        }
        else {
            if (j < i) {
                items[j] = items[i];
                items[j + 1] = items[i + 1];
            }
            j += 2;
        }
    }
    len = j;
    if (oldlen == len)
        return;
    if (need_sync) {
        // The memset needs to be unconditional since the thread might have
        // already read the length.
        // The `memset` (like any other content mutation) has to be done
        // **before** the `cmpxchg` which publishes the length.
        memset(&items[len], 0, (oldlen - len) * sizeof(void *));
        jl_atomic_cmpswap((_Atomic(size_t) *)&list->len, &oldlen, len);
    }
    else {
        list->len = len;
    }
}

// The first two entries are assumed to be empty and the rest are assumed to
// be pointers to `jl_value_t` objects
void jl_gc_push_arraylist(jl_task_t *ct, arraylist_t *list)
{
    void **items = list->items;
    items[0] = (void *)JL_GC_ENCODE_PUSHARGS(list->len - 2);
    items[1] = ct->gcstack;
    ct->gcstack = (jl_gcframe_t *)items;
}

// Same assumption as `jl_gc_push_arraylist`. Requires the finalizers lock
// to be hold for the current thread and will release the lock when the
// function returns.
void jl_gc_run_finalizers_in_list(jl_task_t *ct, arraylist_t *list)
{
    // Avoid marking `ct` as non-migratable via an `@async` task (as noted in the docstring
    // of `finalizer`) in a finalizer:
    uint8_t sticky = ct->sticky;
    // empty out the first two entries for the GC frame
    arraylist_push(list, list->items[0]);
    arraylist_push(list, list->items[1]);
    jl_gc_push_arraylist(ct, list);
    jl_value_t **items = (jl_value_t **)list->items;
    size_t len = list->len;
    JL_UNLOCK_NOGC(&finalizers_lock);
    // run finalizers in reverse order they were added, so lower-level finalizers run last
    for (size_t i = len - 4; i >= 2; i -= 2)
        run_finalizer(ct, items[i], items[i + 1]);
    // first entries were moved last to make room for GC frame metadata
    run_finalizer(ct, items[len - 2], items[len - 1]);
    // matches the jl_gc_push_arraylist above
    JL_GC_POP();
    ct->sticky = sticky;
}

void run_finalizers(jl_task_t *ct)
{
    // Racy fast path:
    // The race here should be OK since the race can only happen if
    // another thread is writing to it with the lock held. In such case,
    // we don't need to run pending finalizers since the writer thread
    // will flush it.
    if (to_finalize.len == 0)
        return;
    JL_LOCK_NOGC(&finalizers_lock);
    if (to_finalize.len == 0) {
        JL_UNLOCK_NOGC(&finalizers_lock);
        return;
    }
    arraylist_t copied_list;
    memcpy(&copied_list, &to_finalize, sizeof(copied_list));
    if (to_finalize.items == to_finalize._space) {
        copied_list.items = copied_list._space;
    }
    jl_atomic_store_relaxed(&jl_gc_have_pending_finalizers, 0);
    arraylist_new(&to_finalize, 0);
    // This releases the finalizers lock.
    jl_gc_run_finalizers_in_list(ct, &copied_list);
    arraylist_free(&copied_list);
}

JL_DLLEXPORT void jl_gc_run_pending_finalizers(jl_task_t *ct)
{
    if (ct == NULL)
        ct = jl_current_task;
    jl_ptls_t ptls = ct->ptls;
    if (!ptls->in_finalizer && ptls->locks.len == 0 && ptls->finalizers_inhibited == 0) {
        ptls->in_finalizer = 1;
        run_finalizers(ct);
        ptls->in_finalizer = 0;
    }
}

JL_DLLEXPORT int jl_gc_get_finalizers_inhibited(jl_ptls_t ptls)
{
    if (ptls == NULL)
        ptls = jl_current_task->ptls;
    return ptls->finalizers_inhibited;
}

JL_DLLEXPORT void jl_gc_disable_finalizers_internal(void)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    ptls->finalizers_inhibited++;
}

JL_DLLEXPORT void jl_gc_enable_finalizers_internal(void)
{
    jl_task_t *ct = jl_current_task;
#ifdef NDEBUG
    ct->ptls->finalizers_inhibited--;
#else
    jl_gc_enable_finalizers(ct, 1);
#endif
}

JL_DLLEXPORT void jl_gc_enable_finalizers(jl_task_t *ct, int on)
{
    if (ct == NULL)
        ct = jl_current_task;
    jl_ptls_t ptls = ct->ptls;
    int old_val = ptls->finalizers_inhibited;
    int new_val = old_val + (on ? -1 : 1);
    if (new_val < 0) {
        JL_TRY {
            jl_error(""); // get a backtrace
        }
        JL_CATCH {
            jl_printf((JL_STREAM *)STDERR_FILENO,
                      "WARNING: GC finalizers already enabled on this thread.\n");
            // Only print the backtrace once, to avoid spamming the logs
            static int backtrace_printed = 0;
            if (backtrace_printed == 0) {
                backtrace_printed = 1;
                jlbacktrace(); // written to STDERR_FILENO
            }
        }
        return;
    }
    ptls->finalizers_inhibited = new_val;
    if (jl_atomic_load_relaxed(&jl_gc_have_pending_finalizers)) {
        jl_gc_run_pending_finalizers(ct);
    }
}

void schedule_all_finalizers(arraylist_t *flist) JL_NOTSAFEPOINT
{
    void **items = flist->items;
    size_t len = flist->len;
    for (size_t i = 0; i < len; i += 2) {
        void *v = items[i];
        void *f = items[i + 1];
        if (__unlikely(!v))
            continue;
        schedule_finalization(v, f);
    }
    flist->len = 0;
}

void jl_gc_run_all_finalizers(jl_task_t *ct)
{
    schedule_all_finalizers(&finalizer_list_marked);
    for (int i = 0; i < jl_n_threads; i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[i];
        schedule_all_finalizers(&ptls2->finalizers);
    }
    run_finalizers(ct);
}

void jl_gc_add_finalizer_(jl_ptls_t ptls, void *v, void *f) JL_NOTSAFEPOINT
{
    assert(jl_atomic_load_relaxed(&ptls->gc_state) == 0);
    arraylist_t *a = &ptls->finalizers;
    // This acquire load and the release store at the end are used to
    // synchronize with `finalize_object` on another thread. Apart from the GC,
    // which is blocked by entering a unsafe region, there might be only
    // one other thread accessing our list in `finalize_object`
    // (only one thread since it needs to acquire the finalizer lock).
    // Similar to `finalize_object`, all content mutation has to be done
    // between the acquire and the release of the length.
    size_t oldlen = jl_atomic_load_acquire((_Atomic(size_t) *)&a->len);
    if (__unlikely(oldlen + 2 > a->max)) {
        JL_LOCK_NOGC(&finalizers_lock);
        // `a->len` might have been modified.
        // Another possibility is to always grow the array to `oldlen + 2` but
        // it's simpler this way and uses slightly less memory =)
        oldlen = a->len;
        arraylist_grow(a, 2);
        a->len = oldlen;
        JL_UNLOCK_NOGC(&finalizers_lock);
    }
    void **items = a->items;
    items[oldlen] = v;
    items[oldlen + 1] = f;
    jl_atomic_store_release((_Atomic(size_t) *)&a->len, oldlen + 2);
}

JL_DLLEXPORT void jl_gc_add_ptr_finalizer(jl_ptls_t ptls, jl_value_t *v,
                                          void *f) JL_NOTSAFEPOINT
{
    jl_gc_add_finalizer_(ptls, (void *)(((uintptr_t)v) | 1), f);
}

JL_DLLEXPORT void jl_gc_add_finalizer_th(jl_ptls_t ptls, jl_value_t *v,
                                         jl_function_t *f) JL_NOTSAFEPOINT
{
    if (__unlikely(jl_typeis(f, jl_voidpointer_type))) {
        jl_gc_add_ptr_finalizer(ptls, v, jl_unbox_voidpointer(f));
    }
    else {
        jl_gc_add_finalizer_(ptls, v, f);
    }
}

JL_DLLEXPORT void jl_finalize_th(jl_task_t *ct, jl_value_t *o)
{
    JL_LOCK_NOGC(&finalizers_lock);
    // Copy the finalizers into a temporary list so that code in the finalizer
    // won't change the list as we loop through them.
    // This list is also used as the GC frame when we are running the finalizers
    arraylist_t copied_list;
    arraylist_new(&copied_list, 0);
    // No need to check the to_finalize list since the user is apparently
    // still holding a reference to the object
    for (int i = 0; i < jl_n_threads; i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[i];
        finalize_object(&ptls2->finalizers, o, &copied_list,
                        jl_atomic_load_relaxed(&ct->tid) != i);
    }
    finalize_object(&finalizer_list_marked, o, &copied_list, 0);
    if (copied_list.len > 0) {
        // This releases the finalizers lock.
        jl_gc_run_finalizers_in_list(ct, &copied_list);
    }
    else {
        JL_UNLOCK_NOGC(&finalizers_lock);
    }
    arraylist_free(&copied_list);
}

JL_DLLEXPORT void jl_gc_add_finalizer(jl_value_t *v, jl_function_t *f)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    jl_gc_add_finalizer_th(ptls, v, f);
}

JL_DLLEXPORT void jl_finalize(jl_value_t *o)
{
    jl_finalize_th(jl_current_task, o);
}

#ifdef __cplusplus
}
#endif