// This file is a part of Julia. License is MIT: https://julialang.org/license

#include "gc-sweep.h"
#include "gc-alloc.h"
#include "gc-callbacks.h"
#include "gc-finalizers.h"
#include "gc-sweep.h"
#include "gc.h"
#include "julia_assert.h"
#include "julia_gcext.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PROMOTE_AGE 1
#define inc_sat(v, s) v = (v) >= s ? s : (v) + 1

// GC knobs and self-measurement variables
int64_t last_gc_total_bytes = 0;
#ifdef _P64
size_t max_collect_interval = 1250000000UL;
// Eventually we can expose this to the user/ci.
memsize_t max_total_memory = (memsize_t)2 * 1024 * 1024 * 1024 * 1024 * 1024;
#else
size_t max_collect_interval = 500000000UL;
// Work really hard to stay within 2GB
// Alternative is to risk running out of address space
// on 32 bit architectures.
memsize_t max_total_memory = (memsize_t)2 * 1024 * 1024 * 1024;
#endif

// Full collection heuristics
int64_t live_bytes = 0;
int64_t promoted_bytes = 0;
int64_t last_live_bytes = 0; // live_bytes at last collection
int64_t t_start = 0; // Time GC starts;
int64_t lazy_freed_pages = 0;
#ifdef __GLIBC__
// maxrss at last malloc_trim
int64_t last_trim_maxrss = 0;
#endif

int prev_sweep_full = 1;

void gc_sweep_weak_refs(void)
{
    for (int i = 0; i < jl_n_threads; i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[i];
        size_t n = 0;
        size_t ndel = 0;
        size_t l = ptls2->heap.weak_refs.len;
        void **lst = ptls2->heap.weak_refs.items;
        if (l == 0)
            continue;
        while (1) {
            jl_weakref_t *wr = (jl_weakref_t *)lst[n];
            if (gc_marked(jl_astaggedvalue(wr)->bits.gc))
                n++;
            else
                ndel++;
            if (n >= l - ndel)
                break;
            void *tmp = lst[n];
            lst[n] = lst[n + ndel];
            lst[n + ndel] = tmp;
        }
        ptls2->heap.weak_refs.len -= ndel;
    }
}

// Sweep list rooted at *pv, removing and freeing any unmarked objects.
// Return pointer to last `next` field in the culled list.
bigval_t **gc_sweep_big_list(int sweep_full, bigval_t **pv) JL_NOTSAFEPOINT
{
    bigval_t *v = *pv;
    while (v) {
        bigval_t *nxt = v->next;
        int bits = v->bits.gc;
        int old_bits = bits;
        if (gc_marked(bits)) {
            pv = &v->next;
            int age = v->age;
            if (age >= PROMOTE_AGE || bits == GC_OLD_MARKED) {
                if (sweep_full || bits == GC_MARKED) {
                    bits = GC_OLD;
                }
            }
            else {
                inc_sat(age, PROMOTE_AGE);
                v->age = age;
                bits = GC_CLEAN;
            }
            v->bits.gc = bits;
        }
        else {
            // Remove v from list and free it
            *pv = nxt;
            if (nxt)
                nxt->prev = pv;
            gc_num.freed += v->sz & ~3;
#ifdef MEMDEBUG
            memset(v, 0xbb, v->sz & ~3);
#endif
            gc_invoke_callbacks(jl_gc_cb_notify_external_free_t,
                                gc_cblist_notify_external_free, (v));
            jl_free_aligned(v);
        }
        gc_time_count_big(old_bits, bits);
        v = nxt;
    }
    return pv;
}

void gc_sweep_big(jl_ptls_t ptls, int sweep_full) JL_NOTSAFEPOINT
{
    gc_time_big_start();
    for (int i = 0; i < jl_n_threads; i++)
        gc_sweep_big_list(sweep_full, &jl_all_tls_states[i]->heap.big_objects);
    if (sweep_full) {
        bigval_t **last_next = gc_sweep_big_list(sweep_full, &big_objects_marked);
        // Move all survivors from big_objects_marked list to big_objects list.
        if (ptls->heap.big_objects)
            ptls->heap.big_objects->prev = last_next;
        *last_next = ptls->heap.big_objects;
        ptls->heap.big_objects = big_objects_marked;
        if (ptls->heap.big_objects)
            ptls->heap.big_objects->prev = &ptls->heap.big_objects;
        big_objects_marked = NULL;
    }
    gc_time_big_end();
}

void gc_free_array(jl_array_t *a) JL_NOTSAFEPOINT
{
    if (a->flags.how == 2) {
        char *d = (char *)a->data - a->offset * a->elsize;
        if (a->flags.isaligned)
            jl_free_aligned(d);
        else
            free(d);
        gc_num.freed += jl_array_nbytes(a);
        gc_num.freecall++;
    }
}

void gc_sweep_malloced_arrays(void) JL_NOTSAFEPOINT
{
    gc_time_mallocd_array_start();
    for (int t_i = 0; t_i < jl_n_threads; t_i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[t_i];
        mallocarray_t *ma = ptls2->heap.mallocarrays;
        mallocarray_t **pma = &ptls2->heap.mallocarrays;
        while (ma) {
            mallocarray_t *nxt = ma->next;
            int bits = jl_astaggedvalue(ma->a)->bits.gc;
            if (gc_marked(bits)) {
                pma = &ma->next;
            }
            else {
                *pma = nxt;
                assert(ma->a->flags.how == 2);
                gc_free_array(ma->a);
                ma->next = ptls2->heap.mafreelist;
                ptls2->heap.mafreelist = ma;
            }
            gc_time_count_mallocd_array(bits);
            ma = nxt;
        }
    }
    gc_time_mallocd_array_end();
}

// sweep over all memory for all pagetable1 that may contain allocated pages
void gc_sweep_pool_pagetable(jl_taggedvalue_t ***pfl, int sweep_full) JL_NOTSAFEPOINT
{
    if (REGION2_PG_COUNT == 1) { // compile-time optimization
        pagetable1_t *pagetable1 = memory_map.meta1[0];
        if (pagetable1)
            sweep_pool_pagetable1(pfl, pagetable1, sweep_full);
        return;
    }
    unsigned ub = 0;
    for (unsigned pg_i = 0; pg_i <= memory_map.ub; pg_i++) {
        uint32_t line = memory_map.allocmap1[pg_i];
        unsigned j;
        for (j = 0; line; j++, line >>= 1) {
            unsigned next = ffs_u32(line);
            j += next;
            line >>= next;
            pagetable1_t *pagetable1 = memory_map.meta1[pg_i * 32 + j];
            if (pagetable1 && !sweep_pool_pagetable1(pfl, pagetable1, sweep_full))
                memory_map.allocmap1[pg_i] &=
                    ~(1 << j); // no allocations found, remember that for next time
        }
        if (memory_map.allocmap1[pg_i]) {
            ub = pg_i;
        }
    }
    memory_map.ub = ub;
}

// setup the data-structures for a sweep over all memory pools
void gc_sweep_pool(int sweep_full)
{
    gc_time_pool_start();
    lazy_freed_pages = 0;

    // For the benfit of the analyzer, which doesn't know that jl_n_threads
    // doesn't change over the course of this function
    size_t n_threads = jl_n_threads;

    // allocate enough space to hold the end of the free list chain
    // for every thread and pool size
    jl_taggedvalue_t ***pfl = (jl_taggedvalue_t ***)alloca(n_threads * JL_GC_N_POOLS *
                                                           sizeof(jl_taggedvalue_t **));

    // update metadata of pages that were pointed to by freelist or newpages from a pool
    // i.e. pages being the current allocation target
    for (int t_i = 0; t_i < n_threads; t_i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[t_i];
        for (int i = 0; i < JL_GC_N_POOLS; i++) {
            jl_gc_pool_t *p = &ptls2->heap.norm_pools[i];
            jl_taggedvalue_t *last = p->freelist;
            if (last) {
                jl_gc_pagemeta_t *pg = jl_assume(page_metadata(last));
                gc_pool_sync_nfree(pg, last);
                pg->has_young = 1;
            }
            p->freelist = NULL;
            pfl[t_i * JL_GC_N_POOLS + i] = &p->freelist;

            last = p->newpages;
            if (last) {
                char *last_p = (char *)last;
                jl_gc_pagemeta_t *pg = jl_assume(page_metadata(last_p - 1));
                assert(last_p - gc_page_data(last_p - 1) >= GC_PAGE_OFFSET);
                pg->nfree = (GC_PAGE_SZ - (last_p - gc_page_data(last_p - 1))) / p->osize;
                pg->has_young = 1;
            }
            p->newpages = NULL;
        }
    }

    // the actual sweeping
    gc_sweep_pool_pagetable(pfl, sweep_full);

    // null out terminal pointers of free lists
    for (int t_i = 0; t_i < n_threads; t_i++) {
        for (int i = 0; i < JL_GC_N_POOLS; i++) {
            *pfl[t_i * JL_GC_N_POOLS + i] = NULL;
        }
    }

    gc_time_pool_end(sweep_full);
}

// find unmarked objects that need to be finalized from the finalizer list "list".
// this must happen last in the mark phase.
void gc_sweep_finalizer_list(arraylist_t *list)
{
    void **items = list->items;
    size_t len = list->len;
    size_t j = 0;
    for (size_t i = 0; i < len; i += 2) {
        void *v0 = items[i];
        void *v = gc_ptr_clear_tag(v0, 1);
        if (__unlikely(!v0)) {
            // remove from this list
            continue;
        }

        void *fin = items[i + 1];
        int isfreed = !gc_marked(jl_astaggedvalue(v)->bits.gc);
        int isold = (list != &finalizer_list_marked &&
                     jl_astaggedvalue(v)->bits.gc == GC_OLD_MARKED &&
                     jl_astaggedvalue(fin)->bits.gc == GC_OLD_MARKED);
        if (isfreed || isold) {
            // remove from this list
        }
        else {
            if (j < i) {
                items[j] = items[i];
                items[j + 1] = items[i + 1];
            }
            j += 2;
        }
        if (isfreed) {
            schedule_finalization(v0, fin);
        }
        if (isold) {
            // The caller relies on the new objects to be pushed to the end of
            // the list!!
            arraylist_push(&finalizer_list_marked, v0);
            arraylist_push(&finalizer_list_marked, fin);
        }
    }
    list->len = j;
}

// explicitly scheduled objects for the sweepfunc callback
void gc_sweep_foreign_objs_in_list(arraylist_t *objs)
{
    size_t p = 0;
    for (size_t i = 0; i < objs->len; i++) {
        jl_value_t *v = (jl_value_t *)(objs->items[i]);
        jl_datatype_t *t = (jl_datatype_t *)(jl_typeof(v));
        const jl_datatype_layout_t *layout = t->layout;
        jl_fielddescdyn_t *desc = (jl_fielddescdyn_t *)jl_dt_layout_fields(layout);

        int bits = jl_astaggedvalue(v)->bits.gc;
        if (!gc_marked(bits))
            desc->sweepfunc(v);
        else
            objs->items[p++] = v;
    }
    objs->len = p;
}

void gc_sweep_foreign_objs(void)
{
    for (int i = 0; i < jl_n_threads; i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[i];
        gc_sweep_foreign_objs_in_list(&ptls2->sweep_objs);
    }
}

#ifdef __cplusplus
}
#endif