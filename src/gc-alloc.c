// This file is a part of Julia. License is MIT: https://julialang.org/license

#include "gc-alloc.h"
#include "gc-callbacks.h"
#include "gc-sweep.h"
#include "gc.h"
#include "julia_assert.h"
#include "julia_gcext.h"

#ifdef __cplusplus
extern "C" {
#endif

void *sysimg_base = NULL;
void *sysimg_end = NULL;

uv_mutex_t gc_perm_lock;
uintptr_t gc_perm_pool = 0;
uintptr_t gc_perm_end = 0;

JL_DLLEXPORT void *jl_malloc(size_t sz)
{
    int64_t *p = (int64_t *)jl_gc_counted_malloc(sz + JL_SMALL_BYTE_ALIGNMENT);
    if (p == NULL)
        return NULL;
    p[0] = sz;
    return (void *)(p + 2); // assumes JL_SMALL_BYTE_ALIGNMENT == 16
}

JL_DLLEXPORT void *jl_calloc(size_t nm, size_t sz)
{
    if (nm > SSIZE_MAX / sz - JL_SMALL_BYTE_ALIGNMENT)
        return NULL;
    return _unchecked_calloc(nm, sz);
}

JL_DLLEXPORT void jl_free(void *p)
{
    if (p != NULL) {
        int64_t *pp = (int64_t *)p - 2;
        size_t sz = pp[0];
        jl_gc_counted_free_with_size(pp, sz + JL_SMALL_BYTE_ALIGNMENT);
    }
}

JL_DLLEXPORT void *jl_realloc(void *p, size_t sz)
{
    int64_t *pp;
    size_t szold;
    if (p == NULL) {
        pp = NULL;
        szold = 0;
    }
    else {
        pp = (int64_t *)p - 2;
        szold = pp[0] + JL_SMALL_BYTE_ALIGNMENT;
    }
    int64_t *pnew = (int64_t *)jl_gc_counted_realloc_with_old_size(
        pp, szold, sz + JL_SMALL_BYTE_ALIGNMENT);
    if (pnew == NULL)
        return NULL;
    pnew[0] = sz;
    return (void *)(pnew + 2); // assumes JL_SMALL_BYTE_ALIGNMENT == 16
}

// Allocating blocks for Arrays and Strings
JL_DLLEXPORT void *jl_gc_managed_malloc(size_t sz)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    maybe_collect(ptls);
    size_t allocsz = LLT_ALIGN(sz, JL_CACHE_BYTE_ALIGNMENT);
    if (allocsz < sz) // overflow in adding offs, size was "negative"
        jl_throw(jl_memory_exception);
    jl_atomic_store_relaxed(&ptls->gc_num.allocd,
                            jl_atomic_load_relaxed(&ptls->gc_num.allocd) + allocsz);
    jl_atomic_store_relaxed(&ptls->gc_num.malloc,
                            jl_atomic_load_relaxed(&ptls->gc_num.malloc) + 1);
    int last_errno = errno;
#ifdef _OS_WINDOWS_
    DWORD last_error = GetLastError();
#endif
    void *b = malloc_cache_align(allocsz);
    if (b == NULL)
        jl_throw(jl_memory_exception);
#ifdef _OS_WINDOWS_
    SetLastError(last_error);
#endif
    errno = last_errno;
    // jl_gc_managed_malloc is currently always used for allocating array buffers.
    maybe_record_alloc_to_profile((jl_value_t *)b, sz, (jl_datatype_t *)jl_buff_tag);
    return b;
}

// Allocation wrappers that track allocation and let collection run
JL_DLLEXPORT void *jl_gc_counted_malloc(size_t sz)
{
    jl_gcframe_t **pgcstack = jl_get_pgcstack();
    jl_task_t *ct = jl_current_task;
    if (pgcstack && ct->world_age) {
        jl_ptls_t ptls = ct->ptls;
        maybe_collect(ptls);
        jl_atomic_store_relaxed(&ptls->gc_num.allocd,
                                jl_atomic_load_relaxed(&ptls->gc_num.allocd) + sz);
        jl_atomic_store_relaxed(&ptls->gc_num.malloc,
                                jl_atomic_load_relaxed(&ptls->gc_num.malloc) + 1);
    }
    return malloc(sz);
}

JL_DLLEXPORT void *jl_gc_counted_calloc(size_t nm, size_t sz)
{
    jl_gcframe_t **pgcstack = jl_get_pgcstack();
    jl_task_t *ct = jl_current_task;
    if (pgcstack && ct->world_age) {
        jl_ptls_t ptls = ct->ptls;
        maybe_collect(ptls);
        jl_atomic_store_relaxed(&ptls->gc_num.allocd,
                                jl_atomic_load_relaxed(&ptls->gc_num.allocd) + nm * sz);
        jl_atomic_store_relaxed(&ptls->gc_num.malloc,
                                jl_atomic_load_relaxed(&ptls->gc_num.malloc) + 1);
    }
    return calloc(nm, sz);
}

JL_DLLEXPORT void jl_gc_counted_free_with_size(void *p, size_t sz)
{
    jl_gcframe_t **pgcstack = jl_get_pgcstack();
    jl_task_t *ct = jl_current_task;
    free(p);
    if (pgcstack && ct->world_age) {
        jl_ptls_t ptls = ct->ptls;
        jl_atomic_store_relaxed(&ptls->gc_num.freed,
                                jl_atomic_load_relaxed(&ptls->gc_num.freed) + sz);
        jl_atomic_store_relaxed(&ptls->gc_num.freecall,
                                jl_atomic_load_relaxed(&ptls->gc_num.freecall) + 1);
    }
}

JL_DLLEXPORT void *jl_gc_counted_realloc_with_old_size(void *p, size_t old, size_t sz)
{
    jl_gcframe_t **pgcstack = jl_get_pgcstack();
    jl_task_t *ct = jl_current_task;
    if (pgcstack && ct->world_age) {
        jl_ptls_t ptls = ct->ptls;
        maybe_collect(ptls);
        if (sz < old)
            jl_atomic_store_relaxed(&ptls->gc_num.freed,
                                    jl_atomic_load_relaxed(&ptls->gc_num.freed) +
                                        (old - sz));
        else
            jl_atomic_store_relaxed(&ptls->gc_num.allocd,
                                    jl_atomic_load_relaxed(&ptls->gc_num.allocd) +
                                        (sz - old));
        jl_atomic_store_relaxed(&ptls->gc_num.realloc,
                                jl_atomic_load_relaxed(&ptls->gc_num.realloc) + 1);
    }
    return realloc(p, sz);
}

JL_DLLEXPORT void *jl_gc_alloc_typed(jl_ptls_t ptls, size_t sz, void *ty)
{
    return jl_gc_alloc(ptls, sz, ty);
}

void *gc_managed_realloc_(jl_ptls_t ptls, void *d, size_t sz, size_t oldsz, int isaligned,
                          jl_value_t *owner, int8_t can_collect)
{
    if (can_collect)
        maybe_collect(ptls);

    size_t allocsz = LLT_ALIGN(sz, JL_CACHE_BYTE_ALIGNMENT);
    if (allocsz < sz) // overflow in adding offs, size was "negative"
        jl_throw(jl_memory_exception);

    if (jl_astaggedvalue(owner)->bits.gc == GC_OLD_MARKED) {
        ptls->gc_cache.perm_scanned_bytes += allocsz - oldsz;
        live_bytes += allocsz - oldsz;
    }
    else if (allocsz < oldsz)
        jl_atomic_store_relaxed(&ptls->gc_num.freed,
                                jl_atomic_load_relaxed(&ptls->gc_num.freed) +
                                    (oldsz - allocsz));
    else
        jl_atomic_store_relaxed(&ptls->gc_num.allocd,
                                jl_atomic_load_relaxed(&ptls->gc_num.allocd) +
                                    (allocsz - oldsz));
    jl_atomic_store_relaxed(&ptls->gc_num.realloc,
                            jl_atomic_load_relaxed(&ptls->gc_num.realloc) + 1);

    int last_errno = errno;
#ifdef _OS_WINDOWS_
    DWORD last_error = GetLastError();
#endif
    void *b;
    if (isaligned)
        b = realloc_cache_align(d, allocsz, oldsz);
    else
        b = realloc(d, allocsz);
    if (b == NULL)
        jl_throw(jl_memory_exception);
#ifdef _OS_WINDOWS_
    SetLastError(last_error);
#endif
    errno = last_errno;
    maybe_record_alloc_to_profile((jl_value_t *)b, sz, jl_gc_unknown_type_tag);
    return b;
}

JL_DLLEXPORT void *jl_gc_managed_realloc(void *d, size_t sz, size_t oldsz, int isaligned,
                                         jl_value_t *owner)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return gc_managed_realloc_(ptls, d, sz, oldsz, isaligned, owner, 1);
}

jl_value_t *jl_gc_realloc_string(jl_value_t *s, size_t sz)
{
    size_t len = jl_string_len(s);
    if (sz <= len)
        return s;
    jl_taggedvalue_t *v = jl_astaggedvalue(s);
    size_t strsz = len + sizeof(size_t) + 1;
    if (strsz <= GC_MAX_SZCLASS ||
        // TODO: because of issue #17971 we can't resize old objects
        gc_marked(v->bits.gc)) {
        // pool allocated; can't be grown in place so allocate a new object.
        jl_value_t *snew = jl_alloc_string(sz);
        memcpy(jl_string_data(snew), jl_string_data(s), len);
        return snew;
    }
    size_t newsz = sz + sizeof(size_t) + 1;
    size_t offs = sizeof(bigval_t);
    size_t oldsz = LLT_ALIGN(strsz + offs, JL_CACHE_BYTE_ALIGNMENT);
    size_t allocsz = LLT_ALIGN(newsz + offs, JL_CACHE_BYTE_ALIGNMENT);
    if (allocsz < sz) // overflow in adding offs, size was "negative"
        jl_throw(jl_memory_exception);
    bigval_t *hdr = bigval_header(v);
    jl_ptls_t ptls = jl_current_task->ptls;
    maybe_collect(ptls); // don't want this to happen during jl_gc_managed_realloc
    gc_big_object_unlink(hdr);
    // TODO: this is not safe since it frees the old pointer. ideally we'd like
    // the old pointer to be left alone if we can't grow in place.
    // for now it's up to the caller to make sure there are no references to the
    // old pointer.
    bigval_t *newbig = (bigval_t *)gc_managed_realloc_(ptls, hdr, allocsz, oldsz, 1, s, 0);
    newbig->sz = allocsz;
    // Big objects are allocated as old
    newbig->age = PROMOTE_AGE;
    v->bits.gc = GC_OLD;
    gc_big_object_link(newbig, &ptls->heap.big_objects);
    jl_value_t *snew = jl_valueof(&newbig->header);
    *(size_t *)snew = sz;
    return snew;
}

// Size includes the tag and the tag is not cleared!!
STATIC_INLINE jl_value_t *jl_gc_big_alloc_inner(jl_ptls_t ptls, size_t sz)
{
    maybe_collect(ptls);
    size_t offs = offsetof(bigval_t, header);
    assert(sz >= sizeof(jl_taggedvalue_t) && "sz must include tag");
    static_assert(offsetof(bigval_t, header) >= sizeof(void *), "Empty bigval header?");
    static_assert(sizeof(bigval_t) % JL_HEAP_ALIGNMENT == 0, "");
    size_t allocsz = LLT_ALIGN(sz + offs, JL_CACHE_BYTE_ALIGNMENT);
    if (allocsz < sz) // overflow in adding offs, size was "negative"
        jl_throw(jl_memory_exception);
    bigval_t *v = (bigval_t *)malloc_cache_align(allocsz);
    if (v == NULL)
        jl_throw(jl_memory_exception);
    gc_invoke_callbacks(jl_gc_cb_notify_external_alloc_t, gc_cblist_notify_external_alloc,
                        (v, allocsz));
    jl_atomic_store_relaxed(&ptls->gc_num.allocd,
                            jl_atomic_load_relaxed(&ptls->gc_num.allocd) + allocsz);
    jl_atomic_store_relaxed(&ptls->gc_num.bigalloc,
                            jl_atomic_load_relaxed(&ptls->gc_num.bigalloc) + 1);
#ifdef MEMDEBUG
    memset(v, 0xee, allocsz);
#endif
    v->sz = allocsz;
    // Big objects are allocated as old
    v->age = PROMOTE_AGE;
    v->bits.gc = GC_OLD;
    gc_big_object_link(v, &ptls->heap.big_objects);
    return jl_valueof(&v->header);
}

// Instrumented version of jl_gc_big_alloc_inner, called into by LLVM-generated code.
JL_DLLEXPORT jl_value_t *jl_gc_big_alloc(jl_ptls_t ptls, size_t sz)
{
    jl_value_t *val = jl_gc_big_alloc_inner(ptls, sz);
    maybe_record_alloc_to_profile(val, sz, jl_gc_unknown_type_tag);
    return val;
}

// This wrapper exists only to prevent `jl_gc_big_alloc_inner` from being inlined into
// its callers. We provide an external-facing interface for callers, and inline
// `jl_gc_big_alloc_inner` into this. (See https://github.com/JuliaLang/julia/pull/43868 for
// more details.)
jl_value_t *jl_gc_big_alloc_noinline(jl_ptls_t ptls, size_t sz)
{
    return jl_gc_big_alloc_inner(ptls, sz);
}

void jl_gc_set_permalloc_region(void *start, void *end)
{
    sysimg_base = start;
    sysimg_end = end;
}

void *gc_perm_alloc_large(size_t sz, int zero, unsigned align,
                          unsigned offset) JL_NOTSAFEPOINT
{
    // `align` must be power of two
    assert(offset == 0 || offset < align);
    const size_t malloc_align = sizeof(void *) == 8 ? 16 : 4;
    if (align > 1 && (offset != 0 || align > malloc_align))
        sz += align - 1;
    int last_errno = errno;
#ifdef _OS_WINDOWS_
    DWORD last_error = GetLastError();
#endif
    uintptr_t base = (uintptr_t)(zero ? calloc(1, sz) : malloc(sz));
    if (base == 0)
        jl_throw(jl_memory_exception);
#ifdef _OS_WINDOWS_
    SetLastError(last_error);
#endif
    errno = last_errno;
    jl_may_leak(base);
    assert(align > 0);
    unsigned diff = (offset - base) % align;
    return (void *)(base + diff);
}

// **NOT** a safepoint
void *jl_gc_perm_alloc_nolock(size_t sz, int zero, unsigned align, unsigned offset)
{
    // The caller should have acquired `gc_perm_lock`
    assert(align < GC_PERM_POOL_LIMIT);
#ifndef MEMDEBUG
    if (__unlikely(sz > GC_PERM_POOL_LIMIT))
#endif
        return gc_perm_alloc_large(sz, zero, align, offset);
    void *ptr = gc_try_perm_alloc_pool(sz, align, offset);
    if (__likely(ptr))
        return ptr;
    int last_errno = errno;
#ifdef _OS_WINDOWS_
    DWORD last_error = GetLastError();
    void *pool = VirtualAlloc(NULL, GC_PERM_POOL_SIZE, MEM_COMMIT, PAGE_READWRITE);
    SetLastError(last_error);
    errno = last_errno;
    if (__unlikely(pool == NULL))
        return NULL;
#else
    void *pool = mmap(0, GC_PERM_POOL_SIZE, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    errno = last_errno;
    if (__unlikely(pool == MAP_FAILED))
        return NULL;
#endif
    gc_perm_pool = (uintptr_t)pool;
    gc_perm_end = gc_perm_pool + GC_PERM_POOL_SIZE;
    return gc_try_perm_alloc_pool(sz, align, offset);
}

// **NOT** a safepoint
void *jl_gc_perm_alloc(size_t sz, int zero, unsigned align, unsigned offset)
{
    assert(align < GC_PERM_POOL_LIMIT);
#ifndef MEMDEBUG
    if (__unlikely(sz > GC_PERM_POOL_LIMIT))
#endif
        return gc_perm_alloc_large(sz, zero, align, offset);
    uv_mutex_lock(&gc_perm_lock);
    void *p = jl_gc_perm_alloc_nolock(sz, zero, align, offset);
    uv_mutex_unlock(&gc_perm_lock);
    return p;
}

JL_DLLEXPORT jl_value_t *jl_gc_allocobj(size_t sz)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return jl_gc_alloc(ptls, sz, NULL);
}

JL_DLLEXPORT jl_value_t *jl_gc_alloc_0w(void)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return jl_gc_alloc(ptls, 0, NULL);
}

JL_DLLEXPORT jl_value_t *jl_gc_alloc_1w(void)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return jl_gc_alloc(ptls, sizeof(void *), NULL);
}

JL_DLLEXPORT jl_value_t *jl_gc_alloc_2w(void)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return jl_gc_alloc(ptls, sizeof(void *) * 2, NULL);
}

JL_DLLEXPORT jl_value_t *jl_gc_alloc_3w(void)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return jl_gc_alloc(ptls, sizeof(void *) * 3, NULL);
}

JL_DLLEXPORT jl_value_t *(jl_gc_alloc)(jl_ptls_t ptls, size_t sz, void *ty)
{
    return jl_gc_alloc_(ptls, sz, ty);
}

// pool allocation
jl_taggedvalue_t *gc_reset_page(const jl_gc_pool_t *p, jl_gc_pagemeta_t *pg,
                                jl_taggedvalue_t *fl) JL_NOTSAFEPOINT
{
    assert(GC_PAGE_OFFSET >= sizeof(void *));
    pg->nfree = (GC_PAGE_SZ - GC_PAGE_OFFSET) / p->osize;
    jl_ptls_t ptls2 = jl_all_tls_states[pg->thread_n];
    pg->pool_n = p - ptls2->heap.norm_pools;
    memset(pg->ages, 0, GC_PAGE_SZ / 8 / p->osize + 1);
    jl_taggedvalue_t *beg = (jl_taggedvalue_t *)(pg->data + GC_PAGE_OFFSET);
    jl_taggedvalue_t *next = (jl_taggedvalue_t *)pg->data;
    if (fl == NULL) {
        next->next = NULL;
    }
    else {
        // Insert free page after first page.
        // This prevents unnecessary fragmentation from multiple pages
        // being allocated from at the same time. Instead, objects will
        // only ever be allocated from the first object in the list.
        // This is specifically being relied on by the implementation
        // of jl_gc_internal_obj_base_ptr() so that the function does
        // not have to traverse the entire list.
        jl_taggedvalue_t *flpage = (jl_taggedvalue_t *)gc_page_data(fl);
        next->next = flpage->next;
        flpage->next = beg;
        beg = fl;
    }
    pg->has_young = 0;
    pg->has_marked = 0;
    pg->fl_begin_offset = -1;
    pg->fl_end_offset = -1;
    return beg;
}

// Add a new page to the pool. Discards any pages in `p->newpages` before.
NOINLINE jl_taggedvalue_t *gc_add_page(jl_gc_pool_t *p) JL_NOTSAFEPOINT
{
    // Do not pass in `ptls` as argument. This slows down the fast path
    // in pool_alloc significantly
    jl_ptls_t ptls = jl_current_task->ptls;
    jl_gc_pagemeta_t *pg = jl_gc_alloc_page();
    pg->osize = p->osize;
    pg->ages = (uint8_t *)malloc_s(GC_PAGE_SZ / 8 / p->osize + 1);
    pg->thread_n = ptls->tid;
    jl_taggedvalue_t *fl = gc_reset_page(p, pg, NULL);
    p->newpages = fl;
    return fl;
}

// Instrumented version of jl_gc_pool_alloc_inner, called into by LLVM-generated code.
JL_DLLEXPORT jl_value_t *jl_gc_pool_alloc(jl_ptls_t ptls, int pool_offset, int osize)
{
    jl_value_t *val = jl_gc_pool_alloc_inner(ptls, pool_offset, osize);
    maybe_record_alloc_to_profile(val, osize, jl_gc_unknown_type_tag);
    return val;
}

// This wrapper exists only to prevent `jl_gc_pool_alloc_inner` from being inlined into
// its callers. We provide an external-facing interface for callers, and inline
// `jl_gc_pool_alloc_inner` into this. (See https://github.com/JuliaLang/julia/pull/43868
// for more details.)
jl_value_t *jl_gc_pool_alloc_noinline(jl_ptls_t ptls, int pool_offset, int osize)
{
    return jl_gc_pool_alloc_inner(ptls, pool_offset, osize);
}

int jl_gc_classify_pools(size_t sz, int *osize)
{
    if (sz > GC_MAX_SZCLASS)
        return -1;
    size_t allocsz = sz + sizeof(jl_taggedvalue_t);
    int klass = jl_gc_szclass(allocsz);
    *osize = jl_gc_sizeclasses[klass];
    return (int)(intptr_t)(&((jl_ptls_t)0)->heap.norm_pools[klass]);
}


#ifdef __cplusplus
}
#endif