// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef JL_GC_ALLOC_H
#define JL_GC_ALLOC_H

#include "gc.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 Allocation and malloc wrappers
*/

#if defined(_OS_WINDOWS_)
STATIC_INLINE void *jl_malloc_aligned(size_t sz, size_t align)
{
    return _aligned_malloc(sz ? sz : 1, align);
}
STATIC_INLINE void *jl_realloc_aligned(void *p, size_t sz, size_t oldsz, size_t align)
{
    (void)oldsz;
    return _aligned_realloc(p, sz ? sz : 1, align);
}
void jl_free_aligned(void *p) JL_NOTSAFEPOINT
{
    _aligned_free(p);
}
#else
STATIC_INLINE void *jl_malloc_aligned(size_t sz, size_t align)
{
#if defined(_P64) || defined(__APPLE__)
    if (align <= 16)
        return malloc(sz);
#endif
    void *ptr;
    if (posix_memalign(&ptr, align, sz))
        return NULL;
    return ptr;
}
STATIC_INLINE void *jl_realloc_aligned(void *d, size_t sz, size_t oldsz, size_t align)
{
#if defined(_P64) || defined(__APPLE__)
    if (align <= 16)
        return realloc(d, sz);
#endif
    void *b = jl_malloc_aligned(sz, align);
    if (b) {
        memcpy(b, d, oldsz > sz ? sz : oldsz);
        free(d);
    }
    return b;
}
STATIC_INLINE void jl_free_aligned(void *p) JL_NOTSAFEPOINT
{
    free(p);
}
#endif
#define malloc_cache_align(sz) jl_malloc_aligned(sz, JL_CACHE_BYTE_ALIGNMENT)
#define realloc_cache_align(p, sz, oldsz) \
    jl_realloc_aligned(p, sz, oldsz, JL_CACHE_BYTE_ALIGNMENT)

// Allocation wrappers that track allocation and let collection run
JL_DLLEXPORT void *jl_gc_counted_malloc(size_t sz);
JL_DLLEXPORT void *jl_gc_counted_calloc(size_t nm, size_t sz);
JL_DLLEXPORT void jl_gc_counted_free_with_size(void *p, size_t sz);
JL_DLLEXPORT void *jl_gc_counted_realloc_with_old_size(void *p, size_t old, size_t sz);

// Allocation wrappers that save the size of allocations, to allow using
// jl_gc_counted_* functions with a libc-compatible API.
STATIC_INLINE void *_unchecked_calloc(size_t nm, size_t sz)
{
    size_t nmsz = nm * sz;
    int64_t *p = (int64_t *)jl_gc_counted_calloc(nmsz + JL_SMALL_BYTE_ALIGNMENT, 1);
    if (p == NULL)
        return NULL;
    p[0] = nmsz;
    return (void *)(p + 2); // assumes JL_SMALL_BYTE_ALIGNMENT == 16
}
JL_DLLEXPORT void *jl_malloc(size_t sz);
JL_DLLEXPORT void *jl_calloc(size_t nm, size_t sz);
JL_DLLEXPORT void jl_free(void *p);
JL_DLLEXPORT void *jl_realloc(void *p, size_t sz);
JL_DLLEXPORT void *jl_gc_managed_malloc(size_t sz);

STATIC_INLINE void maybe_collect(jl_ptls_t ptls)
{
    if (jl_atomic_load_relaxed(&ptls->gc_num.allocd) >= 0 || jl_gc_debug_check_other()) {
        jl_gc_collect(JL_GC_AUTO);
    }
    else {
        jl_gc_safepoint_(ptls);
    }
}
void *gc_managed_realloc_(jl_ptls_t ptls, void *d, size_t sz, size_t oldsz, int isaligned,
                          jl_value_t *owner, int8_t can_collect);
jl_value_t *jl_gc_realloc_string(jl_value_t *s, size_t sz);
JL_DLLEXPORT void *jl_gc_alloc_typed(jl_ptls_t ptls, size_t sz, void *ty);
JL_DLLEXPORT void *jl_gc_managed_realloc(void *d, size_t sz, size_t oldsz, int isaligned,
                                         jl_value_t *owner);

// Big value list

// Size includes the tag and the tag is not cleared!!
STATIC_INLINE jl_value_t *jl_gc_big_alloc_inner(jl_ptls_t ptls, size_t sz);
JL_DLLEXPORT jl_value_t *jl_gc_big_alloc(jl_ptls_t ptls, size_t sz);
jl_value_t *jl_gc_big_alloc_noinline(jl_ptls_t ptls, size_t sz);

// Perm gen allocator
// 2M pool
#define GC_PERM_POOL_SIZE (2 * 1024 * 1024)
// 20k limit for pool allocation. At most 1% fragmentation
#define GC_PERM_POOL_LIMIT (20 * 1024)
extern uv_mutex_t gc_perm_lock;
extern uintptr_t gc_perm_pool;
extern uintptr_t gc_perm_end;

STATIC_INLINE void *gc_try_perm_alloc_pool(size_t sz, unsigned align,
                                           unsigned offset) JL_NOTSAFEPOINT
{
    uintptr_t pool = LLT_ALIGN(gc_perm_pool + offset, (uintptr_t)align) - offset;
    uintptr_t end = pool + sz;
    if (end > gc_perm_end)
        return NULL;
    gc_perm_pool = end;
    return (void *)jl_assume(pool);
}
void *gc_perm_alloc_large(size_t sz, int zero, unsigned align,
                          unsigned offset) JL_NOTSAFEPOINT;
void *jl_gc_perm_alloc_nolock(size_t sz, int zero, unsigned align, unsigned offset);
void *jl_gc_perm_alloc(size_t sz, int zero, unsigned align, unsigned offset);

JL_DLLEXPORT jl_value_t *jl_gc_allocobj(size_t sz);
JL_DLLEXPORT jl_value_t *jl_gc_alloc_0w(void);
JL_DLLEXPORT jl_value_t *jl_gc_alloc_1w(void);
JL_DLLEXPORT jl_value_t *jl_gc_alloc_2w(void);
JL_DLLEXPORT jl_value_t *jl_gc_alloc_3w(void);
JL_DLLEXPORT jl_value_t *(jl_gc_alloc)(jl_ptls_t ptls, size_t sz, void *ty);
#ifdef __cplusplus
}
#endif

#endif