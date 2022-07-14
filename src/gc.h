// This file is a part of Julia. License is MIT: https://julialang.org/license

/*
  allocation and garbage collection
  . non-moving, precise mark and sweep collector
  . pool-allocates small objects, keeps big objects on a simple list
*/

#ifndef JL_GC_H
#define JL_GC_H

#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <inttypes.h>
#include "julia.h"
#include "julia_threads.h"
#include "julia_internal.h"
#include "threading.h"
#ifndef _OS_WINDOWS_
#include <sys/mman.h>
#if defined(_OS_DARWIN_) && !defined(MAP_ANONYMOUS)
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif
#include "julia_assert.h"
#include "gc-alloc-profiler.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GC_PAGE_LG2 14 // log2(size of a page)
#define GC_PAGE_SZ (1 << GC_PAGE_LG2) // 16k
#define GC_PAGE_OFFSET (JL_HEAP_ALIGNMENT - (sizeof(jl_taggedvalue_t) % JL_HEAP_ALIGNMENT))

#define jl_malloc_tag ((void*)0xdeadaa01)
#define jl_singleton_tag ((void*)0xdeadaa02)

// Used by GC_DEBUG_ENV
typedef struct {
    uint64_t num;
    uint64_t next;
    uint64_t min;
    uint64_t interv;
    uint64_t max;
    unsigned short random[3];
} jl_alloc_num_t;

typedef struct {
    int always_full;
    int wait_for_debugger;
    jl_alloc_num_t pool;
    jl_alloc_num_t other;
    jl_alloc_num_t print;
} jl_gc_debug_env_t;

// This struct must be kept in sync with the Julia type of the same name in base/timing.jl
typedef struct {
    int64_t     allocd;
    int64_t     deferred_alloc;
    int64_t     freed;
    uint64_t    malloc;
    uint64_t    realloc;
    uint64_t    poolalloc;
    uint64_t    bigalloc;
    uint64_t    freecall;
    uint64_t    total_time;
    uint64_t    total_allocd;
    uint64_t    since_sweep;
    size_t      interval;
    int         pause;
    int         full_sweep;
    uint64_t    max_pause;
    uint64_t    max_memory;
    uint64_t    time_to_safepoint;
    uint64_t    max_time_to_safepoint;
    uint64_t    sweep_time;
    uint64_t    mark_time;
    uint64_t    total_sweep_time;
    uint64_t    total_mark_time;
} jl_gc_num_t;

// Double the mark queue
static NOINLINE void gc_markqueue_resize(jl_gc_markqueue_t *mq) JL_NOTSAFEPOINT
{
#if defined(PREFETCH_MARK)
    jl_gc_markstack_t *ms = &mq->mark_stack;
    jl_value_t **old_start = ms->start;
    size_t old_queue_size = (ms->end - ms->start);
    size_t offset = (ms->current - old_start);
    ms->start = (jl_value_t**)realloc_s(old_start, 2 * old_queue_size * sizeof(jl_value_t*));
    ms->current = (ms->start + offset);
    ms->end = (ms->start + 2 * old_queue_size);
#elif defined(DFS_MARK)
    jl_value_t **old_start = mq->start;
    size_t old_queue_size = (mq->end - mq->start);
    size_t offset = (mq->current - old_start);
    mq->start =
        (jl_value_t **)realloc_s(old_start, 2 * old_queue_size * sizeof(jl_value_t *));
    mq->current = (mq->start + offset);
    mq->end = (mq->start + 2 * old_queue_size);
#else
    jl_value_t **old_start = mq->start;
    size_t old_capacity = mq->capacity;
    mq->start = (jl_value_t **)malloc(2 * old_capacity * sizeof(jl_value_t *));
    // Copy elements into new buffer
    for (size_t i = mq->top; i < mq->bottom; i++)
        mq->start[i % (2 * old_capacity)] = old_start[i % old_capacity];
    free(old_start);
    mq->capacity = 2 * old_capacity;
#endif
}

#define PF_MIN (1 << 6)
#define PF_SIZE (1 << 8)

// Push a work item to the queue
STATIC_INLINE void gc_markqueue_push(jl_gc_markqueue_t *mq,
                                     jl_value_t *obj) JL_NOTSAFEPOINT
{
#if defined(PREFETCH_MARK)
    jl_gc_prefetch_buf_t *pf_buf = &mq->prefetch_buf;
    jl_gc_markstack_t *ms = &mq->mark_stack;
    // Prefetch buffer overflowed: push to mark-stack
    if (__likely(pf_buf->bottom - pf_buf->top >= pf_buf->size)) {
        // Mark-stack overflowed: resize it
        if (__unlikely(ms->current == ms->end))
            gc_markqueue_resize(mq);
        *ms->current = obj;
        ms->current++;
    }
    else {
        // There is still space in the FIFO buffer: push it there
        pf_buf->start[pf_buf->bottom % pf_buf->size] = obj;
        pf_buf->bottom++;
    }
#elif defined(DFS_MARK)
    if (__unlikely(mq->current == mq->end))
        gc_markqueue_resize(mq);
    *mq->current = obj;
    mq->current++;
#else
    // Mark-stack overflowed: resize it
    if (__unlikely(mq->bottom - mq->top >= mq->capacity))
        gc_markqueue_resize(mq);
    mq->start[mq->bottom % mq->capacity] = obj;
    mq->bottom++;
#endif
}

#if defined(_COMPILER_GCC_) && defined(_CPU_X86_)
#define jl_prefetch(p) __builtin_prefetch((p), 1, 3)
#else
#define jl_prefetch(p)
#endif

// Pop from the mark queue
STATIC_INLINE jl_value_t *gc_markqueue_pop(jl_gc_markqueue_t *mq)
{
#if defined(PREFETCH_MARK)
    jl_gc_prefetch_buf_t *pf_buf = &mq->prefetch_buf;
    jl_gc_markstack_t *ms = &mq->mark_stack;
    jl_value_t *obj = NULL;
    // FIFO buffer is nearly empty and there is element in mark-stack: pop
    // element from stack
    if (pf_buf->bottom - pf_buf->top <= PF_MIN && ms->current != ms->start) {
        ms->current--;
        obj = *ms->current;
    }
    // enough elements in FIFO buffer
    else if (pf_buf->bottom - pf_buf->top > 0) {
        // take element from FIFO buffer
        obj = pf_buf->start[pf_buf->top % pf_buf->size];
        pf_buf->top++;
        // if there is any element in the stack, pop it, prefetch and insert into FIFO buffer
        if (ms->current != ms->start) {
            ms->current--;
            jl_value_t *to_prefetch = *ms->current;
            jl_prefetch(to_prefetch);
            pf_buf->start[pf_buf->bottom % pf_buf->size] = to_prefetch;
            pf_buf->bottom++;
        }
    }
    return obj;
#elif defined(DFS_MARK)
    if (mq->current == mq->start)
        return NULL;
    mq->current--;
    jl_value_t *obj = *mq->current;
    return obj;
#else
    if (mq->bottom == mq->top)
        return NULL;
    jl_value_t *obj = mq->start[mq->top % mq->capacity];
    mq->top++;
    return obj;
#endif
}

// layout for big (>2k) objects

JL_EXTENSION typedef struct _bigval_t {
    struct _bigval_t *next;
    struct _bigval_t **prev; // pointer to the next field of the prev entry
    union {
        size_t sz;
        uintptr_t age : 2;
    };
#ifdef _P64 // Add padding so that the value is 64-byte aligned
    // (8 pointers of 8 bytes each) - (4 other pointers in struct)
    void *_padding[8 - 4];
#else
    // (16 pointers of 4 bytes each) - (4 other pointers in struct)
    void *_padding[16 - 4];
#endif
    //struct jl_taggedvalue_t <>;
    union {
        uintptr_t header;
        struct {
            uintptr_t gc:2;
        } bits;
    };
    // must be 64-byte aligned here, in 32 & 64 bit modes
} bigval_t;

// data structure for tracking malloc'd arrays.

typedef struct _mallocarray_t {
    jl_array_t *a;
    struct _mallocarray_t *next;
} mallocarray_t;

// pool page metadata
typedef struct {
    // index of pool that owns this page
    uint8_t pool_n;
    // Whether any cell in the page is marked
    // This bit is set before sweeping iff there are live cells in the page.
    // Note that before marking or after sweeping there can be live
    // (and young) cells in the page for `!has_marked`.
    uint8_t has_marked;
    // Whether any cell was live and young **before sweeping**.
    // For a normal sweep (quick sweep that is NOT preceded by a
    // full sweep) this bit is set iff there are young or newly dead
    // objects in the page and the page needs to be swept.
    //
    // For a full sweep, this bit should be ignored.
    //
    // For a quick sweep preceded by a full sweep. If this bit is set,
    // the page needs to be swept. If this bit is not set, there could
    // still be old dead objects in the page and `nold` and `prev_nold`
    // should be used to determine if the page needs to be swept.
    uint8_t has_young;
    // number of old objects in this page
    uint16_t nold;
    // number of old objects in this page during the previous full sweep
    uint16_t prev_nold;
    // number of free objects in this page.
    // invalid if pool that owns this page is allocating objects from this page.
    uint16_t nfree;
    uint16_t osize; // size of each object in this page
    uint16_t fl_begin_offset; // offset of first free object in this page
    uint16_t fl_end_offset;   // offset of last free object in this page
    uint16_t thread_n;        // thread id of the heap that owns this page
    char *data;
    uint8_t *ages;
} jl_gc_pagemeta_t;

// Page layout:
//  Newpage freelist: sizeof(void*)
//  Padding: GC_PAGE_OFFSET - sizeof(void*)
//  Blocks: osize * n
//    Tag: sizeof(jl_taggedvalue_t)
//    Data: <= osize - sizeof(jl_taggedvalue_t)

// Memory map:
//  The complete address space is divided up into a multi-level page table.
//  The three levels have similar but slightly different structures:
//    - pagetable0_t: the bottom/leaf level (covers the contiguous addresses)
//    - pagetable1_t: the middle level
//    - pagetable2_t: the top/leaf level (covers the entire virtual address space)
//  Corresponding to these similar structures is a large amount of repetitive
//  code that is nearly the same but not identical. It could be made less
//  repetitive with C macros, but only at the cost of debuggability. The specialized
//  structure of this representation allows us to partially unroll and optimize
//  various conditions at each level.

//  The following constants define the branching factors at each level.
//  The constants and GC_PAGE_LG2 must therefore sum to sizeof(void*).
//  They should all be multiples of 32 (sizeof(uint32_t)) except that REGION2_PG_COUNT may also be 1.
#ifdef _P64
#define REGION0_PG_COUNT (1 << 16)
#define REGION1_PG_COUNT (1 << 16)
#define REGION2_PG_COUNT (1 << 18)
#define REGION0_INDEX(p) (((uintptr_t)(p) >> 14) & 0xFFFF) // shift by GC_PAGE_LG2
#define REGION1_INDEX(p) (((uintptr_t)(p) >> 30) & 0xFFFF)
#define REGION_INDEX(p)  (((uintptr_t)(p) >> 46) & 0x3FFFF)
#else
#define REGION0_PG_COUNT (1 << 8)
#define REGION1_PG_COUNT (1 << 10)
#define REGION2_PG_COUNT (1 << 0)
#define REGION0_INDEX(p) (((uintptr_t)(p) >> 14) & 0xFF) // shift by GC_PAGE_LG2
#define REGION1_INDEX(p) (((uintptr_t)(p) >> 22) & 0x3FF)
#define REGION_INDEX(p)  (0)
#endif

// define the representation of the levels of the page-table (0 to 2)
typedef struct {
    jl_gc_pagemeta_t *meta[REGION0_PG_COUNT];
    uint32_t allocmap[REGION0_PG_COUNT / 32];
    uint32_t freemap[REGION0_PG_COUNT / 32];
    // store a lower bound of the first free page in each region
    int lb;
    // an upper bound of the last non-free page
    int ub;
} pagetable0_t;

typedef struct {
    pagetable0_t *meta0[REGION1_PG_COUNT];
    uint32_t allocmap0[REGION1_PG_COUNT / 32];
    uint32_t freemap0[REGION1_PG_COUNT / 32];
    // store a lower bound of the first free page in each region
    int lb;
    // an upper bound of the last non-free page
    int ub;
} pagetable1_t;

typedef struct {
    pagetable1_t *meta1[REGION2_PG_COUNT];
    uint32_t allocmap1[(REGION2_PG_COUNT + 31) / 32];
    uint32_t freemap1[(REGION2_PG_COUNT + 31) / 32];
    // store a lower bound of the first free page in each region
    int lb;
    // an upper bound of the last non-free page
    int ub;
} pagetable_t;

#ifdef __clang_gcanalyzer__
unsigned ffs_u32(uint32_t bitvec) JL_NOTSAFEPOINT;
#else
STATIC_INLINE unsigned ffs_u32(uint32_t bitvec)
{
    return __builtin_ffs(bitvec) - 1;
}
#endif

extern jl_gc_num_t gc_num;
extern pagetable_t memory_map;
extern bigval_t *big_objects_marked;
extern arraylist_t finalizer_list_marked;
extern arraylist_t to_finalize;
extern int64_t lazy_freed_pages;

STATIC_INLINE bigval_t *bigval_header(jl_taggedvalue_t *o) JL_NOTSAFEPOINT
{
    return container_of(o, bigval_t, header);
}

// round an address inside a gcpage's data to its beginning
STATIC_INLINE char *gc_page_data(void *x) JL_NOTSAFEPOINT
{
    return (char*)(((uintptr_t)x >> GC_PAGE_LG2) << GC_PAGE_LG2);
}

STATIC_INLINE jl_taggedvalue_t *page_pfl_beg(jl_gc_pagemeta_t *p) JL_NOTSAFEPOINT
{
    return (jl_taggedvalue_t*)(p->data + p->fl_begin_offset);
}

STATIC_INLINE jl_taggedvalue_t *page_pfl_end(jl_gc_pagemeta_t *p) JL_NOTSAFEPOINT
{
    return (jl_taggedvalue_t*)(p->data + p->fl_end_offset);
}

STATIC_INLINE int gc_marked(uintptr_t bits) JL_NOTSAFEPOINT
{
    return (bits & GC_MARKED) != 0;
}

STATIC_INLINE int gc_old(uintptr_t bits) JL_NOTSAFEPOINT
{
    return (bits & GC_OLD) != 0;
}

STATIC_INLINE uintptr_t gc_set_bits(uintptr_t tag, int bits) JL_NOTSAFEPOINT
{
    return (tag & ~(uintptr_t)3) | bits;
}

STATIC_INLINE uintptr_t gc_ptr_tag(void *v, uintptr_t mask) JL_NOTSAFEPOINT
{
    return ((uintptr_t)v) & mask;
}

STATIC_INLINE void *gc_ptr_clear_tag(void *v, uintptr_t mask) JL_NOTSAFEPOINT
{
    return (void*)(((uintptr_t)v) & ~mask);
}

NOINLINE uintptr_t gc_get_stack_ptr(void);

STATIC_INLINE jl_gc_pagemeta_t *page_metadata(void *_data) JL_NOTSAFEPOINT
{
    uintptr_t data = ((uintptr_t)_data);
    unsigned i;
    i = REGION_INDEX(data);
    pagetable1_t *r1 = memory_map.meta1[i];
    if (!r1)
        return NULL;
    i = REGION1_INDEX(data);
    pagetable0_t *r0 = r1->meta0[i];
    if (!r0)
        return NULL;
    i = REGION0_INDEX(data);
    return r0->meta[i];
}

struct jl_gc_metadata_ext {
    pagetable1_t *pagetable1;
    pagetable0_t *pagetable0;
    jl_gc_pagemeta_t *meta;
    unsigned pagetable_i32, pagetable_i;
    unsigned pagetable1_i32, pagetable1_i;
    unsigned pagetable0_i32, pagetable0_i;
};

STATIC_INLINE struct jl_gc_metadata_ext page_metadata_ext(void *_data) JL_NOTSAFEPOINT
{
    uintptr_t data = (uintptr_t)_data;
    struct jl_gc_metadata_ext info;
    unsigned i;
    i = REGION_INDEX(data);
    info.pagetable_i = i % 32;
    info.pagetable_i32 = i / 32;
    info.pagetable1 = memory_map.meta1[i];
    i = REGION1_INDEX(data);
    info.pagetable1_i = i % 32;
    info.pagetable1_i32 = i / 32;
    info.pagetable0 = info.pagetable1->meta0[i];
    i = REGION0_INDEX(data);
    info.pagetable0_i = i % 32;
    info.pagetable0_i32 = i / 32;
    info.meta = info.pagetable0->meta[i];
    assert(info.meta);
    return info;
}

STATIC_INLINE void gc_big_object_unlink(const bigval_t *hdr) JL_NOTSAFEPOINT
{
    *hdr->prev = hdr->next;
    if (hdr->next) {
        hdr->next->prev = hdr->prev;
    }
}

STATIC_INLINE void gc_big_object_link(bigval_t *hdr, bigval_t **list) JL_NOTSAFEPOINT
{
    hdr->next = *list;
    hdr->prev = list;
    if (*list)
        (*list)->prev = &hdr->next;
    *list = hdr;
}

void gc_mark_queue_all_roots(jl_ptls_t ptls, jl_gc_markqueue_t *mq);
void gc_mark_finlist(jl_gc_markqueue_t *mq, arraylist_t *list,
                     size_t start);
void _gc_mark_loop(jl_ptls_t ptls, jl_gc_markqueue_t *mq);
void gc_mark_loop(jl_ptls_t ptls);
void sweep_stack_pools(void);
void jl_gc_debug_init(void);

// GC pages

void jl_gc_init_page(void);
NOINLINE jl_gc_pagemeta_t *jl_gc_alloc_page(void) JL_NOTSAFEPOINT;
void jl_gc_free_page(void *p) JL_NOTSAFEPOINT;

// GC debug

#if defined(GC_TIME) || defined(GC_FINAL_STATS)
void gc_settime_premark_end(void);
void gc_settime_postmark_end(void);
#else
#define gc_settime_premark_end()
#define gc_settime_postmark_end()
#endif

#ifdef GC_FINAL_STATS
void gc_final_count_page(size_t pg_cnt);
void gc_final_pause_end(int64_t t0, int64_t tend);
#else
#define gc_final_count_page(pg_cnt)
#define gc_final_pause_end(t0, tend)
#endif

#ifdef GC_TIME
void gc_time_pool_start(void) JL_NOTSAFEPOINT;
void gc_time_count_page(int freedall, int pg_skpd) JL_NOTSAFEPOINT;
void gc_time_pool_end(int sweep_full) JL_NOTSAFEPOINT;
void gc_time_sysimg_end(uint64_t t0) JL_NOTSAFEPOINT;

void gc_time_big_start(void) JL_NOTSAFEPOINT;
void gc_time_count_big(int old_bits, int bits) JL_NOTSAFEPOINT;
void gc_time_big_end(void) JL_NOTSAFEPOINT;

void gc_time_mallocd_array_start(void) JL_NOTSAFEPOINT;
void gc_time_count_mallocd_array(int bits) JL_NOTSAFEPOINT;
void gc_time_mallocd_array_end(void) JL_NOTSAFEPOINT;

void gc_time_mark_pause(int64_t t0, int64_t scanned_bytes,
                        int64_t perm_scanned_bytes);
void gc_time_sweep_pause(uint64_t gc_end_t, int64_t actual_allocd,
                         int64_t live_bytes, int64_t estimate_freed,
                         int sweep_full);
void gc_time_summary(int sweep_full, uint64_t start, uint64_t end,
                     uint64_t freed, uint64_t live, uint64_t interval,
                     uint64_t pause, uint64_t ttsp, uint64_t mark,
                     uint64_t sweep);
#else
#define gc_time_pool_start()
STATIC_INLINE void gc_time_count_page(int freedall, int pg_skpd) JL_NOTSAFEPOINT
{
    (void)freedall;
    (void)pg_skpd;
}
#define gc_time_pool_end(sweep_full) (void)(sweep_full)
#define gc_time_sysimg_end(t0) (void)(t0)
#define gc_time_big_start()
STATIC_INLINE void gc_time_count_big(int old_bits, int bits) JL_NOTSAFEPOINT
{
    (void)old_bits;
    (void)bits;
}
#define gc_time_big_end()
#define gc_time_mallocd_array_start()
STATIC_INLINE void gc_time_count_mallocd_array(int bits) JL_NOTSAFEPOINT
{
    (void)bits;
}
#define gc_time_mallocd_array_end()
#define gc_time_mark_pause(t0, scanned_bytes, perm_scanned_bytes)
#define gc_time_sweep_pause(gc_end_t, actual_allocd, live_bytes,        \
                            estimate_freed, sweep_full)
#define  gc_time_summary(sweep_full, start, end, freed, live,           \
                         interval, pause, ttsp, mark, sweep)
#endif

#ifdef MEMFENCE
void gc_verify_tags(void);
#else
static inline void gc_verify_tags(void)
{
}
#endif

#ifdef GC_VERIFY
extern jl_value_t *lostval;
void gc_verify(jl_ptls_t ptls);
void add_lostval_parent(jl_value_t *parent);
#define verify_val(v) do {                                              \
        if (lostval == (jl_value_t*)(v) && (v) != 0) {                  \
            jl_printf(JL_STDOUT,                                        \
                      "Found lostval %p at %s:%d oftype: ",             \
                      (void*)(lostval), __FILE__, __LINE__);            \
            jl_static_show(JL_STDOUT, jl_typeof(v));                    \
            jl_printf(JL_STDOUT, "\n");                                 \
        }                                                               \
    } while(0);

#define verify_parent(ty, obj, slot, args...) do {                      \
        if (gc_ptr_clear_tag(*(void**)(slot), 3) == (void*)lostval &&   \
            (jl_value_t*)(obj) != lostval) {                            \
            jl_printf(JL_STDOUT, "Found parent %p %p at %s:%d\n",       \
                      (void*)(ty), (void*)(obj), __FILE__, __LINE__);   \
            jl_printf(JL_STDOUT, "\tloc %p : ", (void*)(slot));         \
            jl_printf(JL_STDOUT, args);                                 \
            jl_printf(JL_STDOUT, "\n");                                 \
            jl_printf(JL_STDOUT, "\ttype: ");                           \
            jl_static_show(JL_STDOUT, jl_typeof(obj));                  \
            jl_printf(JL_STDOUT, "\n");                                 \
            add_lostval_parent((jl_value_t*)(obj));                     \
        }                                                               \
    } while(0);

#define verify_parent1(ty,obj,slot,arg1) verify_parent(ty,obj,slot,arg1)
#define verify_parent2(ty,obj,slot,arg1,arg2) verify_parent(ty,obj,slot,arg1,arg2)
extern int gc_verifying;
#else
#define gc_verify(ptls)
#define verify_val(v)
#define verify_parent1(ty,obj,slot,arg1) do {} while (0)
#define verify_parent2(ty,obj,slot,arg1,arg2) do {} while (0)
#define gc_verifying (0)
#endif
int gc_slot_to_fieldidx(void *_obj, void *slot);
int gc_slot_to_arrayidx(void *_obj, void *begin);
NOINLINE void gc_mark_loop_unwind(jl_ptls_t ptls, jl_gc_markqueue_t *mq, int pc_offset);

#ifdef GC_DEBUG_ENV
JL_DLLEXPORT extern jl_gc_debug_env_t jl_gc_debug_env;
#define gc_sweep_always_full jl_gc_debug_env.always_full
int jl_gc_debug_check_other(void);
int gc_debug_check_pool(void);
void jl_gc_debug_print(void);
void gc_scrub_record_task(jl_task_t *ta) JL_NOTSAFEPOINT;
void gc_scrub(void);
#else
#define gc_sweep_always_full 0
static inline int jl_gc_debug_check_other(void)
{
    return 0;
}
static inline int gc_debug_check_pool(void)
{
    return 0;
}
static inline void jl_gc_debug_print(void)
{
}
static inline void gc_scrub_record_task(jl_task_t *ta) JL_NOTSAFEPOINT
{
    (void)ta;
}
static inline void gc_scrub(void)
{
}
#endif

#ifdef OBJPROFILE
void objprofile_count(void *ty, int old, int sz) JL_NOTSAFEPOINT;
void objprofile_printall(void);
void objprofile_reset(void);
#else
static inline void objprofile_count(void *ty, int old, int sz) JL_NOTSAFEPOINT
{
}

static inline void objprofile_printall(void)
{
}

static inline void objprofile_reset(void)
{
}
#endif

#ifdef MEMPROFILE
void gc_stats_all_pool(void);
void gc_stats_big_obj(void);
#else
#define gc_stats_all_pool()
#define gc_stats_big_obj()
#endif

// For debugging
void gc_count_pool(void);

size_t jl_array_nbytes(jl_array_t *a) JL_NOTSAFEPOINT;

JL_DLLEXPORT void jl_enable_gc_logging(int enable);
void _report_gc_finished(uint64_t pause, uint64_t freed, int full, int recollect) JL_NOTSAFEPOINT;

#ifdef __cplusplus
}
#endif

#endif
