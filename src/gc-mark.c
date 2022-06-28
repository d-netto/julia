// This file is a part of Julia. License is MIT: https://julialang.org/license

#include "gc-callbacks.h"
#include "gc.h"
#include "julia_assert.h"
#include "julia_gcext.h"

#ifdef __cplusplus
extern "C" {
#endif

// ====================
//
// Marking phase
//
// ====================

extern int mark_reset_age;
extern void *sysimg_base;
extern void *sysimg_end;
extern void gc_sync_cache(jl_ptls_t ptls) JL_NOTSAFEPOINT;

extern jl_gc_callback_list_t *gc_cblist_root_scanner;
extern jl_gc_callback_list_t *gc_cblist_task_scanner;
extern jl_gc_callback_list_t *gc_cblist_pre_gc;
extern jl_gc_callback_list_t *gc_cblist_post_gc;
extern jl_gc_callback_list_t *gc_cblist_notify_external_alloc;
extern jl_gc_callback_list_t *gc_cblist_notify_external_free;

JL_DLLEXPORT void jl_gc_queue_root(const jl_value_t *ptr)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    jl_taggedvalue_t *o = jl_astaggedvalue(ptr);
    // The modification of the `gc_bits` is not atomic but it
    // should be safe here since GC is not allowed to run here and we only
    // write GC_OLD to the GC bits outside GC. This could cause
    // duplicated objects in the remset but that shouldn't be a problem.
    o->bits.gc = GC_MARKED;
    arraylist_push(ptls->heap.remset, (jl_value_t *)ptr);
    ptls->heap.remset_nptr++; // conservative
}

void jl_gc_queue_multiroot(const jl_value_t *parent, const jl_value_t *ptr) JL_NOTSAFEPOINT
{
    // First check if this is really necessary
    // TODO: should we store this info in one of the extra gc bits?
    jl_datatype_t *dt = (jl_datatype_t *)jl_typeof(ptr);
    const jl_datatype_layout_t *ly = dt->layout;
    uint32_t npointers = ly->npointers;
    // if (npointers == 0) // this was checked by the caller
    //     return;
    jl_value_t *ptrf = ((jl_value_t **)ptr)[ly->first_ptr];
    if (ptrf && (jl_astaggedvalue(ptrf)->bits.gc & 1) == 0) {
        // this pointer was young, move the barrier back now
        jl_gc_wb_back(parent);
        return;
    }
    const uint8_t *ptrs8 = (const uint8_t *)jl_dt_layout_ptrs(ly);
    const uint16_t *ptrs16 = (const uint16_t *)jl_dt_layout_ptrs(ly);
    const uint32_t *ptrs32 = (const uint32_t *)jl_dt_layout_ptrs(ly);
    for (size_t i = 1; i < npointers; i++) {
        uint32_t fld;
        if (ly->fielddesc_type == 0) {
            fld = ptrs8[i];
        }
        else if (ly->fielddesc_type == 1) {
            fld = ptrs16[i];
        }
        else {
            assert(ly->fielddesc_type == 2);
            fld = ptrs32[i];
        }
        jl_value_t *ptrf = ((jl_value_t **)ptr)[fld];
        if (ptrf && (jl_astaggedvalue(ptrf)->bits.gc & 1) == 0) {
            // this pointer was young, move the barrier back now
            jl_gc_wb_back(parent);
            return;
        }
    }
}

JL_DLLEXPORT void jl_gc_queue_binding(jl_binding_t *bnd)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    jl_taggedvalue_t *buf = jl_astaggedvalue(bnd);
    buf->bits.gc = GC_MARKED;
    arraylist_push(&ptls->heap.rem_bindings, bnd);
}

STATIC_INLINE void gc_queue_big_marked(jl_ptls_t ptls, bigval_t *hdr,
                                       int toyoung) JL_NOTSAFEPOINT
{
    const int nentry = sizeof(ptls->gc_cache.big_obj) / sizeof(void *);
    size_t nobj = ptls->gc_cache.nbig_obj;
    if (__unlikely(nobj >= nentry)) {
        gc_sync_cache(ptls);
        nobj = 0;
    }
    uintptr_t v = (uintptr_t)hdr;
    ptls->gc_cache.big_obj[nobj] = (void *)(toyoung ? (v | 1) : v);
    ptls->gc_cache.nbig_obj = nobj + 1;
}

// Set the tag of an object and return whether it was marked
STATIC_INLINE int gc_try_setmark_tag(jl_taggedvalue_t *o, uint8_t mark_mode) JL_NOTSAFEPOINT
{
    assert(gc_marked(mark_mode));
    uintptr_t tag = o->header;
    if (gc_marked(tag))
        return 0;
    if (mark_reset_age) {
        // Reset the object as if it was just allocated
        mark_mode = GC_MARKED;
        tag = gc_set_bits(tag, mark_mode);
    }
    else {
        if (gc_old(tag))
            mark_mode = GC_OLD_MARKED;
        tag = tag | mark_mode;
        assert((tag & 0x3) == mark_mode);
    }
    tag = jl_atomic_exchange_relaxed((_Atomic(uintptr_t) *)&o->header, tag);
    verify_val(jl_valueof(o));
    return !gc_marked(tag);
}

// This function should be called exactly once during marking for each big
// object being marked to update the big objects metadata.
STATIC_INLINE void gc_setmark_big(jl_ptls_t ptls, jl_taggedvalue_t *o,
                                  uint8_t mark_mode) JL_NOTSAFEPOINT
{
    assert(!page_metadata(o));
    bigval_t *hdr = bigval_header(o);
    if (mark_mode == GC_OLD_MARKED) {
        ptls->gc_cache.perm_scanned_bytes += hdr->sz & ~3;
        gc_queue_big_marked(ptls, hdr, 0);
    }
    else {
        ptls->gc_cache.scanned_bytes += hdr->sz & ~3;
        // We can't easily tell if the object is old or being promoted
        // from the gc bits but if the `age` is `0` then the object
        // must be already on a young list.
        if (mark_reset_age && hdr->age) {
            // Reset the object as if it was just allocated
            hdr->age = 0;
            gc_queue_big_marked(ptls, hdr, 1);
        }
    }
    objprofile_count(jl_typeof(jl_valueof(o)), mark_mode == GC_OLD_MARKED, hdr->sz & ~3);
}

// This function should be called exactly once during marking for each pool
// object being marked to update the page metadata.
STATIC_INLINE void gc_setmark_pool_(jl_ptls_t ptls, jl_taggedvalue_t *o, uint8_t mark_mode,
                                    jl_gc_pagemeta_t *page) JL_NOTSAFEPOINT
{
#ifdef MEMDEBUG
    gc_setmark_big(ptls, o, mark_mode);
#else
    jl_assume(page);
    if (mark_mode == GC_OLD_MARKED) {
        ptls->gc_cache.perm_scanned_bytes += page->osize;
        static_assert(sizeof(_Atomic(uint16_t)) == sizeof(page->nold), "");
        jl_atomic_fetch_add_relaxed((_Atomic(uint16_t) *)&page->nold, 1);
    }
    else {
        ptls->gc_cache.scanned_bytes += page->osize;
        if (mark_reset_age) {
            page->has_young = 1;
            char *page_begin = gc_page_data(o) + GC_PAGE_OFFSET;
            int obj_id = (((char *)o) - page_begin) / page->osize;
            uint8_t *ages = page->ages + obj_id / 8;
            jl_atomic_fetch_and_relaxed((_Atomic(uint8_t) *)ages, ~(1 << (obj_id % 8)));
        }
    }
    objprofile_count(jl_typeof(jl_valueof(o)), mark_mode == GC_OLD_MARKED, page->osize);
    page->has_marked = 1;
#endif
}

STATIC_INLINE void gc_setmark_pool(jl_ptls_t ptls, jl_taggedvalue_t *o,
                                   uint8_t mark_mode) JL_NOTSAFEPOINT
{
    gc_setmark_pool_(ptls, o, mark_mode, page_metadata(o));
}

STATIC_INLINE void gc_setmark(jl_ptls_t ptls, jl_taggedvalue_t *o, uint8_t mark_mode,
                              size_t sz) JL_NOTSAFEPOINT
{
    if (sz <= GC_MAX_SZCLASS) {
        gc_setmark_pool(ptls, o, mark_mode);
    }
    else {
        gc_setmark_big(ptls, o, mark_mode);
    }
}

STATIC_INLINE void gc_setmark_buf_(jl_ptls_t ptls, void *o, uint8_t mark_mode,
                                   size_t minsz) JL_NOTSAFEPOINT
{
    jl_taggedvalue_t *buf = jl_astaggedvalue(o);
    uint8_t bits = (gc_old(buf->header) && !mark_reset_age) ? GC_OLD_MARKED : GC_MARKED;
    ;
    // If the object is larger than the max pool size it can't be a pool object.
    // This should be accurate most of the time but there might be corner cases
    // where the size estimate is a little off so we do a pool lookup to make
    // sure.
    if (__likely(gc_try_setmark_tag(buf, mark_mode)) && !gc_verifying) {
        if (minsz <= GC_MAX_SZCLASS) {
            jl_gc_pagemeta_t *page = page_metadata(buf);
            if (page) {
                gc_setmark_pool_(ptls, buf, bits, page);
                return;
            }
        }
        gc_setmark_big(ptls, buf, bits);
    }
}

void gc_setmark_buf(jl_ptls_t ptls, void *o, uint8_t mark_mode,
                    size_t minsz) JL_NOTSAFEPOINT
{
    gc_setmark_buf_(ptls, o, mark_mode, minsz);
}

void jl_gc_force_mark_old(jl_ptls_t ptls, jl_value_t *v) JL_NOTSAFEPOINT
{
    jl_taggedvalue_t *o = jl_astaggedvalue(v);
    jl_datatype_t *dt = (jl_datatype_t *)jl_typeof(v);
    size_t dtsz = jl_datatype_size(dt);
    if (o->bits.gc == GC_OLD_MARKED)
        return;
    o->bits.gc = GC_OLD_MARKED;
    if (dt == jl_simplevector_type) {
        size_t l = jl_svec_len(v);
        dtsz = l * sizeof(void *) + sizeof(jl_svec_t);
    }
    else if (dt->name == jl_array_typename) {
        jl_array_t *a = (jl_array_t *)v;
        if (!a->flags.pooled)
            dtsz = GC_MAX_SZCLASS + 1;
    }
    else if (dt == jl_module_type) {
        dtsz = sizeof(jl_module_t);
    }
    else if (dt == jl_task_type) {
        dtsz = sizeof(jl_task_t);
    }
    else if (dt == jl_symbol_type) {
        return;
    }
    gc_setmark(ptls, o, GC_OLD_MARKED, dtsz);
    if (dt->layout->npointers != 0)
        jl_gc_queue_root(v);
}

// Handle the case where the stack is only partially copied.
STATIC_INLINE uintptr_t gc_get_stack_addr(void *_addr, uintptr_t offset, uintptr_t lb,
                                          uintptr_t ub)
{
    uintptr_t addr = (uintptr_t)_addr;
    if (addr >= lb && addr < ub)
        return addr + offset;
    return addr;
}

STATIC_INLINE uintptr_t gc_read_stack(void *_addr, uintptr_t offset, uintptr_t lb,
                                      uintptr_t ub)
{
    uintptr_t real_addr = gc_get_stack_addr(_addr, offset, lb, ub);
    return *(uintptr_t *)real_addr;
}

JL_NORETURN NOINLINE void gc_assert_datatype_fail(jl_ptls_t ptls, jl_datatype_t *vt,
                                                  jl_gc_markqueue_t *mq)
{
    jl_safe_printf("GC error (probable corruption) :\n");
    jl_gc_debug_print_status();
    jl_(vt);
    jl_gc_debug_critical_error();
    gc_mark_loop_unwind(ptls, mq, 0);
    abort();
}

// Check if `nptr` is tagged for `old + refyoung`,
// Push the object to the remset and update the `nptr` counter if necessary.
STATIC_INLINE void gc_mark_push_remset(jl_ptls_t ptls, jl_value_t *obj,
                                       uintptr_t nptr) JL_NOTSAFEPOINT
{
    if (__unlikely((nptr & 0x3) == 0x3)) {
        ptls->heap.remset_nptr += nptr >> 2;
        arraylist_t *remset = ptls->heap.remset;
        size_t len = remset->len;
        if (__unlikely(len >= remset->max)) {
            arraylist_push(remset, obj);
        }
        else {
            remset->len = len + 1;
            remset->items[len] = obj;
        }
    }
}

// Enqueue an unmarked obj. last bit of `nptr` is set if `_obj` is young
STATIC_INLINE void gc_try_claim_and_push(jl_gc_markqueue_t *mq, void *_obj,
                                         uintptr_t *nptr) JL_NOTSAFEPOINT
{
    if (!_obj)
        return;
    jl_value_t *obj = (jl_value_t *)jl_assume(_obj);
    jl_taggedvalue_t *o = jl_astaggedvalue(obj);
    if (!gc_old(o->header) && nptr)
        *nptr |= 1;
    if (gc_try_setmark_tag(o, GC_MARKED))
        gc_markqueue_push(mq, obj);
}

// Mark object with 8bit field descriptors
STATIC_INLINE void gc_mark_obj8(jl_ptls_t ptls, char *obj8_parent, uint8_t *obj8_begin,
                                uint8_t *obj8_end, uintptr_t nptr) JL_NOTSAFEPOINT
{
    (void)jl_assume(obj8_begin < obj8_end);
    jl_gc_markqueue_t *mq = &ptls->mark_queue;
    jl_value_t *new_obj;
    for (; obj8_begin < obj8_end; obj8_begin++) {
        new_obj = ((jl_value_t **)obj8_parent)[*obj8_begin];
        if (new_obj)
            verify_parent2("object", obj8_parent, &new_obj, "field(%d)",
                           gc_slot_to_fieldidx(obj8_parent, &new_obj));
        gc_try_claim_and_push(mq, new_obj, &nptr);
    }
    gc_mark_push_remset(ptls, (jl_value_t *)obj8_parent, nptr);
}

// Mark object with 16bit field descriptors
STATIC_INLINE void gc_mark_obj16(jl_ptls_t ptls, char *obj16_parent, uint16_t *obj16_begin,
                                 uint16_t *obj16_end, uintptr_t nptr) JL_NOTSAFEPOINT
{
    (void)jl_assume(obj16_begin < obj16_end);
    jl_gc_markqueue_t *mq = &ptls->mark_queue;
    jl_value_t *new_obj;
    for (; obj16_begin < obj16_end; obj16_begin++) {
        new_obj = ((jl_value_t **)obj16_parent)[*obj16_begin];
        if (new_obj)
            verify_parent2("object", obj16_parent, &new_obj, "field(%d)",
                           gc_slot_to_fieldidx(obj16_parent, &new_obj));
        gc_try_claim_and_push(mq, new_obj, &nptr);
    }
    gc_mark_push_remset(ptls, (jl_value_t *)obj16_parent, nptr);
}

// Mark object with 32bit field descriptors
STATIC_INLINE void gc_mark_obj32(jl_ptls_t ptls, char *obj32_parent, uint32_t *obj32_begin,
                                 uint32_t *obj32_end, uintptr_t nptr) JL_NOTSAFEPOINT
{
    (void)jl_assume(obj32_begin < obj32_end);
    jl_gc_markqueue_t *mq = &ptls->mark_queue;
    jl_value_t *new_obj;
    for (; obj32_begin < obj32_end; obj32_begin++) {
        new_obj = ((jl_value_t **)obj32_parent)[*obj32_begin];
        if (new_obj)
            verify_parent2("object", obj32_parent, &new_obj, "field(%d)",
                           gc_slot_to_fieldidx(obj32_parent, &new_obj));
        gc_try_claim_and_push(mq, new_obj, &nptr);
    }
    gc_mark_push_remset(ptls, (jl_value_t *)obj32_parent, nptr);
}

// Mark object array
STATIC_INLINE void gc_mark_objarray(jl_ptls_t ptls, jl_value_t *obj_parent,
                                    jl_value_t **obj_begin, jl_value_t **obj_end,
                                    uint32_t step, uintptr_t nptr) JL_NOTSAFEPOINT
{
    jl_gc_markqueue_t *mq = &ptls->mark_queue;
    jl_value_t *new_obj;
    for (; obj_begin < obj_end; obj_begin += step) {
        new_obj = *obj_begin;
        if (new_obj)
            verify_parent2("obj array", obj_parent, obj_begin, "elem(%d)",
                           gc_slot_to_arrayidx(obj_parent, obj_begin));
        gc_try_claim_and_push(mq, new_obj, &nptr);
    }
    gc_mark_push_remset(ptls, obj_parent, nptr);
}

// Mark array with 8bit field descriptors
STATIC_INLINE void gc_mark_array8(jl_ptls_t ptls, jl_value_t *ary8_parent,
                                  jl_value_t **ary8_begin, jl_value_t **ary8_end,
                                  uint8_t *elem_begin, uint8_t *elem_end,
                                  uintptr_t nptr) JL_NOTSAFEPOINT
{
    jl_gc_markqueue_t *mq = &ptls->mark_queue;
    jl_value_t *new_obj;
    size_t elsize = ((jl_array_t *)ary8_parent)->elsize / sizeof(jl_value_t *);
    for (; ary8_begin < ary8_end; ary8_begin += elsize) {
        for (uint8_t *pindex = elem_begin; pindex < elem_end; pindex++) {
            new_obj = ary8_begin[*pindex];
            if (new_obj)
                verify_parent2("array", ary8_parent, &new_obj, "elem(%d)",
                               gc_slot_to_arrayidx(ary8_parent, ary8_begin));
            gc_try_claim_and_push(mq, new_obj, &nptr);
        }
    }
    gc_mark_push_remset(ptls, ary8_parent, nptr);
}

// Mark array with 16bit field descriptors
STATIC_INLINE void gc_mark_array16(jl_ptls_t ptls, jl_value_t *ary16_parent,
                                   jl_value_t **ary16_begin, jl_value_t **ary16_end,
                                   uint16_t *elem_begin, uint16_t *elem_end,
                                   uintptr_t nptr) JL_NOTSAFEPOINT
{
    jl_gc_markqueue_t *mq = &ptls->mark_queue;
    jl_value_t *new_obj;
    size_t elsize = ((jl_array_t *)ary16_parent)->elsize / sizeof(jl_value_t *);
    for (; ary16_begin < ary16_end; ary16_begin += elsize) {
        for (uint16_t *pindex = elem_begin; pindex < elem_end; pindex++) {
            new_obj = ary16_begin[*pindex];
            if (new_obj)
                verify_parent2("array", ary16_parent, &new_obj, "elem(%d)",
                               gc_slot_to_arrayidx(ary16_parent, ary16_begin));
            gc_try_claim_and_push(mq, new_obj, &nptr);
        }
    }
    gc_mark_push_remset(ptls, ary16_parent, nptr);
}

// Mark gc frame
STATIC_INLINE void gc_mark_stack(jl_ptls_t ptls, jl_gcframe_t *s, uint32_t nroots,
                                 uintptr_t offset, uintptr_t lb,
                                 uintptr_t ub) JL_NOTSAFEPOINT
{
    jl_gc_markqueue_t *mq = &ptls->mark_queue;
    jl_value_t *new_obj;
    uint32_t nr = nroots >> 2;
    while (1) {
        jl_value_t ***rts = (jl_value_t ***)(((void **)s) + 2);
        for (uint32_t i = 0; i < nr; i++) {
            if (nroots & 1) {
                void **slot = (void **)gc_read_stack(&rts[i], offset, lb, ub);
                new_obj = (jl_value_t *)gc_read_stack(slot, offset, lb, ub);
            }
            else {
                new_obj = (jl_value_t *)gc_read_stack(&rts[i], offset, lb, ub);
                if (gc_ptr_tag(new_obj, 1)) {
                    // handle tagged pointers in finalizer list
                    new_obj = gc_ptr_clear_tag(new_obj, 1);
                    i++;
                }
            }
            gc_try_claim_and_push(mq, new_obj, NULL);
        }
        s = (jl_gcframe_t *)gc_read_stack(&s->prev, offset, lb, ub);
        if (!s)
            break;
        uintptr_t new_nroots = gc_read_stack(&s->nroots, offset, lb, ub);
        assert(new_nroots <= UINT32_MAX);
        nroots = (uint32_t)new_nroots;
        nr = nroots >> 2;
    }
}

// Mark exception stack
STATIC_INLINE void gc_mark_excstack(jl_ptls_t ptls, jl_excstack_t *excstack,
                                    size_t itr) JL_NOTSAFEPOINT
{
    jl_gc_markqueue_t *mq = &ptls->mark_queue;
    jl_value_t *new_obj;
    while (itr > 0) {
        size_t bt_size = jl_excstack_bt_size(excstack, itr);
        jl_bt_element_t *bt_data = jl_excstack_bt_data(excstack, itr);
        for (size_t bt_index = 0; bt_index < bt_size;
             bt_index += jl_bt_entry_size(bt_data + bt_index)) {
            jl_bt_element_t *bt_entry = bt_data + bt_index;
            if (jl_bt_is_native(bt_entry))
                continue;
            // Found an extended backtrace entry: iterate over any
            // GC-managed values inside.
            size_t njlvals = jl_bt_num_jlvals(bt_entry);
            for (size_t jlval_index = 0; jlval_index < njlvals; jlval_index++) {
                new_obj = jl_bt_entry_jlvalue(bt_entry, jlval_index);
                gc_try_claim_and_push(mq, new_obj, NULL);
            }
        }
        // The exception comes last - mark it
        new_obj = jl_excstack_exception(excstack, itr);
        itr = jl_excstack_next(excstack, itr);
        gc_try_claim_and_push(mq, new_obj, NULL);
    }
}

// Mark module binding
STATIC_INLINE void gc_mark_module_binding(jl_ptls_t ptls, jl_module_t *mb_parent,
                                          jl_binding_t **mb_begin, jl_binding_t **mb_end,
                                          uintptr_t nptr, uint8_t bits) JL_NOTSAFEPOINT
{
    jl_gc_markqueue_t *mq = &ptls->mark_queue;
    for (; mb_begin < mb_end; mb_begin += 2) {
        jl_binding_t *b = *mb_begin;
        if (b == (jl_binding_t *)HT_NOTFOUND)
            continue;
        if ((void *)b >= sysimg_base && (void *)b < sysimg_end) {
            jl_taggedvalue_t *buf = jl_astaggedvalue(b);
            gc_try_setmark_tag(buf, GC_OLD_MARKED);
        }
        else {
            gc_setmark_buf_(ptls, b, bits, sizeof(jl_binding_t));
        }
        void *vb = jl_astaggedvalue(b);
        verify_parent1("module", mb_parent, &vb, "binding_buff");
        (void)vb;
        jl_value_t *value = jl_atomic_load_relaxed(&b->value);
        jl_value_t *globalref = jl_atomic_load_relaxed(&b->globalref);
        if (value) {
            verify_parent2("module", mb_parent, &b->value, "binding(%s)",
                           jl_symbol_name(b->name));
            gc_try_claim_and_push(mq, value, &nptr);
        }
        gc_try_claim_and_push(mq, globalref, &nptr);
    }
    gc_try_claim_and_push(mq, (jl_value_t *)mb_parent->parent, &nptr);
    size_t nusings = mb_parent->usings.len;
    if (nusings > 0) {
        // This is only necessary because bindings for "using" modules
        // are added only when accessed. therefore if a module is replaced
        // after "using" it but before accessing it, this array might
        // contain the only reference.
        jl_value_t *obj_parent = (jl_value_t *)mb_parent;
        jl_value_t **objary_begin = (jl_value_t **)mb_parent->usings.items;
        jl_value_t **objary_end = objary_begin + nusings;
        gc_mark_objarray(ptls, obj_parent, objary_begin, objary_end, 1, nptr);
    }
    else {
        gc_mark_push_remset(ptls, (jl_value_t *)mb_parent, nptr);
    }
}

// Mark finalizer list (or list of objects following same format)
void gc_mark_finlist(jl_ptls_t ptls, arraylist_t *list, size_t start) JL_NOTSAFEPOINT
{
    jl_gc_markqueue_t *mq = &ptls->mark_queue;
    jl_value_t *new_obj;
    size_t len = list->len;
    if (len <= start)
        return;
    jl_value_t **fl_begin = (jl_value_t **)list->items + start;
    jl_value_t **fl_end = (jl_value_t **)list->items + len;
    for (; fl_begin < fl_end; fl_begin++) {
        new_obj = *fl_begin;
        if (__unlikely(!new_obj))
            continue;
        if (gc_ptr_tag(new_obj, 1)) {
            new_obj = (jl_value_t *)gc_ptr_clear_tag(new_obj, 1);
            fl_begin++;
            assert(fl_begin < fl_end);
        }
        gc_try_claim_and_push(mq, new_obj, NULL);
    }
}

JL_DLLEXPORT int jl_gc_mark_queue_obj(jl_ptls_t ptls, jl_value_t *obj)
{
    int may_claim = gc_try_setmark_tag(jl_astaggedvalue(obj), GC_MARKED);
    if (may_claim)
        gc_markqueue_push(&ptls->mark_queue, obj);
    return may_claim;
}

JL_DLLEXPORT void jl_gc_mark_queue_objarray(jl_ptls_t ptls, jl_value_t *parent,
                                            jl_value_t **objs, size_t nobjs)
{
    uintptr_t nptr = (nobjs << 2) & (jl_astaggedvalue(parent)->bits.gc & 3);
    gc_mark_objarray(ptls, parent, objs, objs + nobjs, 1, nptr);
}

// Enqueue and mark all outgoing references from `new_obj` which have not been marked
// yet. `meta_updated` is mostly used to make sure we don't update metadata twice for
// objects which have been enqueued into the `remset`
NOINLINE void gc_mark_outrefs(jl_ptls_t ptls, jl_value_t *new_obj, int meta_updated)
{
    jl_gc_markqueue_t *mq = &ptls->mark_queue;
#ifdef JL_DEBUG_BUILD
    if (new_obj == gc_findval)
        jl_raise_debugger();
#endif
    jl_taggedvalue_t *o = jl_astaggedvalue(new_obj);
    jl_datatype_t *vt = (jl_datatype_t *)(o->header & ~(uintptr_t)0xf);
    uint8_t bits = (gc_old(o->header) && !mark_reset_age) ? GC_OLD_MARKED : GC_MARKED;
    int update_meta = __likely(!meta_updated && !gc_verifying);
    int foreign_alloc = 0;
    if (update_meta && (void *)o >= sysimg_base && (void *)o < sysimg_end) {
        foreign_alloc = 1;
        update_meta = 0;
    }
    // Symbols are always marked
    assert(vt != jl_symbol_type);
    if (vt == jl_simplevector_type) {
        size_t l = jl_svec_len(new_obj);
        jl_value_t **data = jl_svec_data(new_obj);
        size_t dtsz = l * sizeof(void *) + sizeof(jl_svec_t);
        if (update_meta)
            gc_setmark(ptls, o, bits, dtsz);
        else if (foreign_alloc)
            objprofile_count(vt, bits == GC_OLD_MARKED, dtsz);
        jl_value_t *objary_parent = new_obj;
        jl_value_t **objary_begin = data;
        jl_value_t **objary_end = data + l;
        uint32_t step = 1;
        uintptr_t nptr = (l << 2) | (bits & GC_OLD);
        gc_mark_objarray(ptls, objary_parent, objary_begin, objary_end, step, nptr);
    }
    else if (vt->name == jl_array_typename) {
        jl_array_t *a = (jl_array_t *)new_obj;
        jl_array_flags_t flags = a->flags;
        if (update_meta) {
            if (flags.pooled)
                gc_setmark_pool(ptls, o, bits);
            else
                gc_setmark_big(ptls, o, bits);
        }
        else if (foreign_alloc) {
            objprofile_count(vt, bits == GC_OLD_MARKED, sizeof(jl_array_t));
        }
        if (flags.how == 1) {
            void *val_buf = jl_astaggedvalue((char *)a->data - a->offset * a->elsize);
            verify_parent1("array", new_obj, &val_buf,
                           "buffer ('loc' addr is meaningless)");
            (void)val_buf;
            gc_setmark_buf_(ptls, (char *)a->data - a->offset * a->elsize, bits,
                            jl_array_nbytes(a));
        }
        else if (flags.how == 2) {
            if (update_meta || foreign_alloc) {
                objprofile_count(jl_malloc_tag, bits == GC_OLD_MARKED, jl_array_nbytes(a));
                if (bits == GC_OLD_MARKED)
                    ptls->gc_cache.perm_scanned_bytes += jl_array_nbytes(a);
                else
                    ptls->gc_cache.scanned_bytes += jl_array_nbytes(a);
            }
        }
        else if (flags.how == 3) {
            jl_value_t *owner = jl_array_data_owner(a);
            uintptr_t nptr = (1 << 2) | (bits & GC_OLD);
            gc_try_claim_and_push(mq, owner, &nptr);
            gc_mark_push_remset(ptls, new_obj, nptr);
            return;
        }
        if (!a->data || jl_array_len(a) == 0)
            return;
        if (flags.ptrarray) {
            if ((jl_datatype_t *)jl_tparam0(vt) == jl_symbol_type)
                return;
            size_t l = jl_array_len(a);
            jl_value_t *objary_parent = new_obj;
            jl_value_t **objary_begin = (jl_value_t **)a->data;
            jl_value_t **objary_end = objary_begin + l;
            uint32_t step = 1;
            uintptr_t nptr = (l << 2) | (bits & GC_OLD);
            gc_mark_objarray(ptls, objary_parent, objary_begin, objary_end, step, nptr);
        }
        else if (flags.hasptr) {
            jl_datatype_t *et = (jl_datatype_t *)jl_tparam0(vt);
            const jl_datatype_layout_t *layout = et->layout;
            unsigned npointers = layout->npointers;
            unsigned elsize = a->elsize / sizeof(jl_value_t *);
            size_t l = jl_array_len(a);
            jl_value_t *objary_parent = new_obj;
            jl_value_t **objary_begin = (jl_value_t **)a->data;
            jl_value_t **objary_end = objary_begin + l * elsize;
            uint32_t step = elsize;
            uintptr_t nptr = ((l * npointers) << 2) | (bits & GC_OLD);
            if (npointers == 1) { // TODO: detect anytime time stride is uniform?
                objary_begin += layout->first_ptr;
                gc_mark_objarray(ptls, objary_parent, objary_begin, objary_end, step, nptr);
            }
            else if (layout->fielddesc_type == 0) {
                uint8_t *obj8_begin = (uint8_t *)jl_dt_layout_ptrs(layout);
                uint8_t *obj8_end = obj8_begin + npointers;
                gc_mark_array8(ptls, objary_parent, objary_begin, objary_end, obj8_begin,
                               obj8_end, nptr);
            }
            else if (layout->fielddesc_type == 1) {
                uint16_t *obj16_begin = (uint16_t *)jl_dt_layout_ptrs(layout);
                uint16_t *obj16_end = obj16_begin + npointers;
                gc_mark_array16(ptls, objary_parent, objary_begin, objary_end, obj16_begin,
                                obj16_end, nptr);
            }
            else {
                assert(0 && "unimplemented");
            }
        }
    }
    else if (vt == jl_module_type) {
        if (update_meta)
            gc_setmark(ptls, o, bits, sizeof(jl_module_t));
        else if (foreign_alloc)
            objprofile_count(vt, bits == GC_OLD_MARKED, sizeof(jl_module_t));
        jl_module_t *mb_parent = (jl_module_t *)new_obj;
        jl_binding_t **mb_begin = (jl_binding_t **)mb_parent->bindings.table + 1;
        size_t bsize = mb_parent->bindings.size;
        jl_binding_t **mb_end = (jl_binding_t **)mb_parent->bindings.table + bsize;
        uintptr_t nptr = ((bsize + mb_parent->usings.len + 1) << 2) | (bits & GC_OLD);
        gc_mark_module_binding(ptls, mb_parent, mb_begin, mb_end, nptr, bits);
    }
    else if (vt == jl_task_type) {
        if (update_meta)
            gc_setmark(ptls, o, bits, sizeof(jl_task_t));
        else if (foreign_alloc)
            objprofile_count(vt, bits == GC_OLD_MARKED, sizeof(jl_task_t));
        jl_task_t *ta = (jl_task_t *)new_obj;
        gc_scrub_record_task(ta);
        if (gc_cblist_task_scanner) {
            int16_t tid = jl_atomic_load_relaxed(&ta->tid);
            gc_invoke_callbacks(jl_gc_cb_task_scanner_t, gc_cblist_task_scanner,
                                (ta, tid != -1 && ta == jl_all_tls_states[tid]->root_task));
        }
#ifdef COPY_STACKS
        void *stkbuf = ta->stkbuf;
        if (stkbuf && ta->copy_stack)
            gc_setmark_buf_(ptls, stkbuf, bits, ta->bufsz);
#endif
        jl_gcframe_t *s = ta->gcstack;
        size_t nroots;
        uintptr_t offset = 0;
        uintptr_t lb = 0;
        uintptr_t ub = (uintptr_t)-1;
#ifdef COPY_STACKS
        if (stkbuf && ta->copy_stack && !ta->ptls) {
            int16_t tid = jl_atomic_load_relaxed(&ta->tid);
            assert(tid >= 0);
            jl_ptls_t ptls2 = jl_all_tls_states[tid];
            ub = (uintptr_t)ptls2->stackbase;
            lb = ub - ta->copy_stack;
            offset = (uintptr_t)stkbuf - lb;
        }
#endif
        if (s) {
            nroots = gc_read_stack(&s->nroots, offset, lb, ub);
            assert(nroots <= UINT32_MAX);
            gc_mark_stack(ptls, s, (uint32_t)nroots, offset, lb, ub);
        }
        if (ta->excstack) {
            jl_excstack_t *excstack = ta->excstack;
            size_t itr = ta->excstack->top;
            gc_setmark_buf_(ptls, excstack, bits,
                            sizeof(jl_excstack_t) +
                                sizeof(uintptr_t) * excstack->reserved_size);
            gc_mark_excstack(ptls, excstack, itr);
        }
        const jl_datatype_layout_t *layout = jl_task_type->layout;
        assert(layout->fielddesc_type == 0);
        assert(layout->nfields > 0);
        uint32_t npointers = layout->npointers;
        char *obj8_parent = (char *)ta;
        uint8_t *obj8_begin = (uint8_t *)jl_dt_layout_ptrs(layout);
        uint8_t *obj8_end = obj8_begin + npointers;
        // assume tasks always reference young objects: set lowest bit
        uintptr_t nptr = (npointers << 2) | 1 | bits;
        gc_mark_obj8(ptls, obj8_parent, obj8_begin, obj8_end, nptr);
    }
    else if (vt == jl_string_type) {
        size_t dtsz = jl_string_len(new_obj) + sizeof(size_t) + 1;
        if (update_meta)
            gc_setmark(ptls, o, bits, dtsz);
        else if (foreign_alloc)
            objprofile_count(vt, bits == GC_OLD_MARKED, dtsz);
    }
    else {
        if (__unlikely(!jl_is_datatype(vt)))
            gc_assert_datatype_fail(ptls, vt, mq);
        size_t dtsz = jl_datatype_size(vt);
        if (update_meta)
            gc_setmark(ptls, o, bits, dtsz);
        else if (foreign_alloc)
            objprofile_count(vt, bits == GC_OLD_MARKED, dtsz);
        if (vt == jl_weakref_type)
            return;
        const jl_datatype_layout_t *layout = vt->layout;
        uint32_t npointers = layout->npointers;
        if (npointers == 0)
            return;
        uintptr_t nptr = (npointers << 2 | (bits & GC_OLD));
        assert((layout->nfields > 0 || layout->fielddesc_type == 3) &&
               "opaque types should have been handled specially");
        if (layout->fielddesc_type == 0) {
            char *obj8_parent = (char *)new_obj;
            uint8_t *obj8_begin = (uint8_t *)jl_dt_layout_ptrs(layout);
            uint8_t *obj8_end = obj8_begin + npointers;
            assert(obj8_begin < obj8_end);
            gc_mark_obj8(ptls, obj8_parent, obj8_begin, obj8_end, nptr);
        }
        else if (layout->fielddesc_type == 1) {
            char *obj16_parent = (char *)new_obj;
            uint16_t *obj16_begin = (uint16_t *)jl_dt_layout_ptrs(layout);
            uint16_t *obj16_end = obj16_begin + npointers;
            assert(obj16_begin < obj16_end);
            gc_mark_obj16(ptls, obj16_parent, obj16_begin, obj16_end, nptr);
        }
        else if (layout->fielddesc_type == 2) {
            // This is very uncommon
            // Do not do store to load forwarding to save some code size
            char *obj32_parent = (char *)new_obj;
            uint32_t *obj32_begin = (uint32_t *)jl_dt_layout_ptrs(layout);
            uint32_t *obj32_end = obj32_begin + npointers;
            gc_mark_obj32(ptls, obj32_parent, obj32_begin, obj32_end, nptr);
        }
        else {
            assert(layout->fielddesc_type == 3);
            jl_fielddescdyn_t *desc = (jl_fielddescdyn_t *)jl_dt_layout_fields(layout);
            int old = jl_astaggedvalue(new_obj)->bits.gc & 2;
            uintptr_t young = desc->markfunc(ptls, new_obj);
            if (old && young)
                gc_mark_push_remset(ptls, new_obj, young * 4 + 3);
        }
    }
}

// Main mark loop. Single stack (allocated on the heap) of `jl_value_t *`
// is used to keep track of processed items. Maintaning this stack (instead of
// native one) avoids stack overflow when marking deep objects and
// makes it easier to implement parallel marking via work-stealing
JL_EXTENSION NOINLINE void gc_mark_loop(jl_ptls_t ptls)
{
    while (1) {
        jl_value_t *new_obj = gc_markqueue_pop(&ptls->mark_queue);
        // No more objects to mark
        if (!new_obj) {
            // TODO: work-stealing comes here...
            return;
        }
        gc_mark_outrefs(ptls, new_obj, 0);
    }
}

void gc_premark(jl_ptls_t ptls2)
{
    arraylist_t *remset = ptls2->heap.remset;
    ptls2->heap.remset = ptls2->heap.last_remset;
    ptls2->heap.last_remset = remset;
    ptls2->heap.remset->len = 0;
    ptls2->heap.remset_nptr = 0;
    // Avoid counting remembered objects & bindings twice
    // in `perm_scanned_bytes`
    size_t len = remset->len;
    void **items = remset->items;
    for (size_t i = 0; i < len; i++) {
        jl_value_t *item = (jl_value_t *)items[i];
        objprofile_count(jl_typeof(item), 2, 0);
        jl_astaggedvalue(item)->bits.gc = GC_OLD_MARKED;
    }
    len = ptls2->heap.rem_bindings.len;
    items = ptls2->heap.rem_bindings.items;
    for (size_t i = 0; i < len; i++) {
        void *ptr = items[i];
        jl_astaggedvalue(ptr)->bits.gc = GC_OLD_MARKED;
    }
}

void gc_queue_thread_local(jl_gc_markqueue_t *mq, jl_ptls_t ptls2)
{
    gc_try_claim_and_push(mq, jl_atomic_load_relaxed(&ptls2->current_task), NULL);
    gc_try_claim_and_push(mq, ptls2->root_task, NULL);
    gc_try_claim_and_push(mq, ptls2->next_task, NULL);
    gc_try_claim_and_push(mq, ptls2->previous_task, NULL);
    gc_try_claim_and_push(mq, ptls2->previous_exception, NULL);
}

void gc_queue_bt_buf(jl_gc_markqueue_t *mq, jl_ptls_t ptls2)
{
    jl_bt_element_t *bt_data = ptls2->bt_data;
    size_t bt_size = ptls2->bt_size;
    for (size_t i = 0; i < bt_size; i += jl_bt_entry_size(bt_data + i)) {
        jl_bt_element_t *bt_entry = bt_data + i;
        if (jl_bt_is_native(bt_entry))
            continue;
        size_t njlvals = jl_bt_num_jlvals(bt_entry);
        for (size_t j = 0; j < njlvals; j++)
            gc_try_claim_and_push(mq, jl_bt_entry_jlvalue(bt_entry, j), NULL);
    }
}

void gc_queue_remset(jl_ptls_t ptls, jl_ptls_t ptls2)
{
    size_t len = ptls2->heap.last_remset->len;
    void **items = ptls2->heap.last_remset->items;
    for (size_t i = 0; i < len; i++) {
        // Objects in the `remset` are already marked,
        // so a `gc_try_claim_and_push` wouldn't work here
        gc_mark_outrefs(ptls, (jl_value_t *)items[i], 1);
    }
    int n_bnd_refyoung = 0;
    len = ptls2->heap.rem_bindings.len;
    items = ptls2->heap.rem_bindings.items;
    for (size_t i = 0; i < len; i++) {
        jl_binding_t *ptr = (jl_binding_t *)items[i];
        // A null pointer can happen here when the binding is cleaned up
        // as an exception is thrown after it was already queued (#10221)
        jl_value_t *v = jl_atomic_load_relaxed(&ptr->value);
        gc_try_claim_and_push(&ptls->mark_queue, v, NULL);
        if (v && !gc_old(jl_astaggedvalue(v)->header)) {
            items[n_bnd_refyoung] = ptr;
            n_bnd_refyoung++;
        }
    }
    ptls2->heap.rem_bindings.len = n_bnd_refyoung;
}

extern jl_value_t *cmpswap_names JL_GLOBALLY_ROOTED;

// Mark the initial root set
void gc_mark_roots(jl_gc_markqueue_t *mq)
{
    // Modules
    gc_try_claim_and_push(mq, jl_main_module, NULL);
    // Invisible builtin values
    gc_try_claim_and_push(mq, jl_an_empty_vec_any, NULL);
    gc_try_claim_and_push(mq, jl_module_init_order, NULL);
    for (size_t i = 0; i < jl_current_modules.size; i += 2) {
        if (jl_current_modules.table[i + 1] != HT_NOTFOUND) {
            gc_try_claim_and_push(mq, jl_current_modules.table[i], NULL);
        }
    }
    gc_try_claim_and_push(mq, jl_anytuple_type_type, NULL);
    for (size_t i = 0; i < N_CALL_CACHE; i++) {
        jl_typemap_entry_t *v = jl_atomic_load_relaxed(&call_cache[i]);
        gc_try_claim_and_push(mq, v, NULL);
    }
    gc_try_claim_and_push(mq, jl_all_methods, NULL);
    gc_try_claim_and_push(mq, _jl_debug_method_invalidation, NULL);
    // Constants
    gc_try_claim_and_push(mq, jl_emptytuple_type, NULL);
    gc_try_claim_and_push(mq, cmpswap_names, NULL);
}

#ifdef __cplusplus
}
#endif