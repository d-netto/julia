; This file is a part of Julia. License is MIT: https://julialang.org/license

; RUN: opt -enable-new-pm=0 --opaque-pointers=0 -load libjulia-codegen%shlibext -JuliaLICM -S %s | FileCheck %s --check-prefixes=CHECK,TYPED
; RUN: opt -enable-new-pm=1 --opaque-pointers=0 --load-pass-plugin=libjulia-codegen%shlibext -passes='JuliaLICM' -S %s | FileCheck %s --check-prefixes=CHECK,TYPED

; RUN: opt -enable-new-pm=0 --opaque-pointers=1 -load libjulia-codegen%shlibext -JuliaLICM -S %s | FileCheck %s --check-prefixes=CHECK,OPAQUE
; RUN: opt -enable-new-pm=1 --opaque-pointers=1 --load-pass-plugin=libjulia-codegen%shlibext -passes='JuliaLICM' -S %s | FileCheck %s --check-prefixes=CHECK,OPAQUE

@tag = external addrspace(10) global {}, align 16

declare void @julia.write_barrier({}*, ...)

declare {}*** @julia.get_pgcstack()

; COM: check basic allocation hoisting functionality
; CHECK-LABEL: @julia_allocation_hoist
define nonnull {} addrspace(10)* @julia_allocation_hoist(i64 signext %0) #0 {
top:
  %1 = call {}*** @julia.get_pgcstack()
  %2 = icmp sgt i64 %0, 0
  br i1 %2, label %L4, label %L3

L3.loopexit:                                      ; preds = %L22
  %.lcssa = phi {} addrspace(10)* [ %3, %L22 ]
  br label %L3

L3:                                               ; preds = %L3.loopexit, %top
  %merge = phi {} addrspace(10)* [ addrspacecast ({}* inttoptr (i64 139952239804424 to {}*) to {} addrspace(10)*), %top ], [ %.lcssa, %L3.loopexit ]
  ret {} addrspace(10)* %merge

L4:                                               ; preds = %top
  %current_task112 = getelementptr inbounds {}**, {}*** %1, i64 -12
  %current_task1 = bitcast {}*** %current_task112 to {}**
  ; TYPED: %3 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task1, i64 8, {} addrspace(10)* @tag)
  ; TYPED-NEXT: %4 = bitcast {} addrspace(10)* %3 to i8 addrspace(10)*
  ; TYPED-NEXT: call void @llvm.memset.p10i8.i64(i8 addrspace(10)* align {{[0-9]+}} %4, i8 0, i64 8, i1 false)

  ; OPAQUE: %3 = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr nonnull %current_task1, i64 8, ptr addrspace(10) @tag)
  ; OPAQUE-NEXT: call void @llvm.memset.p10.i64(ptr addrspace(10) align {{[0-9]+}} %3, i8 0, i64 8, i1 false)

  ; CHECK-NEXT: br label %L22
  br label %L22

L22:                                              ; preds = %L4, %L22
  %value_phi5 = phi i64 [ 1, %L4 ], [ %5, %L22 ]
  ; TYPED: %value_phi5 = phi i64 [ 1, %L4 ], [ %6, %L22 ]
  ; TYPED-NEXT %5 = bitcast {} addrspace(10)* %3 to i64 addrspace(10)*

  ; OPAQUE: %value_phi5 = phi i64 [ 1, %L4 ], [ %5, %L22 ]
  ; OPAQUE-NEXT %4 = bitcast ptr addrspace(10) %3 to ptr addrspace(10)
  %3 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task1, i64 8, {} addrspace(10)* @tag) #1
  %4 = bitcast {} addrspace(10)* %3 to i64 addrspace(10)*
  store i64 %value_phi5, i64 addrspace(10)* %4, align 8, !tbaa !2
  %.not = icmp eq i64 %value_phi5, %0
  %5 = add i64 %value_phi5, 1
  br i1 %.not, label %L3.loopexit, label %L22
}

; COM: check that we hoist the allocation out of the loop despite returning the allocation
; CHECK-LABEL: @julia_hoist_returned
define nonnull {} addrspace(10)* @julia_hoist_returned(i64 signext %n, i1 zeroext %ret) {
top:
  %pgcstack = call {}*** @julia.get_pgcstack()
  %current_task = bitcast {}*** %pgcstack to {}**
; CHECK: br label %preheader
  br label %preheader
; CHECK: preheader:
preheader:
; TYPED-NEXT: %alloc = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task, i64 8, {} addrspace(10)* @tag)
; TYPED-NEXT: [[casted:%.*]] = bitcast {} addrspace(10)* %alloc to i8 addrspace(10)*
; TYPED-NEXT: call void @llvm.memset.p10i8.i64(i8 addrspace(10)* align {{[0-9]+}} [[casted]], i8 0, i64 8, i1 false)

; OPAQUE-NEXT: %alloc = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr nonnull %current_task, i64 8, ptr addrspace(10) @tag)
; OPAQUE-NEXT: call void @llvm.memset.p10.i64(ptr addrspace(10) align {{[0-9]+}} %alloc, i8 0, i64 8, i1 false)

; CHECK-NEXT: br label %loop
  br label %loop
loop:
  %alloc = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task, i64 8, {} addrspace(10)* @tag)
  br i1 %ret, label %return, label %loop
return:
  ret {} addrspace(10)* %alloc
}

; Function Attrs: allocsize(1)
declare noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}**, i64, {} addrspace(10)*) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: inaccessiblemem_or_argmemonly
declare void @ijl_gc_queue_root({} addrspace(10)*) #3

; Function Attrs: allocsize(1)
declare noalias nonnull {} addrspace(10)* @ijl_gc_pool_alloc(i8*, i32, i32) #1

; Function Attrs: allocsize(1)
declare noalias nonnull {} addrspace(10)* @ijl_gc_big_alloc(i8*, i64) #1

attributes #0 = { "probe-stack"="inline-asm" }
attributes #1 = { allocsize(1) }
attributes #2 = { argmemonly nofree nosync nounwind willreturn }
attributes #3 = { inaccessiblemem_or_argmemonly }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{!3, !3, i64 0}
!3 = !{!"jtbaa_mutab", !4, i64 0}
!4 = !{!"jtbaa_value", !5, i64 0}
!5 = !{!"jtbaa_data", !6, i64 0}
!6 = !{!"jtbaa", !7, i64 0}
!7 = !{!"jtbaa"}
