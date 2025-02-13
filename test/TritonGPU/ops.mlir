// RUN: triton-opt --split-input-file %s | FileCheck %s

// CHECK: #[[$WMMA_GEN1:.*]] = #ttg.amd_wmma<{{.*}}version = 1{{.*}}>
// CHECK: #[[$WMMA_GEN2:.*]] = #ttg.amd_wmma<{{.*}}version = 2{{.*}}>
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>

module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_layout
  tt.func @wmma_layout(%0: tensor<16x16xf16, #blocked>) {
    %1 = ttg.convert_layout %0 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #ttg.amd_wmma<{version = 1, warpsPerCTA = [1, 1]}>>
    // CHECK:  %{{.+}} = ttg.convert_layout %{{.+}} : tensor<16x16xf16, #{{.+}}> -> tensor<16x16xf16, #[[$WMMA_GEN1]]>
    tt.return
  }

  // CHECK-LABEL: wmma_dot_op_layout
  tt.func @wmma_dot_op_layout(%0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>) {
    %1 = ttg.convert_layout %0 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_wmma<{version = 1, warpsPerCTA = [1, 1]}>, kWidth = 16}>>
    // CHECK:  %{{.+}} = ttg.convert_layout %{{.+}} : tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.+}}}>> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$WMMA_GEN1]], kWidth = 16}>>
    tt.return
  }

  // CHECK-LABEL: wmma_gen2_layout
  tt.func @wmma_gen2_layout(%0: tensor<16x16xf16, #blocked>) {
    %1 = ttg.convert_layout %0 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #ttg.amd_wmma<{version = 2, warpsPerCTA = [1, 1]}>>
    // CHECK:  %{{.+}} = ttg.convert_layout %{{.+}} : tensor<16x16xf16, #{{.+}}> -> tensor<16x16xf16, #[[$WMMA_GEN2]]>
    tt.return
  }

  // CHECK-LABEL: wmma_gen2_dot_op_layout
  tt.func @wmma_gen2_dot_op_layout(%0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>) {
    %1 = ttg.convert_layout %0 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_wmma<{version = 2, warpsPerCTA = [1, 1]}>, kWidth = 8}>>
    // CHECK:  %{{.+}} = ttg.convert_layout %{{.+}} : tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.+}}}>> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$WMMA_GEN2]], kWidth = 8}>>
    tt.return
  }
}
// -----

#blocked= #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #[[$LINEAR:.*]] = #ttg.linear<{{.*}}>

module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @blocked_to_linear
  tt.func @blocked_to_linear(%input: tensor<32x4xi8, #blocked>) {
    // The layout is the basic layout generated by DecomposeScaledBlocked
    %output = ttg.convert_layout %input {allocation.offset = 0 : i32} : tensor<32x4xi8, #blocked> -> tensor<32x4xi8, #ttg.linear<{register = [], lane = [[0, 1], [1, 0], [2, 0], [4, 0], [8, 0]], warp = [[0, 0], [16, 0]], block = []}>>
    // CHECK:  %{{.+}} = ttg.convert_layout %{{.+}} : tensor<32x4xi8, #blocked> -> tensor<32x4xi8, #[[$LINEAR]]>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: memdesc
  // CHECK-SAME: !ttg.memdesc<1x64x16xf16, #{{.+}}>
  tt.func @memdesc(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>) {
    tt.return
  }
}

// -----

// CHECK-LABEL: @warp_specialize_nothing
tt.func @warp_specialize_nothing() {
  // CHECK-NEXT: ttg.warp_specialize()
  ttg.warp_specialize()
  // CHECK-NEXT: default {
  default {
    // CHECK-NEXT: ttg.warp_yield
    ttg.warp_yield
  // CHECK-NEXT: } : () -> ()
  } : () -> ()
  tt.return
}

// CHECK-LABEL: @warp_specialize_no_partitions
tt.func @warp_specialize_no_partitions(%arg0: i32, %arg1: i64) -> i64 {
  // CHECK-NEXT: %0 = ttg.warp_specialize(%arg0)
  %0 = ttg.warp_specialize(%arg0)
  // CHECK-NEXT: default {
  default {
    // CHECK-NEXT: ttg.warp_yield %arg1 : i64
    ttg.warp_yield %arg1 : i64
  // CHECK-NEXT: } : (i32) -> i64
  } : (i32) -> i64
  tt.return %0 : i64
}

// CHECK-LABEL: @warp_specialize_partitions
tt.func @warp_specialize_partitions(%arg0: i32, %arg1: i64) -> i64 {
  // CHECK-NEXT: %0 = ttg.warp_specialize(%arg0)
  %0 = ttg.warp_specialize(%arg0)
  // CHECK-NEXT: default {
  default {
    // CHECK-NEXT: ttg.warp_yield %arg1 : i64
    ttg.warp_yield %arg1 : i64
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: partition0(%arg2: i32) num_warps(4) {
  partition0(%arg2: i32) num_warps(4) {
    // CHECK-NEXT: arith.addi %arg2, %arg2 : i32
    %1 = arith.addi %arg2, %arg2 : i32
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: partition1(%arg2: i32) num_warps(1) {
  partition1(%arg2: i32) num_warps(1) {
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: partition2(%arg2: i32) num_warps(8) {
  partition2(%arg2: i32) num_warps(8) {
    // CHECK-NEXT: arith.muli
    %1 = arith.muli %arg2, %arg2 : i32
  // CHECK-NEXT: } : (i32) -> i64
  } : (i32) -> i64
  tt.return %0 : i64
}

// CHECK-LABEL: @warp_specialize_multiple_args
tt.func @warp_specialize_multiple_args_res(%arg0: i32, %arg1: i32) -> (i32, i32) {
  // CHECK-NEXT: %0:2 = ttg.warp_specialize(%arg0, %arg1)
  %0:2 = ttg.warp_specialize(%arg0, %arg1)
  // CHECK-NEXT: default {
  default {
    // CHECK-NEXT: ttg.warp_yield %arg0, %arg1 : i32, i32
    ttg.warp_yield %arg0, %arg1 : i32, i32
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: partition0(%arg2: i32, %arg3: i32) num_warps(4) {
  partition0(%arg2: i32, %arg3: i32) num_warps(4) {
    // CHECK-NEXT: arith.addi %arg2, %arg3 : i32
    %1 = arith.addi %arg2, %arg3 : i32
  // CHECK-NEXT: } : (i32, i32) -> (i32, i32)
  } : (i32, i32) -> (i32, i32)
  tt.return %0#0, %0#1 : i32, i32
}
