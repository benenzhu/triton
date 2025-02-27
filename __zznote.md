---
markmap:
  duration:
    - 0
  # other options
---
# triton 简介
- An Intermediate Language and Compiler for Tiled Neural Network Computations
- Tiled Computation，面向 GPU 体系特点，自动分析和实施 tiling
- 分层: 
    - froundend
    - optimizer 
    - backend 
- 复用 Dialect:
    - arith:   数学操作
    - scf:  if / for 控制流
    - nvvm: thread_id 等操作
    - nvgpu: ...
# step1: python -> tir
-  基本是一比一的对应过去, 从 python 的 ast  通过 mlir builder 直接映射到 triton / affine / scf 等的 ir, 
    - SCF (Structured Control Flow) 结构化控制流 scf.for, scf.if...
- [def compile(src, target=None](python/triton/compiler/compiler.py:219#def▪️compile(src,▪️target=None,▪️options=None):) 
    - [CUDABackend = make_backend(target)](python/triton/compiler/compiler.py:224#backend:"CUDABackend"▪️=▪️make_backend(target))  # 初始化了一些 nv 的库
    - [ir.load_dialects(context)](python/triton/compiler/compiler.py:271#ir.load_dialects(context)) # 加载dialet
        - [TritonDialect, TritonGPU, Math, Arith, Index, SCF](python/src/ir.cc:234#registry.insert<TritonDialect,▪️::mlir::triton::gpu::TritonGPUDialect,)
        - [ControlFlowDialect, GPU, LLVM, UB...](python/src/ir.cc:237#cf::ControlFlowDialect,▪️LLVM::LLVMDialect,)
        - 还做了一下 mlir 的公共dialet 的 load
    - [backend.load_dialects(context)](python/triton/compiler/compiler.py:272#backend.load_dialects(context)▪️#▪️nv的▪️load▪️dialects▪️#▪️python/src/ir.cc:232▪️triton▪️arith▪️math▪️等等▪️...)
        -  [nvidia_gpu::TritonNvidiaGPUDialect](third_party/nvidia/triton_nvidia.cc:66#registry.insert<mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,)
        - [nvgpu::NVGPUDialect](third_party/nvidia/triton_nvidia.cc:67#mlir::triton::nvgpu::NVGPUDialect>();)
    -  [src.make_ir](python/triton/compiler/compiler.py:276#module▪️=▪️src.make_ir(options,▪️codegen_fns,▪️module_map,▪️context)▪️#▪️走到▪️▪️▪️triton/compiler/compiler.py:101▪️ast_to_ttir)
        - [ast_to_ttir](python/triton/compiler/compiler.py:101#return▪️ast_to_ttir(self.fn,▪️self,▪️context=context,▪️options=options,▪️codegen_fns=codegen_fns,⬇️))
            - [ast.parse](python/triton/runtime/jit.py:760#tree▪️=▪️ast.parse(self.src)) # 调用 NodeVisitor的'visit_' + node.__class__.__name__
                - 例如 [visit_BinOp](python/triton/compiler/code_generator.py:546#def▪️visit_BinOp(self,▪️node):)
                    - [__add__](python/triton/language/core.py:763#def▪️__add__(self,▪️other,▪️_builder=None):)
                        - [semantic.add](python/triton/language/core.py:1903#return▪️semantic.add(x,▪️y,▪️sanitize_overflow,▪️_builder))
                            - [tl.tensor(builder.create_add](python/triton/language/semantic.py:241#return▪️tl.tensor(builder.create_add(input.handle,▪️other.handle),▪️input.type)▪️#▪️python/src/ir.cc:1013▪️▪️▪️return▪️self.create<arith::AddIOp>(lhs,▪️rhs);) # 最后是通过一个 pybinding 的形式
                                - [self.create<arith::AddIOp>(lhs, rhs)](python/src/ir.cc:1013#return▪️self.create<arith::AddIOp>(lhs,▪️rhs);) 调用 c++ 里面的 builder 创建 op
        - 更多 ir 的转换, 都有哪些? 看着和 xla ops 挺像的
        - [直接转换出来是比较抽象的, 类似](https://yunxiao.devops.xiaohongshu.com/cr/diff?mrId=2&projectId=48063&sourceProjectId=48063&fromCommitId=4fca481421a633bce8b4970f26bc9a967cd33122&toCommitId=381a0746aba7251482af6615f297204e31b192c5&fileKey=00000_a_py.mlir) yunxiao
            - 里面有很多的 arith.addi arith.cmpi 等等等
            - 过了规范化以后就都没有了
    - [compile_ir](python/triton/compiler/compiler.py:282#next_module▪️=▪️compile_ir(module,▪️metadata)) 对应上面的几个 stage 分开讲
        - [make_ttir](python/triton/backends/nvidia/compiler.py:379#stages["ttir"]▪️=▪️lambda▪️src,▪️metadata:▪️self.make_ttir(src,▪️metadata,▪️options))
        - [make_ttgir](python/triton/backends/nvidia/compiler.py:380#stages["ttgir"]▪️=▪️lambda▪️src,▪️metadata:▪️self.make_ttgir(src,▪️metadata,▪️options,▪️self.capability))
            - [cluster_info = nvidia.ClusterInfo()](python/triton/backends/nvidia/compiler.py:202#cluster_info▪️=▪️nvidia.ClusterInfo())
            - TODO:....
        - [make_llir](python/triton/backends/nvidia/compiler.py:381#stages["llir"]▪️=▪️lambda▪️src,▪️metadata:▪️self.make_llir(src,▪️metadata,▪️options,▪️self.capability))
        - [make_ptx](python/triton/backends/nvidia/compiler.py:382#stages["ptx"]▪️=▪️lambda▪️src,▪️metadata:▪️self.make_ptx(src,▪️metadata,▪️options,▪️self.capability))
        - [make_cubin](python/triton/backends/nvidia/compiler.py:383#stages["cubin"]▪️=▪️lambda▪️src,▪️metadata:▪️self.make_cubin(src,▪️metadata,▪️options,▪️self.capability))
# make_ttir
- 语言层面做一些硬件无关的恒等变换和简化
- [common.add_inliner](python/triton/backends/nvidia/compiler.py:188#passes.common.add_inliner(pm)) # inline function call, 类似mlir的 toy, 方便分析
    - TODO: 记得 inliner 是需要继承是否可以 inline 什么的, 找一下代码在哪里
- [ttir.add_rewrite_tensor_pointer](python/triton/backends/nvidia/compiler.py:189#passes.ttir.add_rewrite_tensor_pointer(pm))
- [ttir.add_combine](python/triton/backends/nvidia/compiler.py:190#passes.ttir.add_combine(pm))
    - 简单的重写 如 select(cond, load(ptrs, broadcast(cond), ???), other) => load(ptrs, broadcast(cond), other)
- [common.add_canonicalizer](python/triton/backends/nvidia/compiler.py:191#passes.common.add_canonicalizer(pm))
    - 规范化, 转换成更标准的形式, 里面其实可以做一些简单的恒等变换
- [ttir.add_reorder_broadcast](python/triton/backends/nvidia/compiler.py:192#passes.ttir.add_reorder_broadcast(pm))
- [common.add_cse](python/triton/backends/nvidia/compiler.py:193#passes.common.add_cse(pm))
- [common.add_licm](python/triton/backends/nvidia/compiler.py:194#passes.common.add_licm(pm)) # MLIR 的 LoopInvariantCodeMotion Pass ，将循环无关的变量挪到 forloop 外面, 这个 clang 编译 c++ 时候也有
- [common.add_symbol_dce](python/triton/backends/nvidia/compiler.py:195#passes.common.add_symbol_dce(pm))
- [ttir.add_loop_unroll](python/triton/backends/nvidia/compiler.py:196#passes.ttir.add_loop_unroll(pm))
    - [unrollFactor = getUnrollFactorOrDefault](lib/Dialect/Triton/Transforms/LoopUnroll.cpp:55#auto▪️unrollFactor▪️=▪️getUnrollFactorOrDefault(loop);) 获取 c++端的unroll factor
    - 直接用mlir 的 scf.unrooll来做 
    - epilog loop 加上 tt.num_stages = 1 后面就不要pipeline 了? why?
    
# make_ttgir
## [ttir.add_convert_to_ttgpuir](python/triton/backends/nvidia/compiler.py:215#passes.ttir.add_convert_to_ttgpuir(pm,▪️f"cuda:{capability}",▪️opt.num_warps,▪️32,▪️opt.num_ctas)▪️#▪️ConvertTritonToTritonGPU) ⭐️ # 将 Triton IR 转换为 TritonGPU IR，主要是增加 TritonGPU 特有的 layout
- 简单的 dialect 转换 pass
- 见下面的 #附录1
- [TritonGPUTypeConverter](lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:766#TritonGPUTypeConverter▪️typeConverter(context,▪️numWarps,▪️threadsPerWarp,⬇️)) 
    - [RankedTensorType](lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp:26#addConversion([this](RankedTensorType▪️tensorType)▪️->▪️RankedTensorType▪️{⬇️)) # 下面会用到
        - [getDefaultBlockedEncoding](lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp:33#getDefaultBlockedEncoding(this->context,▪️shape,▪️this->numWarps,⬇️))
            - [BlockedEncodingAttr::get](lib/Dialect/TritonGPU/IR/Dialect.cpp:517#triton::gpu::BlockedEncodingAttr::get(context,▪️shape,▪️sizePerThread,⬇️))
                - TOOD: 这里的默认 encoding 出来是什么?
                - 会有一段 ops.td映射的代码
                - [AttrBuilder<(ins "ArrayRef<int64_t>":$shape](include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:723#AttrBuilder<(ins▪️"ArrayRef<int64_t>":$shape,⬇️))
                - [默认直接加了 block的调用?](https://yunxiao.devops.xiaohongshu.com/cr/diff?mrId=2&projectId=48063&sourceProjectId=48063&diffVersionId=2075526&fromCommitId=c4b298efefc1f3d197129cd69be99d7687d1bede&toCommitId=24b5c6bbd9383d4b78285bc34ac8e7b73a556a4e&fileKey=00016_before_16_ConvertTritonToTritonGPU%20(convert-triton-to-tritongpu)%20(%27builtin.module%27%20operation).mlir) 
                    - #blocked = #triton_gpu.blocked<{
                        - sizePerThread = [1], 
                        - threadsPerWarp = [32], 
                        - warpsPerCTA = [8], 
                        - order = [0]}>
    - 其他的?
- [TritonGPUConversionTarget](lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:768#TritonGPUConversionTarget▪️target(*context,▪️typeConverter);)
- // rules
    - [GenericOpPattern](lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:35#template▪️<class▪️Op>▪️struct▪️GenericOpPattern▪️:▪️public▪️OpConversionPattern<Op>▪️{)
        - [convertTypes](lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:42#if▪️(failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),⬇️)⬇️)⬇️))
            - 使用上面的 type converter
- [setAttr(triton_gpu.num-warps)](lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:783#mod->setAttr(⬇️)) # 8  # 值都是python 传过来的
- [triton_gpu.num-warps](include/triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h:15#constexpr▪️static▪️char▪️AttrNumWarpsName[]▪️=▪️"triton_gpu.num-warps";) # 4
- [triton_gpu.num-ctas](include/triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h:16#constexpr▪️static▪️char▪️AttrNumCTAsName[]▪️=▪️"triton_gpu.num-ctas";) # 1 用户传的
- [triton_gpu.target](include/triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h:17#constexpr▪️static▪️char▪️AttrTargetName[]▪️=▪️"triton_gpu.target";) # "cuda:90"
- 举例, 给 module 加 attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
## [ttgpuir.add_coalesce](python/triton/backends/nvidia/compiler.py:220#passes.ttgpuir.add_coalesce(pm)▪️#▪️这个pass▪️也发生了变换▪️⭐️▪️⭐️▪️layout▪️convertions) ⭐️ # 重排 order，使得最大 contiguity 的维度排在最前面
- [ModuleAxisInfoAnalysis](lib/Dialect/TritonGPU/Transforms/Coalesce.cpp:159#ModuleAxisInfoAnalysis▪️axisInfoAnalysis(moduleOp);)
- 分析访问模式
- 找出一个更优的布局 L2, 然后替换原有 L1 到 L2
- 最后再转换L1, 依赖后面的 dce 来做消除?
## [ttgpuir.add_f32_dot_tc](python/triton/backends/nvidia/compiler.py:222#passes.ttgpuir.add_f32_dot_tc(pm))
- pass 用来添加 tensor core 转换的?
- Decompose fp32 `DotOp` instructions into 4 pointwise ops and 3 fp16 `DotOp`s to allow using TensorCores. See https://github.com/NVIDIA/cutlass/discussions/385
- tensor cor e的 trick
## [ttnvgpuir.add_plan_cta](third_party/nvidia/backend/compiler.py:224#nvidia.passes.ttnvgpuir.add_plan_cta(pm,▪️cluster_info))
  let summary = "plan CTA";

  let description = [{
    This pass computes and applies "optimized" CTA tilings to DotOp, ReduceOp
    and StoreLikeOps operations.
  }];
## [ttgpuir.add_remove_layout_conversions](python/triton/backends/nvidia/compiler.py:225#passes.ttgpuir.add_remove_layout_conversions(pm)▪️#▪️⭐️▪️⭐️▪️mlir▪️中把▪️layout▪️convertions▪️给移除掉了)
- The purpose of this pass is to rewrite the `ConvertLayoutOps` to reduce the number of operations and to prefer favorable layouts like `BlockedEncodingAttr` layout for "expensive" loads and stores (good for coalescing) and `NvidiaMmaEncodingAttr` otherwise (good for tensor ops).
## [add_optimize_thread_locality](python/triton/backends/nvidia/compiler.py:226#passes.ttgpuir.add_optimize_thread_locality(pm))
let summary = "Reduce the cost of synchronization between threads in an SM";
let description = [{ The aim of this pass is to reduce cross-thread communication for reduction operations, by adjusting the reduction size (or layout) to avoid splitting the reduction operation between multiple threads. Currently, this pass only optimizes reduction yielded by loop to be thread-local until after the loop completes. }];
test: test/TritonGPU/optimize-locality.mlir:1
TODO: 没看懂...


## [add_accelerate_matmul](python/triton/backends/nvidia/compiler.py:227#passes.ttgpuir.add_accelerate_matmul(pm)) TritonGPUAccelerateMatmul
- desc: Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators (e.g., Nvidia tensor cores) 
- todo: 具体
```py
# from  test/TritonGPU/accelerate-matmul.mlir:11
result1 = zeros(128, 64)
for i in range(8):
    temp = matmul(A1(128,64), B1(64,16))     # -> (128,16)
    result1 += matmul(temp, C1(16,64))       # -> (128,64) 原始代码
# ===========================>
A1_shared = load_to_shared_mem(A1)
B1_shared = load_to_shared_mem(B1)
C1_shared = load_to_shared_mem(C1)

# 第一个循环
result1 = zeros(128, 64)
for i in range(8):
    # 使用Tensor Core进行计算
    temp = convert_to_mma_layout(zeros(128, 16))
    temp = warp_group_dot(A1_shared, B1_shared, temp)
    temp = convert_to_blocked_layout(temp)
    temp_shared = allocate_in_shared(temp)
    
    result1 = convert_to_mma_layout(result1)
    result1 = warp_group_dot(temp_shared, C1_shared, result1)
    result1 = convert_to_blocked_layout(result1)
store(result1)

## mma layout
mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
```

## [add_optimize_dot_operands](python/triton/backends/nvidia/compiler.py:229#passes.ttgpuir.add_optimize_dot_operands(pm,▪️capability▪️>=▪️80))
- desc:Re-arranged layouts of tensors used as matrix multiplication operands so as to promote the use of hardware-accelerated transpositions
主要作用: 将layout 转换尽可能提前, 
```py
def matrix_mul(ptr_a, ptr_b, c):
    # 1. 加载数据
    a_i8 = load(ptr_a)    # 从内存加载int8数据
    b = load(ptr_b)       # 从内存加载f16数据
    # 2. 数据类型转换
    a_f8 = bitcast(a_i8)  # i8 -> f8E5M2
    a = fp_convert(a_f8)  # f8E5M2 -> f16
    # 3. 布局转换
    dot_a = convert_layout(a)  # ALR -> Av2k4
    dot_b = convert_layout(b)  # BLC -> Bv2k4
    # 4. 矩阵乘法
    result = dot(dot_a, dot_b, c)
    return result
# 优化后
def matrix_mul(ptr_a: Tensor[16,16,i8], ptr_b: Tensor[16,16,f16], c: Tensor[16,16,f32]) -> Tensor[16,16,f32]:
    a_i8 = tt.load(ptr_a)           # %0 = tt.load %arg0
    b_f16 = tt.load(ptr_b)          # %1 = tt.load %arg1
    # 2. A矩阵的处理流程
    # 首先转换布局到dot操作所需的格式
    a_i8_dot = convert_layout(a_i8) # %2 = triton_gpu.convert_layout %0
    # 然后进行位转换到f8E5M2
    a_f8 = bitcast(a_i8_dot)        # %3 = tt.bitcast %2
    # 最后转换到f16
    a_f16 = fp_convert(a_f8)        # %4 = tt.fp_to_fp %3
    
    # 3. B矩阵的处理
    # 直接转换布局到dot操作所需的格式
    b_converted = convert_layout(b_f16)  # %5 = triton_gpu.convert_layout %1
    
    # 4. 执行矩阵乘法
    result = tt.dot(a_f16, b_converted, c)  # %6 = tt.dot %4, %5, %arg2
    
    return result                          # tt.return %6
```
## [add_optimize_accumulator_init](python/triton/backends/nvidia/compiler.py:232#passes.ttgpuir.add_optimize_accumulator_init(pm)) CU8
- desc: For the dot operations that support accumulator-use flag this pass replaces the zero-initialization of the accumulator with the flag indicating the first use of the accumulator
## [add_combine_tensor_select_and_if](python/triton/backends/nvidia/compiler.py:233#passes.ttgpuir.add_combine_tensor_select_and_if(pm)) CU8
- desc: For select instruction that uses the same condidtion as the if instruction in the same block " "this pass combines the select into the if instruction, making the select operands returned by the " "then/else yields.
## [add_pipeline](python/triton/backends/nvidia/compiler.py:234#passes.ttgpuir.add_pipeline(pm,▪️opt.num_stages)) ⭐️ CU8
- desc: Applies software pipelining to loops in the module based on number of stages. This may convert some load into asynchronous loads, and multi-buffer the data.
- 附录1: MMA 指令对应的 global memory 到 shared memory 的 double buffer N-Buffer 优化 
- [void runOnOperation()](lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp:110#void▪️runOnOperation()▪️override▪️{) 
    - TODO: triton_gpu.alloc_tensor 
    - TODO: triton_gpu.insert_slice_async 
    - tensor.extract_slice 表示从 Tensor 中读取一个 slice
    - async_wait 的语义对应到 cp.async.wait_group 指令
```python
# 初始化
A_ptr = init_A_ptr(128x32)
B_ptr = init_B_ptr(32x128)
C = zeros(128x128)

# 简单的循环计算
for iv in range(lb, ub, step):
    # 1. 加载数据
    A = load(A_ptr)                     # 加载 128x32 数据
    A = convert_layout(A)               # 转换到dot操作布局
    
    B = load(B_ptr)                     # 加载 32x128 数据
    B = convert_layout(B)               # 转换到dot操作布局
    B = B * 4.0                         # 缩放B矩阵
    
    # 2. 计算
    C = dot(A, B, C)                   # 矩阵乘法累加
    
    # 3. 更新指针
    A_ptr += 4
    B_ptr += 4
return C
```

```python
# 初始化
A_buffer = allocate_local_buffer(2)     # 双缓冲
B_buffer = allocate_local_buffer(2)
C = zeros(128x128)

# 预取第一批数据
if lb < ub:
    async_copy(A_ptr -> A_buffer[0])    # 异步预取
    async_copy(B_ptr -> B_buffer[0])    # 异步预取

# 预取第二批数据
if lb + step < ub:
    async_copy(A_ptr+4 -> A_buffer[1])  # 异步预取下一块
    async_copy(B_ptr+4 -> B_buffer[1])  # 异步预取下一块

# 初始化索引
ins_idx = 1    # 插入索引（新数据写入位置）
ext_idx = 0    # 提取索引（当前计算数据位置）

# 流水线主循环
for iv in range(lb, ub, step):
    # 1. 等待当前数据就绪
    async_wait(2)    # 等待两个异步拷贝完成
    
    # 2. 计算当前数据
    A = local_load(A_buffer[ext_idx])
    B = local_load(B_buffer[ext_idx])
    B = B * 4.0
    C = dot(A, B, C)
    
    # 3. 启动下一轮预取
    next_ins_idx = (ins_idx + 1) % 2    # 循环使用缓冲区
    if iv + 2*step < ub:                # 检查是否需要预取
        async_copy(A_ptr + 8 -> A_buffer[next_ins_idx])
        async_copy(B_ptr + 8 -> B_buffer[next_ins_idx])
    
    # 4. 更新索引和指针
    ext_idx = (ext_idx + 1) % 2
    ins_idx = next_ins_idx
    A_ptr += 4
    B_ptr += 4

return C
```
## [add_prefetch](python/triton/backends/nvidia/compiler.py:235#passes.ttgpuir.add_prefetch(pm)) ⭐️ # createTritonGPUPrefetch MMA 指令对应的 shared memory 到 register file 的 N-Buffer 优化
This pass attempts to prefetch from shared memory the operands (A and B) of a `tt.dot`, when this operation is located in a loop. Decompose `DotOp` instructions in loops into several finer-grained `DotOp` that may have their operands constructed at the end of the previous iteration.
Transformations are performed in five different places:
  1. The pass emits a prologue to the loop where the data for the first
     loop iteration are prefetched.
  2. The loop arguments are extended with the new prefetched values.
  3. The dotOp parameters is updated with the new args.
  4. The prefetch operations for the next iteration are added to the loop.
  5. The yieldOp is updated by adding the prefetched values for the next
     iteration.
todo: 
- triton_gpu.convert_layout %37 : (tensor<16x16xf16, #blocked0>) -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> 
- ldmatrix
```

```
## [add_reduce_data_duplication](python/triton/backends/nvidia/compiler.py:238#passes.ttgpuir.add_reduce_data_duplication(pm))
  let summary = "Reduce data duplication in register by decomposing convert[distributed -> dotOperand] "
                "into convert[distributed -> shared -> dotOperand]";

  let description = "Decomposing conversions this way makes it possible to use CSE and reuse #shared tensors";
## [add_reorder_instructions](python/triton/backends/nvidia/compiler.py:239#passes.ttgpuir.add_reorder_instructions(pm))
  let summary = "Reorder instructions";

  let description = "This pass reorder instructions so as to (1) decrease register pressure (e.g., by moving "
                    "conversions from shared memory before their first use) and (2) promote LLVM instruction "
                    "order more friendly to `ptxas`.";
## [add_fence_insertion](python/triton/backends/nvidia/compiler.py:243#nvidia.passes.ttnvgpuir.add_fence_insertion(pm))
  let summary = "Insert fences across generic and async proxy";

  let description = [{
    This pass is to insert memory fences to ensure that memory operations are
    properly ordered across generic and async operations.
  }];
## [add_tma_lowering](python/triton/backends/nvidia/compiler.py:244#nvidia.passes.ttnvgpuir.add_tma_lowering(pm))
  let summary = "lower to TMA load/store operations";

  let description = [{
    Lower Triton experimental descriptor load to TMA load/store operations in TritonNvidiaGPUDialect.
  }];
# make_llir
- [ttgpuir.add_decompose_unsupported_conversions](python/triton/backends/nvidia/compiler.py:267#nvidia.passes.ttgpuir.add_decompose_unsupported_conversions(pm)) 
- [ttgpuir.add_combine_tensor_select_and_if](python/triton/backends/nvidia/compiler.py:268#passes.ttgpuir.add_combine_tensor_select_and_if(pm))
- [convert.add_scf_to_cf](python/triton/backends/nvidia/compiler.py:269#passes.convert.add_scf_to_cf(pm)) 
- [convert.add_index_to_llvmir](python/triton/backends/nvidia/compiler.py:270#passes.convert.add_index_to_llvmir(pm))
- [ttgpuir.add_allocate_shared_memory](python/triton/backends/nvidia/compiler.py:271#passes.ttgpuir.add_allocate_shared_memory(pm)▪️#▪️⭐️▪️⭐️▪️▪️简单添加了▪️shared▪️=▪️0) AllocateSharedMemory pass, 名字叫这个
    - 这个 pass 会标记 shared memory 的用量什么的,
    - 还有如何来做 swizzle 避免 bank conflict
    - e.g. 当前: triton_gpu.shared = 0 : i32,
## add_to_llvmir ⭐️ 真正lowering的 pass ⭐️ 主要分析一下 store op 做了什么?

举例: 
```mlir
    %c2_i32 = arith.constant 2 : i32 loc(#loc1) // y - 2 里面的constant
    %0 = tt.load %arg0 : !tt.ptr<i32> loc(#loc2) // tl.load(x_ptr)
    %1 = arith.subi %0, %c2_i32 : i32 loc(#loc3) // output = y - 2
    tt.store %arg1, %1 : !tt.ptr<i32> loc(#loc4) // tl.store(output, y_ptr)
    tt.return loc(#loc5) // tl.return()
```


```mlir
    %0 = llvm.mlir.constant(2 : i32) : i32 loc(#loc1) // 2
    // tl.load(x_ptr)
    %1 = llvm.mlir.constant(true) : i1 loc(#loc2) 
    %2 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b" %arg0, %1 : (!llvm.ptr<1>, i1) -> i32 loc(#loc2)
    %3 = llvm.bitcast %2 : i32 to vector<1xi32> loc(#loc2) 
    %4 = llvm.mlir.constant(0 : index) : i32 loc(#loc2) 
    %5 = llvm.extractelement %3[%4 : i32] : vector<1xi32> loc(#loc2)
    // output = y - 2
    %6 = llvm.sub %5, %0 : i32 loc(#loc3)
    // tl.store(output, y_ptr)
    %7 = llvm.mlir.constant(true) : i1 loc(#loc4)
    %8 = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc4)
    %9 = llvm.mlir.constant(0 : i32) : i32 loc(#loc4)
    %10 = nvgpu.cluster_id loc(#loc4)
    %11 = llvm.mlir.constant(0 : i32) : i32 loc(#loc4)
    %12 = llvm.icmp "eq" %9, %11 : i32 loc(#loc4)
    %13 = llvm.and %7, %12  : i1 loc(#loc4)
    %14 = llvm.mlir.constant(0 : i32) : i32 loc(#loc4)
    %15 = llvm.icmp "eq" %8, %14 : i32 loc(#loc4)
    %16 = llvm.and %13, %15  : i1 loc(#loc4)
    %17 = llvm.mlir.undef : vector<1xi32> loc(#loc4)
    %18 = llvm.bitcast %6 : i32 to i32 loc(#loc4)
    %19 = llvm.mlir.constant(0 : i32) : i32 loc(#loc4)
    %20 = llvm.insertelement %18, %17[%19 : i32] : vector<1xi32> loc(#loc4)
    %21 = llvm.bitcast %20 : vector<1xi32> to i32 loc(#loc4)
    %22 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b" %21, %arg1, %16 : (i32, !llvm.ptr<1>, i1) -> !llvm.void loc(#loc4)
    // tl.return
    llvm.return loc(#loc5)

```
- [add_to_llvmir](python/triton/backends/nvidia/compiler.py:272#nvidia.passes.ttgpuir.add_to_llvmir(pm,▪️capability,▪️ptx_version)▪️#▪️⭐️▪️⭐️▪️⭐️▪️⭐️▪️▪️▪️最终要的不分,▪️直接进行递降了▪️#▪️https://yunxiao.devops.xiaohongshu.com/cr/diff?mrId=2&projectId=48063&sourceProjectId=48063&diffVersionId=2075526&fromCommitId=c4b298efefc1f3d197129cd69be99d7687d1bede&toCommitId=24b5c6bbd9383d4b78285bc34ac8e7b73a556a4e&fileKey=00043_before_43_ConvertTritonGPUToLLVM%20%28convert-triton-gpu-to-llvm%29%20%28%27builtin.module%27%20operation%29.mlir) ⭐️ # ConvertTritonGPUToLLVM
    - [TritonGPUToLLVMTypeConverter](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:92#TritonGPUToLLVMTypeConverter▪️typeConverter(context,▪️option,▪️targetInfo);)
        - [addConversion([&](triton::PointerType type) ](lib/Conversion/TritonGPUToLLVM/TypeConverter.cpp:22#addConversion([&](triton::PointerType▪️type)▪️->▪️std::optional<Type>▪️{⬇️))
        - [addConversion([&](RankedTensorType type)](lib/Conversion/TritonGPUToLLVM/TypeConverter.cpp:25#addConversion([&](RankedTensorType▪️type)▪️->▪️std::optional<Type>▪️{⬇️)) 
        - ... 多种的类型转换
    - [TritonLLVMConversionTarget](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:93#TritonLLVMConversionTarget▪️convTarget(*context);)  真正的convertion target
        - [LLVMDialect](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:59#addLegalDialect<LLVM::LLVMDialect>();)
        - [NVVMDialect](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:60#addLegalDialect<NVVM::NVVMDialect>();) 
        - [NVGPUDialect](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:61#addLegalDialect<mlir::triton::nvgpu::NVGPUDialect>();) 
    - allocate shared memory and set barrier
        - [ModuleMembarAnalysis](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:100#ModuleMembarAnalysis▪️membarPass(&allocation,▪️NVIDIA::canSkipBarSync);) 分析 mem barrier
    - func conversion
        - [TritonLLVMFunctionConversionTarget](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:107#TritonLLVMFunctionConversionTarget▪️funcTarget(*context);) 
        - [populateFuncOpConversionPattern](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:109#mlir::triton::populateFuncOpConversionPattern(typeConverter,▪️funcPatterns,⬇️)) 
        - [mlir::cf::populateControlFlowToLLVMConversionPatterns](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:112#mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,⬇️)) ??
        - [applyPartialConversion](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:115#(☺️(☺️applyPartialConversion(mod,▪️funcTarget,▪️std::move(funcPatterns)))))
    - [initSharedMemory(typeConverter)](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:122#initSharedMemory(typeConverter);) 先跑让其他后面的 op 知道shared memory base address ptr 的位置
    - ... 好多 populateXXXopToLLVM
        -  举例: [populateLoadStoreOpToLLVMPatterns](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:142#populateLoadStoreOpToLLVMPatterns(typeConverter,▪️targetInfo,▪️patterns,⬇️))
            - [LoadOpConversion](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp:1290#AtomicRMWOpConversion,▪️LoadOpConversion,▪️StoreOpConversion>(⬇️))
                - [auto &ld = ptxBuilder.create<>("ld")](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp:310#auto▪️&ld▪️=▪️ptxBuilder.create<>("ld")) # 全部是内联的汇编
                    - 很多其他的阶段? 向量化, 线程访问计算 mask 等等
        - [populateArithToLLVMConversionPatterns](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:164#mlir::arith::populateArithToLLVMConversionPatterns(typeConverter,▪️patterns);) mlir 内部自带的
    - [applyPartialConversion](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:180#if▪️(failed(applyPartialConversion(mod,▪️convTarget,▪️std::move(patterns))))) 
    
## [add_nvgpu_to_llvm](python/triton/backends/nvidia/compiler.py:273#nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)▪️#▪️⭐️▪️⭐️▪️⭐️▪️⭐️▪️这个▪️pass▪️是用来把▪️nvgpu▪️的▪️dialect▪️给转换成▪️llvm▪️的▪️dialect▪️的▪️???▪️变了好多东西▪️#▪️https://yunxiao.devops.xiaohongshu.com/cr/diff?mrId=2&projectId=48063&sourceProjectId=48063&diffVersionId=2075526&fromCommitId=c4b298efefc1f3d197129cd69be99d7687d1bede&toCommitId=24b5c6bbd9383d4b78285bc34ac8e7b73a556a4e&fileKey=00044_before_44_ConvertNVGPUToLLVM%20(convert-nv-gpu-to-llvm)%20(%27builtin.module%27%20operation).mlir)
## [add_arith_to_llvmir](python/triton/backends/nvidia/compiler.py:274#passes.convert.add_arith_to_llvmir(pm))
- ....直接 lower 到 ir 
## [llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O4)](python/triton/backends/nvidia/compiler.py:302#llvm.optimize_module(llvm_mod,▪️llvm.OPTIMIZE_O3))






# 附录 1

## cp.async
安培架构引入
```python
// 基本格式
cp.async.cg.shared.global [addr_shared], [addr_global], [size];  // 异步拷贝
cp.async.commit_group;                                           // 提交一组异步拷贝
cp.async.wait_group [number];                                    // 等待若干组完成
cp.async.wait_all;                                               // 等待所有完成
```


## double buffer 双缓冲
``` python 
# 分配两倍大小的缓冲区（双缓冲）
A = alloc_tensor(shape=[2*16,16])  # 2倍大小用于双缓冲
B = alloc_tensor(shape=[2*16,8])   # 2倍大小用于双缓冲

# 预取第一组数据到缓冲区0
A = insert_slice_async(A, ptr0, 0)  # 异步加载到buffer 0
B = insert_slice_async(B, ptr1, 0)  # 异步加载到buffer 0

# 预取第二组数据到缓冲区1
A = insert_slice_async(A, ptr00, 1) # 异步加载到buffer 1
B = insert_slice_async(B, ptr11, 0) # 异步加载到buffer 1

async_wait(num=2) # cp.async.wait_group
A_slice0 = extract_slice(A, offset=(0,0,0), size=(1,16,16))
B_slice0 = extract_slice(B, offset=(0,0,0), size=(1,16,8))

for i in range(...):
    # 1. 使用当前缓冲区的数据进行计算
    a = ldmatrix(A_slice0)    # 从shared memory加载到寄存器
    b = ldmatrix(B_slice0)    # 从shared memory加载到寄存器
    c = dot(a, b)             # 执行计算

    # 2. 同时，异步加载下一轮数据到另一个缓冲区
    offset = (i+1) % 2        # 在0和1之间交替
    A = insert_slice_async(A, ptr2, offset)  # 异步加载下一批数据
    B = insert_slice_async(B, ptr3, offset)  # 异步加载下一批数据

    # 3. 等待异步加载完成
    async_wait(num=2)
    
    # 4. 切换到新加载的数据缓冲区
    A_slice0 = extract_slice(A, offset=(offset,0,0), size=(1,16,16))
    B_slice0 = extract_slice(B, offset=(offset,0,0), size=(1,16,8))
```


# 附录 2
## datalayout
1. Blocked Layout 表示 thread 间平均分配 workload 的情况，每个线程 own 一块 memory 上连续的 data 进行处理。
其包含了如下三个字段用于帮助确定 thread 和数据之间的映射关系：
-   sizePerThread：每个 thread 处理的 **连续排布** 的元素数目
-   threadsPerWarp：每个 Warp 在不同维度上的线程数，用向量表示
-   warpsPerCTA：每个 CTA 对应的 Warp 数目，这个由用户在 Python 层制定
```c++
Example 1, a row-major coalesced layout may partition a 16x16 tensor over 2 warps (i.e. 64 threads) as follows:

[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]

for

#triton_gpu.blocked_layout<{
  sizePerThread = {2, 2}
  threadsPerWarp = {8, 4}
  warpsPerCTA = {1, 2}
  CTAsPerCGA = {1, 1}
  CTASplitNum = {1, 1}
}
```
----------
2.  Shared Layout

Shared Layout：表示数据在 shared memory 的一些特性，比如 swizzle 访问的一些参数。

其包含了如下字段

-   vec, 支持 vectorization 的单位
-   perPhase, 每个 phase 包含多少个 vec
-   maxPhase, tensor 总共包含多少个 phase
-   order, axis 的次序

其中，vec, perPhase, maxPhase 是用于避免 bank conflict 的 swizzle 操作需要的参数。
------------
3. ... TODO




# ...
- triton
    - 语法
        - load一个 tensor
            - tl.arange(0, 8)
                - -> triton.language.core.tensor
            - tl.load(x_ptr + range, range < 5, 0)
                - range<5 -> 返回一个值只有 1/0的 mask
                - 类似 tf的 bollean mask?
        - load 一个二维 tensor,
            - i_range = tl.arange(0, 8)[:, None]
                - None 的地方添加一个, 1的 shape -> [8, 1] 的 shape
            - j_range = tl.arange(0, 4)[None, :]
                - [1,4]的  shape
            - range = i_range * 4 + j_range
                - -> [8,4]的一个 tensor,里面都是 index
            - mask = (i_range < 4) & (j_range <3) 
                - ->[8,4]的 tensor, 会自动进行 broudcast
            - tf.load(x_ptr + range, mask, 0)
        - store a one dim tensor
            - tl.store(z_ptr, 10, range < 5)
            - x = tl.load(x_ptr + offset)
            - tl.load(z_ptr + off_x, x + 10.0)
                - 可以直接 store 不同的值
        - use multiple blocks
            - pid = tl.programd_id(0)
            - range = itl.arange(0,8) + pid * 9
            - x = tl.load(x_ptr + range, range < 20
            - launch: demo4[(3,1,1)](x)
        - 好习惯
            - 对于tl.load(ptr + range, mask, value)
                - 提前写好range + mask
            - 对于tl.store(ptr + range, value, mask)
                - 提前写好 ptr, value 和 mask
        - 来做relu
            - tl.where(x>0, x,0
    - lowering 过程
        - source -> AST -> dialects
        - ttir -> ttgir -> ttllir ->backend
            - lowering 代码
            - ｜ @triton.jit
            - ｜ def addi_kernel(x_ptr,  # Pointer to first input vector.
            - ｜    output_ptr,  # Pointer to output vector.
            - ｜ n_elements,  # Size of the vector.
            - ｜ BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
            - ｜                ):
            - ｜ x = tl.load(x_ptr)
            - ｜ output = x + 1
            - ｜ tl.store(output_ptr, output)
            - ｜ src = triton.compiler.ASTSource(fn=addi_kernel, signature="*i32,*i32", constants={"n_elements": 1,"BLOCK_SIZE": 1})
            - ｜ ret = triton.compile(src, target=("cuda", 80))
            - ｜ for k in ['ttir', 'ttgir', 'llir']:
            - ｜ print(ret.asm[k])
        - ttgir -> ttllir
            - add_to_llvmir pass
                - 别名： ConvertNVGPUToLLVM
                - ｜ third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:70
                - converTarget
                    - LLVM + NVVM + NVGPU 三个dialect
- mlir
    - mlir 的ops组成
        - 返回值： %t_tensor 
        - opname： "toy.transpose"  
        - args：(%tensor)  
        - op's attributes： {inplace = ture}   
        - 签名：(tensor<2x3xf64>) ->  tensor<3x2xf64>
        - 位置： loc("example/xxx:12:1")
    - dialect
        - def Toy_dialect: Dialect
        - let name = "xxx"
        - let summary = "xxx"
        - let description = xxx
        - let cppNamespace = xxx
        - 生成直接使用 mlir-tbgen -gen-dialect-decls xxx/Ops.td -I{mlir_src_root}/inlcude/
        - load: context.loadDialect<ToyDialect>();
        - 会自动  load builtin 的 dialect Builtin Dialect - MLIR
    - ops:
        - class Toy_Op<string mnemonic, list<Trait> traits = []> :    Op<Toy_Dialect, mnemonic, traits>;
            - 给出了 name 和 traits
        - 生成： mlir-tbgen -gen-op-defs / -gen-op-decls
        - def xxx Op: Toy_Op<"name">
            - let arguments = (ins F64ElementsAttr:$vlaue)
            - $vlaue 会自动生成一个 xxxOp::value() 来方便拿到做一些变换
            - let result = (outs F64Tensor
            - let summary / description = xxx
                - add docs
            - let hasVerifier = 1
                - 会 gen 一个 verifier 出来，需要自己进行验证
            - let builds =
                - OpBuilder<(ins "DenseElementsAttr":$value),
                     - build(builder, result, value.getType(), value);
                - OpBuilder<(ins "double":$value)>
            - let hasCustomAssemblyFormat = 1;
                - 自己来写
                - xxxOp::print(mlir::OpAsmPrinter)
                - xxxOp::parse(mlir::OpAsmParse, mlir::OperationState)
                - 或者直接用一些魔法变量
                - let assemblyFormat = "$input attr-dict : type($input)"
            - let hasCanonicalizer
                - 创建然后添加
        - DDR (Declarative, pattern match rewirter)
            - 直接写 pass 变换，感觉很丑
            - RewritePatterns
    - partial lowering
        - DialectConversion Frameworks
            - - 
        - runOnOperation()
            - mlir::ConversionTarget target(getContext())
            - target.addLegalDialect<affine::AffineDialect...xxx>()
            - target.addIllegalDialect<ToyDialect>()
            - target.addDynamicallyLegalOp()....
                - 加条件 xxx ... 满足比如 inputs .. 
                - 优先级比前面的高，随便加在哪里都行
            - mlir::RewritePatternSet  patterns(&getContext())
            - patterns.add<..., xxxxLowering>(&getContext())
            - mlir::applyPartialConversion(getOperation(), target, patterns) ...
        - ConversionPattern
            - matchAndRewrite(Ops, operands, rewriter)...
            

代码生成: 

简单的 load, save, add
https://yunxiao.devops.xiaohongshu.com/cr/diff?mrId=2&projectId=48063&sourceProjectId=48063&fromCommitId=4fca481421a633bce8b4970f26bc9a967cd33122&toCommitId=381a0746aba7251482af6615f297204e31b192c5















这个测试文件主要验证 Triton GPU 的 coalesce（合并）优化 pass。让我逐个分析主要的测试用例：

1. 第一个测试用例（transpose 函数）:
- 测试矩阵转置操作中的内存访问模式优化
- CHECK 语句验证了优化器会将内存布局从原始的 blocked 布局转换为更优的行/列布局：
  - 对于 load 操作，转换为 row_layout (sizePerThread = [1, 4])
  - 对于 store 操作，转换为 col_layout (sizePerThread = [4, 1])
- 这种优化可以提高内存访问的合并度(coalescing)，从而提升性能

2. 第二个测试用例（load_tensors_two_types）:
- 测试处理不同数据类型(f32和f16)时的内存访问优化
- 验证优化器能够：
  - 将原始的 sizePerThread=[1] 优化为更大的块大小：
    - f32类型使用 sizePerThread=[4]
    - f16类型使用 sizePerThread=[8]
- 这种优化可以减少内存事务数量，提高内存带宽利用率

3. 第三个测试用例（另一个load_tensors_two_types）:
- 类似第二个测试，但所有store操作都是f16类型
- 验证在统一数据类型场景下的优化效果
- CHECK 确保所有操作都使用了更大的块大小 (sizePerThread=[8])

4. 第四个测试用例（test_3866）:
- 这是一个回归测试，用于验证 issue #3866 的修复
- 测试张量指针的加载操作

5. 第五个测试用例（test_5122）:
- 另一个回归测试，用于验证 issue #5122 的修复
- 测试控制流（if和for循环）相关的优化

总的来说，这个测试文件主要验证：
1. 内存访问模式的优化（提高合并度）
2. 不同数据类型下的块大小优化
3. 确保之前报告的bug得到修复
4. 验证优化器在各种场景下（矩阵转置、混合数据类型、控制流等）的正确性

这些优化的主要目标是提高 GPU 内存访问效率，通过合并内存访问和调整块大小来最大化内存带宽利用率。