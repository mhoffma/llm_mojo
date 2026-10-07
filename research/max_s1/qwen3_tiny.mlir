module attributes {mo.device_info_mapping = {cpu = #M.device_info<"cpu", "cpu", "unknown", "cpu">}} {
  mo.graph @qwen3<total_seq_len, input_row_offsets_len, return_n_logits, total_num_pages, replica_0_batch_size, replica_0_max_num_pages>(%arg0: !mo.tensor<[total_seq_len], si64>, %arg1: !mo.tensor<[input_row_offsets_len], ui32>, %arg2: !mo.tensor<[return_n_logits], si64>, %arg3: !mo.buffer<[294387728], ui8>, %arg4: !mo.buffer<[total_num_pages, 2, 2, 128, 2, 16], f32>, %arg5: !mo.tensor<[replica_0_batch_size], ui32>, %arg6: !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32>, %arg7: !mo.tensor<[1], ui32>, %arg8: !mo.tensor<[1], ui32>, %arg9: !mo.tensor<[4], si64>) -> !mo.tensor<[add(input_row_offsets_len, -1), 256], f32> attributes {_kernel_library_paths = [], argument_names = ["input0", "input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9"], result_names = ["output0"]} {
    %0 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "norm.weight"} : !mo.tensor<[64], f32>
    %1 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.mlp.down_proj.weight"} : !mo.tensor<[64, 128], f32>
    %2 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.mlp.up_proj.weight"} : !mo.tensor<[128, 64], f32>
    %3 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.mlp.gate_proj.weight"} : !mo.tensor<[128, 64], f32>
    %4 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.post_attention_layernorm.weight"} : !mo.tensor<[64], f32>
    %5 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.self_attn.o_proj.weight"} : !mo.tensor<[64, 64], f32>
    %6 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.self_attn.k_norm.weight"} : !mo.tensor<[16], f32>
    %7 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.self_attn.q_norm.weight"} : !mo.tensor<[16], f32>
    %8 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.self_attn.v_proj.weight"} : !mo.tensor<[32, 64], f32>
    %9 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.self_attn.k_proj.weight"} : !mo.tensor<[32, 64], f32>
    %10 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.self_attn.q_proj.weight"} : !mo.tensor<[64, 64], f32>
    %11 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.input_layernorm.weight"} : !mo.tensor<[64], f32>
    %12 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.mlp.down_proj.weight"} : !mo.tensor<[64, 128], f32>
    %13 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.mlp.up_proj.weight"} : !mo.tensor<[128, 64], f32>
    %14 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.mlp.gate_proj.weight"} : !mo.tensor<[128, 64], f32>
    %15 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.post_attention_layernorm.weight"} : !mo.tensor<[64], f32>
    %16 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.self_attn.o_proj.weight"} : !mo.tensor<[64, 64], f32>
    %17 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.self_attn.k_norm.weight"} : !mo.tensor<[16], f32>
    %18 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.self_attn.q_norm.weight"} : !mo.tensor<[16], f32>
    %19 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.self_attn.v_proj.weight"} : !mo.tensor<[32, 64], f32>
    %20 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.self_attn.k_proj.weight"} : !mo.tensor<[32, 64], f32>
    %21 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.self_attn.q_proj.weight"} : !mo.tensor<[64, 64], f32>
    %22 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.input_layernorm.weight"} : !mo.tensor<[64], f32>
    %23 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "embed_tokens.weight"} : !mo.tensor<[256, 64], f32>
    %24 = mo.chain.create()
    %outputs, %outChain = mo.distributed.broadcast(%arg0, %arg3, %24) : (!mo.tensor<[total_seq_len], si64>, !mo.buffer<[294387728], ui8>, !mo.chain) -> (!mo.tensor<[total_seq_len], si64>, !mo.chain)
    %25 = rmo.slice(%23) {starts = #mosh<ape[0, 0]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape, stops = #mosh<ape[256, 64]> : !mosh.ape} : (!mo.tensor<[256, 64], f32>) -> !mo.tensor<[256, 64], f32>
    %26 = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
    %27 = rmo.greater_equal(%outputs, %26) : (!mo.tensor<[total_seq_len], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[total_seq_len], bool>
    %28 = mo.constant {value = #M.dense_array<256> : tensor<si64>} : !mo.tensor<[], si64>
    %29 = rmo.greater_equal(%outputs, %28) : (!mo.tensor<[total_seq_len], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[total_seq_len], bool>
    %30 = rmo.mo.not(%29) : (!mo.tensor<[total_seq_len], bool>) -> !mo.tensor<[total_seq_len], bool>
    %31 = rmo.and(%27, %30) : (!mo.tensor<[total_seq_len], bool>, !mo.tensor<[total_seq_len], bool>) -> !mo.tensor<[total_seq_len], bool>
    %32 = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
    %33 = rmo.sub(%outputs, %32) : (!mo.tensor<[total_seq_len], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[total_seq_len], si64>
    %34 = rmo.mul(%33, %31) : (!mo.tensor<[total_seq_len], si64>, !mo.tensor<[total_seq_len], bool>) -> !mo.tensor<[total_seq_len], si64>
    %35 = rmo.mo.gather(%25, %34) {axis = 0 : index} : (!mo.tensor<[256, 64], f32>, !mo.tensor<[total_seq_len], si64>) -> !mo.tensor<[total_seq_len, 64], f32>
    %36 = rmo.reshape(%31) {newShape = #mosh<ape[total_seq_len, 1]> : !mosh.ape} : (!mo.tensor<[total_seq_len], bool>) -> !mo.tensor<[total_seq_len, 1], bool>
    %37 = mo.cast(%36) : (!mo.tensor<[total_seq_len, 1], bool>) -> !mo.tensor<[total_seq_len, 1], f32>
    %38 = rmo.mul(%35, %37) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[total_seq_len, 1], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %outputs_0, %outChain_1 = mo.distributed.allreduce.sum(%38, %arg3, %outChain) {group_size = 1 : i64} : (!mo.tensor<[total_seq_len, 64], f32>, !mo.buffer<[294387728], ui8>, !mo.chain) -> (!mo.tensor<[total_seq_len, 64], f32>, !mo.chain)
    %39 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f64>} : !mo.tensor<[], f64>
    %40 = mo.constant {value = #M.dense_array<1.600000e+01> : tensor<f64>} : !mo.tensor<[], f64>
    %41 = mo.constant {value = #M.dense_array<2.000000e+00> : tensor<f64>} : !mo.tensor<[], f64>
    %42 = rmo.mo.range(%39, %40, %41) : (!mo.tensor<[], f64>, !mo.tensor<[], f64>, !mo.tensor<[], f64>) -> !mo.tensor<[8], f64>
    %43 = mo.constant {value = #M.dense_array<1.600000e+01> : tensor<f64>} : !mo.tensor<[], f64>
    %44 = rmo.div(%42, %43) : (!mo.tensor<[8], f64>, !mo.tensor<[], f64>) -> !mo.tensor<[8], f64>
    %45 = mo.constant {value = #M.dense_array<1.000000e+06> : tensor<f64>} : !mo.tensor<[], f64>
    %46 = rmo.pow(%45, %44) : (!mo.tensor<[], f64>, !mo.tensor<[8], f64>) -> !mo.tensor<[8], f64>
    %47 = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f64>} : !mo.tensor<[], f64>
    %48 = rmo.div(%47, %46) : (!mo.tensor<[], f64>, !mo.tensor<[8], f64>) -> !mo.tensor<[8], f64>
    %49 = mo.cast(%48) : (!mo.tensor<[8], f64>) -> !mo.tensor<[8], f32>
    %50 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %51 = mo.constant {value = #M.dense_array<2.560000e+02> : tensor<f32>} : !mo.tensor<[], f32>
    %52 = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %53 = rmo.mo.range(%50, %51, %52) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[256], f32>
    %54 = rmo.reshape(%53) {newShape = #mosh<ape[256, 1]> : !mosh.ape} : (!mo.tensor<[256], f32>) -> !mo.tensor<[256, 1], f32>
    %55 = rmo.reshape(%49) {newShape = #mosh<ape[1, 8]> : !mosh.ape} : (!mo.tensor<[8], f32>) -> !mo.tensor<[1, 8], f32>
    %56 = rmo.mul(%54, %55) : (!mo.tensor<[256, 1], f32>, !mo.tensor<[1, 8], f32>) -> !mo.tensor<[256, 8], f32>
    %57 = rmo.mo.cos(%56) : (!mo.tensor<[256, 8], f32>) -> !mo.tensor<[256, 8], f32>
    %58 = rmo.mo.sin(%56) : (!mo.tensor<[256, 8], f32>) -> !mo.tensor<[256, 8], f32>
    %59 = rmo.reshape(%57) {newShape = #mosh<ape[256, 8, 1]> : !mosh.ape} : (!mo.tensor<[256, 8], f32>) -> !mo.tensor<[256, 8, 1], f32>
    %60 = rmo.reshape(%58) {newShape = #mosh<ape[256, 8, 1]> : !mosh.ape} : (!mo.tensor<[256, 8], f32>) -> !mo.tensor<[256, 8, 1], f32>
    %61 = rmo.concat(%59, %60) {axis = -1 : index} : (!mo.tensor<[256, 8, 1], f32>, !mo.tensor<[256, 8, 1], f32>) -> !mo.tensor<[256, 8, 2], f32>
    %62 = rmo.reshape(%61) {newShape = #mosh<ape[256, 16]> : !mosh.ape} : (!mo.tensor<[256, 8, 2], f32>) -> !mo.tensor<[256, 16], f32>
    %outputs_2, %outChain_3 = mo.distributed.broadcast(%arg1, %arg3, %outChain_1) : (!mo.tensor<[input_row_offsets_len], ui32>, !mo.buffer<[294387728], ui8>, !mo.chain) -> (!mo.tensor<[input_row_offsets_len], ui32>, !mo.chain)
    %63 = mo.constant {value = #M.dense_array<0> : tensor<ui32>} : !mo.tensor<[], ui32>
    %64 = rmo.slice(%22) {starts = #mosh<ape[0]> : !mosh.ape, steps = #mosh<ape[1]> : !mosh.ape, stops = #mosh<ape[64]> : !mosh.ape} : (!mo.tensor<[64], f32>) -> !mo.tensor<[64], f32>
    %65 = mo.constant {value = #M.dense_array<9.99999997E-7> : tensor<f32>} : !mo.tensor<[], f32>
    %66 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %67 = mo.reduce.rms_norm(%outputs_0, %64, %65, %66) {multiply_before_cast = false} : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %68 = rmo.slice(%21) {starts = #mosh<ape[0, 0]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape, stops = #mosh<ape[64, 64]> : !mosh.ape} : (!mo.tensor<[64, 64], f32>) -> !mo.tensor<[64, 64], f32>
    %69 = rmo.slice(%20) {starts = #mosh<ape[0, 0]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape, stops = #mosh<ape[32, 64]> : !mosh.ape} : (!mo.tensor<[32, 64], f32>) -> !mo.tensor<[32, 64], f32>
    %70 = rmo.slice(%19) {starts = #mosh<ape[0, 0]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape, stops = #mosh<ape[32, 64]> : !mosh.ape} : (!mo.tensor<[32, 64], f32>) -> !mo.tensor<[32, 64], f32>
    %71 = rmo.concat(%68, %69, %70) {axis = 0 : index} : (!mo.tensor<[64, 64], f32>, !mo.tensor<[32, 64], f32>, !mo.tensor<[32, 64], f32>) -> !mo.tensor<[128, 64], f32>
    %72 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %73 = rmo.mo.transpose(%71, %72) : (!mo.tensor<[128, 64], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[64, 128], f32>
    %74 = rmo.matmul(%67, %73) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64, 128], f32>) -> !mo.tensor<[total_seq_len, 128], f32>
    %75 = mo.constant {value = #M.dense_array<64, 32, 32> : tensor<3xsi64>} : !mo.tensor<[3], si64>
    %76:3 = mo.split(%74, %75) {axis = 1 : index} : (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[3], si64>) -> (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[total_seq_len, 32], f32>, !mo.tensor<[total_seq_len, 32], f32>)
    %77 = rmo.reshape(%76#0) {newShape = #mosh<ape[total_seq_len, 4, 16]> : !mosh.ape} : (!mo.tensor<[total_seq_len, 64], f32>) -> !mo.tensor<[total_seq_len, 4, 16], f32>
    %78 = rmo.slice(%18) {starts = #mosh<ape[0]> : !mosh.ape, steps = #mosh<ape[1]> : !mosh.ape, stops = #mosh<ape[16]> : !mosh.ape} : (!mo.tensor<[16], f32>) -> !mo.tensor<[16], f32>
    %79 = mo.constant {value = #M.dense_array<9.99999997E-7> : tensor<f32>} : !mo.tensor<[], f32>
    %80 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %81 = mo.reduce.rms_norm(%77, %78, %79, %80) {multiply_before_cast = false} : (!mo.tensor<[total_seq_len, 4, 16], f32>, !mo.tensor<[16], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 4, 16], f32>
    %82 = rmo.reshape(%81) {newShape = #mosh<ape[total_seq_len, 64]> : !mosh.ape} : (!mo.tensor<[total_seq_len, 4, 16], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %83 = rmo.reshape(%76#1) {newShape = #mosh<ape[total_seq_len, 2, 16]> : !mosh.ape} : (!mo.tensor<[total_seq_len, 32], f32>) -> !mo.tensor<[total_seq_len, 2, 16], f32>
    %84 = rmo.slice(%17) {starts = #mosh<ape[0]> : !mosh.ape, steps = #mosh<ape[1]> : !mosh.ape, stops = #mosh<ape[16]> : !mosh.ape} : (!mo.tensor<[16], f32>) -> !mo.tensor<[16], f32>
    %85 = mo.constant {value = #M.dense_array<9.99999997E-7> : tensor<f32>} : !mo.tensor<[], f32>
    %86 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %87 = mo.reduce.rms_norm(%83, %84, %85, %86) {multiply_before_cast = false} : (!mo.tensor<[total_seq_len, 2, 16], f32>, !mo.tensor<[16], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 2, 16], f32>
    %88 = rmo.reshape(%87) {newShape = #mosh<ape[total_seq_len, 32]> : !mosh.ape} : (!mo.tensor<[total_seq_len, 2, 16], f32>) -> !mo.tensor<[total_seq_len, 32], f32>
    %89 = rmo.concat(%82, %88, %76#2) {axis = -1 : index} : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[total_seq_len, 32], f32>, !mo.tensor<[total_seq_len, 32], f32>) -> !mo.tensor<[total_seq_len, 128], f32>
    %90:2 = mo.custom {parameters = {interleaved = false}, symbol = "mo.rope_split_store.ragged.paged"}(%89, %outputs_2, %62, %arg4, %arg5, %arg6, %arg7, %arg8, %63, %outChain_3) : (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[input_row_offsets_len], ui32>, !mo.tensor<[256, 16], f32>, !mo.buffer<[total_num_pages, 2, 2, 128, 2, 16], f32>, !mo.tensor<[replica_0_batch_size], ui32>, !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[], ui32>, !mo.chain) -> (!mo.tensor<[total_seq_len, 64], f32>, !mo.chain)
    %91 = rmo.reshape(%90#0) {newShape = #mosh<ape[total_seq_len, 4, 16]> : !mosh.ape} : (!mo.tensor<[total_seq_len, 64], f32>) -> !mo.tensor<[total_seq_len, 4, 16], f32>
    %92 = mo.constant {value = #M.dense_array<2.500000e-01> : tensor<f32>} : !mo.tensor<[], f32>
    %93:2 = mo.custom {parameters = {local_window_size = -1 : index, mask_str = "causal"}, symbol = "mo.mha.ragged.paged"}(%91, %outputs_2, %arg4, %arg5, %arg6, %arg7, %arg8, %63, %92, %arg9, %90#1) : (!mo.tensor<[total_seq_len, 4, 16], f32>, !mo.tensor<[input_row_offsets_len], ui32>, !mo.buffer<[total_num_pages, 2, 2, 128, 2, 16], f32>, !mo.tensor<[replica_0_batch_size], ui32>, !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[], ui32>, !mo.tensor<[], f32>, !mo.tensor<[4], si64>, !mo.chain) -> (!mo.tensor<[total_seq_len, 4, 16], f32>, !mo.chain)
    %94 = rmo.reshape(%93#0) {newShape = #mosh<ape[total_seq_len, 64]> : !mosh.ape} : (!mo.tensor<[total_seq_len, 4, 16], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %95 = rmo.slice(%16) {starts = #mosh<ape[0, 0]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape, stops = #mosh<ape[64, 64]> : !mosh.ape} : (!mo.tensor<[64, 64], f32>) -> !mo.tensor<[64, 64], f32>
    %96 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %97 = rmo.mo.transpose(%95, %96) : (!mo.tensor<[64, 64], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[64, 64], f32>
    %98 = rmo.matmul(%94, %97) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64, 64], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %99 = rmo.add(%outputs_0, %98) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[total_seq_len, 64], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %100 = rmo.slice(%15) {starts = #mosh<ape[0]> : !mosh.ape, steps = #mosh<ape[1]> : !mosh.ape, stops = #mosh<ape[64]> : !mosh.ape} : (!mo.tensor<[64], f32>) -> !mo.tensor<[64], f32>
    %101 = mo.constant {value = #M.dense_array<9.99999997E-7> : tensor<f32>} : !mo.tensor<[], f32>
    %102 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %103 = mo.reduce.rms_norm(%99, %100, %101, %102) {multiply_before_cast = false} : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %104 = rmo.slice(%14) {starts = #mosh<ape[0, 0]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape, stops = #mosh<ape[128, 64]> : !mosh.ape} : (!mo.tensor<[128, 64], f32>) -> !mo.tensor<[128, 64], f32>
    %105 = rmo.slice(%13) {starts = #mosh<ape[0, 0]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape, stops = #mosh<ape[128, 64]> : !mosh.ape} : (!mo.tensor<[128, 64], f32>) -> !mo.tensor<[128, 64], f32>
    %106 = rmo.concat(%104, %105) {axis = 0 : index} : (!mo.tensor<[128, 64], f32>, !mo.tensor<[128, 64], f32>) -> !mo.tensor<[256, 64], f32>
    %107 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %108 = rmo.mo.transpose(%106, %107) : (!mo.tensor<[256, 64], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[64, 256], f32>
    %109 = rmo.matmul(%103, %108) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64, 256], f32>) -> !mo.tensor<[total_seq_len, 256], f32>
    %110 = mo.constant {value = #M.dense_array<128, 128> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %111:2 = mo.split(%109, %110) {axis = 1 : index} : (!mo.tensor<[total_seq_len, 256], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[total_seq_len, 128], f32>)
    %112 = rmo.mo.silu(%111#0) : (!mo.tensor<[total_seq_len, 128], f32>) -> !mo.tensor<[total_seq_len, 128], f32>
    %113 = rmo.mul(%112, %111#1) : (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[total_seq_len, 128], f32>) -> !mo.tensor<[total_seq_len, 128], f32>
    %114 = rmo.slice(%12) {starts = #mosh<ape[0, 0]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape, stops = #mosh<ape[64, 128]> : !mosh.ape} : (!mo.tensor<[64, 128], f32>) -> !mo.tensor<[64, 128], f32>
    %115 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %116 = rmo.mo.transpose(%114, %115) : (!mo.tensor<[64, 128], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[128, 64], f32>
    %117 = rmo.matmul(%113, %116) : (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[128, 64], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %118 = rmo.add(%99, %117) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[total_seq_len, 64], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %119 = mo.constant {value = #M.dense_array<1> : tensor<ui32>} : !mo.tensor<[], ui32>
    %120 = rmo.slice(%11) {starts = #mosh<ape[0]> : !mosh.ape, steps = #mosh<ape[1]> : !mosh.ape, stops = #mosh<ape[64]> : !mosh.ape} : (!mo.tensor<[64], f32>) -> !mo.tensor<[64], f32>
    %121 = mo.constant {value = #M.dense_array<9.99999997E-7> : tensor<f32>} : !mo.tensor<[], f32>
    %122 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %123 = mo.reduce.rms_norm(%118, %120, %121, %122) {multiply_before_cast = false} : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %124 = rmo.slice(%10) {starts = #mosh<ape[0, 0]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape, stops = #mosh<ape[64, 64]> : !mosh.ape} : (!mo.tensor<[64, 64], f32>) -> !mo.tensor<[64, 64], f32>
    %125 = rmo.slice(%9) {starts = #mosh<ape[0, 0]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape, stops = #mosh<ape[32, 64]> : !mosh.ape} : (!mo.tensor<[32, 64], f32>) -> !mo.tensor<[32, 64], f32>
    %126 = rmo.slice(%8) {starts = #mosh<ape[0, 0]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape, stops = #mosh<ape[32, 64]> : !mosh.ape} : (!mo.tensor<[32, 64], f32>) -> !mo.tensor<[32, 64], f32>
    %127 = rmo.concat(%124, %125, %126) {axis = 0 : index} : (!mo.tensor<[64, 64], f32>, !mo.tensor<[32, 64], f32>, !mo.tensor<[32, 64], f32>) -> !mo.tensor<[128, 64], f32>
    %128 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %129 = rmo.mo.transpose(%127, %128) : (!mo.tensor<[128, 64], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[64, 128], f32>
    %130 = rmo.matmul(%123, %129) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64, 128], f32>) -> !mo.tensor<[total_seq_len, 128], f32>
    %131 = mo.constant {value = #M.dense_array<64, 32, 32> : tensor<3xsi64>} : !mo.tensor<[3], si64>
    %132:3 = mo.split(%130, %131) {axis = 1 : index} : (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[3], si64>) -> (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[total_seq_len, 32], f32>, !mo.tensor<[total_seq_len, 32], f32>)
    %133 = rmo.reshape(%132#0) {newShape = #mosh<ape[total_seq_len, 4, 16]> : !mosh.ape} : (!mo.tensor<[total_seq_len, 64], f32>) -> !mo.tensor<[total_seq_len, 4, 16], f32>
    %134 = rmo.slice(%7) {starts = #mosh<ape[0]> : !mosh.ape, steps = #mosh<ape[1]> : !mosh.ape, stops = #mosh<ape[16]> : !mosh.ape} : (!mo.tensor<[16], f32>) -> !mo.tensor<[16], f32>
    %135 = mo.constant {value = #M.dense_array<9.99999997E-7> : tensor<f32>} : !mo.tensor<[], f32>
    %136 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %137 = mo.reduce.rms_norm(%133, %134, %135, %136) {multiply_before_cast = false} : (!mo.tensor<[total_seq_len, 4, 16], f32>, !mo.tensor<[16], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 4, 16], f32>
    %138 = rmo.reshape(%137) {newShape = #mosh<ape[total_seq_len, 64]> : !mosh.ape} : (!mo.tensor<[total_seq_len, 4, 16], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %139 = rmo.reshape(%132#1) {newShape = #mosh<ape[total_seq_len, 2, 16]> : !mosh.ape} : (!mo.tensor<[total_seq_len, 32], f32>) -> !mo.tensor<[total_seq_len, 2, 16], f32>
    %140 = rmo.slice(%6) {starts = #mosh<ape[0]> : !mosh.ape, steps = #mosh<ape[1]> : !mosh.ape, stops = #mosh<ape[16]> : !mosh.ape} : (!mo.tensor<[16], f32>) -> !mo.tensor<[16], f32>
    %141 = mo.constant {value = #M.dense_array<9.99999997E-7> : tensor<f32>} : !mo.tensor<[], f32>
    %142 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %143 = mo.reduce.rms_norm(%139, %140, %141, %142) {multiply_before_cast = false} : (!mo.tensor<[total_seq_len, 2, 16], f32>, !mo.tensor<[16], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 2, 16], f32>
    %144 = rmo.reshape(%143) {newShape = #mosh<ape[total_seq_len, 32]> : !mosh.ape} : (!mo.tensor<[total_seq_len, 2, 16], f32>) -> !mo.tensor<[total_seq_len, 32], f32>
    %145 = rmo.concat(%138, %144, %132#2) {axis = -1 : index} : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[total_seq_len, 32], f32>, !mo.tensor<[total_seq_len, 32], f32>) -> !mo.tensor<[total_seq_len, 128], f32>
    %146:2 = mo.custom {parameters = {interleaved = false}, symbol = "mo.rope_split_store.ragged.paged"}(%145, %outputs_2, %62, %arg4, %arg5, %arg6, %arg7, %arg8, %119, %93#1) : (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[input_row_offsets_len], ui32>, !mo.tensor<[256, 16], f32>, !mo.buffer<[total_num_pages, 2, 2, 128, 2, 16], f32>, !mo.tensor<[replica_0_batch_size], ui32>, !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[], ui32>, !mo.chain) -> (!mo.tensor<[total_seq_len, 64], f32>, !mo.chain)
    %147 = rmo.reshape(%146#0) {newShape = #mosh<ape[total_seq_len, 4, 16]> : !mosh.ape} : (!mo.tensor<[total_seq_len, 64], f32>) -> !mo.tensor<[total_seq_len, 4, 16], f32>
    %148 = mo.constant {value = #M.dense_array<2.500000e-01> : tensor<f32>} : !mo.tensor<[], f32>
    %149:2 = mo.custom {parameters = {local_window_size = -1 : index, mask_str = "causal"}, symbol = "mo.mha.ragged.paged"}(%147, %outputs_2, %arg4, %arg5, %arg6, %arg7, %arg8, %119, %148, %arg9, %146#1) : (!mo.tensor<[total_seq_len, 4, 16], f32>, !mo.tensor<[input_row_offsets_len], ui32>, !mo.buffer<[total_num_pages, 2, 2, 128, 2, 16], f32>, !mo.tensor<[replica_0_batch_size], ui32>, !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[], ui32>, !mo.tensor<[], f32>, !mo.tensor<[4], si64>, !mo.chain) -> (!mo.tensor<[total_seq_len, 4, 16], f32>, !mo.chain)
    %150 = rmo.reshape(%149#0) {newShape = #mosh<ape[total_seq_len, 64]> : !mosh.ape} : (!mo.tensor<[total_seq_len, 4, 16], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %151 = rmo.slice(%5) {starts = #mosh<ape[0, 0]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape, stops = #mosh<ape[64, 64]> : !mosh.ape} : (!mo.tensor<[64, 64], f32>) -> !mo.tensor<[64, 64], f32>
    %152 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %153 = rmo.mo.transpose(%151, %152) : (!mo.tensor<[64, 64], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[64, 64], f32>
    %154 = rmo.matmul(%150, %153) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64, 64], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %155 = rmo.add(%118, %154) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[total_seq_len, 64], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %156 = rmo.slice(%4) {starts = #mosh<ape[0]> : !mosh.ape, steps = #mosh<ape[1]> : !mosh.ape, stops = #mosh<ape[64]> : !mosh.ape} : (!mo.tensor<[64], f32>) -> !mo.tensor<[64], f32>
    %157 = mo.constant {value = #M.dense_array<9.99999997E-7> : tensor<f32>} : !mo.tensor<[], f32>
    %158 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %159 = mo.reduce.rms_norm(%155, %156, %157, %158) {multiply_before_cast = false} : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %160 = rmo.slice(%3) {starts = #mosh<ape[0, 0]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape, stops = #mosh<ape[128, 64]> : !mosh.ape} : (!mo.tensor<[128, 64], f32>) -> !mo.tensor<[128, 64], f32>
    %161 = rmo.slice(%2) {starts = #mosh<ape[0, 0]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape, stops = #mosh<ape[128, 64]> : !mosh.ape} : (!mo.tensor<[128, 64], f32>) -> !mo.tensor<[128, 64], f32>
    %162 = rmo.concat(%160, %161) {axis = 0 : index} : (!mo.tensor<[128, 64], f32>, !mo.tensor<[128, 64], f32>) -> !mo.tensor<[256, 64], f32>
    %163 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %164 = rmo.mo.transpose(%162, %163) : (!mo.tensor<[256, 64], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[64, 256], f32>
    %165 = rmo.matmul(%159, %164) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64, 256], f32>) -> !mo.tensor<[total_seq_len, 256], f32>
    %166 = mo.constant {value = #M.dense_array<128, 128> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %167:2 = mo.split(%165, %166) {axis = 1 : index} : (!mo.tensor<[total_seq_len, 256], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[total_seq_len, 128], f32>)
    %168 = rmo.mo.silu(%167#0) : (!mo.tensor<[total_seq_len, 128], f32>) -> !mo.tensor<[total_seq_len, 128], f32>
    %169 = rmo.mul(%168, %167#1) : (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[total_seq_len, 128], f32>) -> !mo.tensor<[total_seq_len, 128], f32>
    %170 = rmo.slice(%1) {starts = #mosh<ape[0, 0]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape, stops = #mosh<ape[64, 128]> : !mosh.ape} : (!mo.tensor<[64, 128], f32>) -> !mo.tensor<[64, 128], f32>
    %171 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %172 = rmo.mo.transpose(%170, %171) : (!mo.tensor<[64, 128], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[128, 64], f32>
    %173 = rmo.matmul(%169, %172) : (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[128, 64], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %174 = rmo.add(%155, %173) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[total_seq_len, 64], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %175 = rmo.slice(%outputs_2) {starts = #mosh<ape[1]> : !mosh.ape, steps = #mosh<ape[1]> : !mosh.ape, stops = #mosh<ape[input_row_offsets_len]> : !mosh.ape} : (!mo.tensor<[input_row_offsets_len], ui32>) -> !mo.tensor<[add(input_row_offsets_len, -1)], ui32>
    %176 = mo.constant {value = #M.dense_array<1> : tensor<ui32>} : !mo.tensor<[], ui32>
    %177 = rmo.sub(%175, %176) : (!mo.tensor<[add(input_row_offsets_len, -1)], ui32>, !mo.tensor<[], ui32>) -> !mo.tensor<[add(input_row_offsets_len, -1)], ui32>
    %178 = rmo.mo.gather(%174, %177) {axis = 0 : index} : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[add(input_row_offsets_len, -1)], ui32>) -> !mo.tensor<[add(input_row_offsets_len, -1), 64], f32>
    %179 = rmo.slice(%0) {starts = #mosh<ape[0]> : !mosh.ape, steps = #mosh<ape[1]> : !mosh.ape, stops = #mosh<ape[64]> : !mosh.ape} : (!mo.tensor<[64], f32>) -> !mo.tensor<[64], f32>
    %180 = mo.constant {value = #M.dense_array<9.99999997E-7> : tensor<f32>} : !mo.tensor<[], f32>
    %181 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %182 = mo.reduce.rms_norm(%178, %179, %180, %181) {multiply_before_cast = false} : (!mo.tensor<[add(input_row_offsets_len, -1), 64], f32>, !mo.tensor<[64], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[add(input_row_offsets_len, -1), 64], f32>
    %183 = rmo.slice(%23) {starts = #mosh<ape[0, 0]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape, stops = #mosh<ape[256, 64]> : !mosh.ape} : (!mo.tensor<[256, 64], f32>) -> !mo.tensor<[256, 64], f32>
    %184 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %185 = rmo.mo.transpose(%183, %184) : (!mo.tensor<[256, 64], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[64, 256], f32>
    %186 = rmo.matmul(%182, %185) : (!mo.tensor<[add(input_row_offsets_len, -1), 64], f32>, !mo.tensor<[64, 256], f32>) -> !mo.tensor<[add(input_row_offsets_len, -1), 256], f32>
    mo.output %186 : !mo.tensor<[add(input_row_offsets_len, -1), 256], f32>
  } {counter = 196 : i64}
}
