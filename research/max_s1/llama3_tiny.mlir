module attributes {mo.device_info_mapping = {cpu = #M.device_info<"cpu", "cpu", "unknown", "cpu">}} {
  mo.graph @llama3<total_seq_len, input_row_offsets_len, return_n_logits, total_num_pages, replica_0_batch_size, replica_0_max_num_pages>(%arg0: !mo.tensor<[total_seq_len], si64>, %arg1: !mo.tensor<[input_row_offsets_len], ui32>, %arg2: !mo.tensor<[return_n_logits], si64>, %arg3: !mo.buffer<[total_num_pages, 2, 2, 128, 2, 16], f32>, %arg4: !mo.tensor<[replica_0_batch_size], ui32>, %arg5: !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32>, %arg6: !mo.tensor<[1], ui32>, %arg7: !mo.tensor<[1], ui32>, %arg8: !mo.tensor<[4], si64>) -> !mo.tensor<[add(input_row_offsets_len, -1), 256], f32> attributes {_kernel_library_paths = [], argument_names = ["input0", "input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8"], result_names = ["output0"]} {
    %0 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "lm_head.weight"} : !mo.tensor<[256, 64], f32>
    %1 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "norm.weight"} : !mo.tensor<[64], f32>
    %2 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.mlp.down_proj.weight"} : !mo.tensor<[64, 128], f32>
    %3 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.mlp.up_proj.weight"} : !mo.tensor<[128, 64], f32>
    %4 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.mlp.gate_proj.weight"} : !mo.tensor<[128, 64], f32>
    %5 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.post_attention_layernorm.weight"} : !mo.tensor<[64], f32>
    %6 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.self_attn.o_proj.weight"} : !mo.tensor<[64, 64], f32>
    %7 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.self_attn.v_proj.weight"} : !mo.tensor<[32, 64], f32>
    %8 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.self_attn.k_proj.weight"} : !mo.tensor<[32, 64], f32>
    %9 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.self_attn.q_proj.weight"} : !mo.tensor<[64, 64], f32>
    %10 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.1.input_layernorm.weight"} : !mo.tensor<[64], f32>
    %11 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.mlp.down_proj.weight"} : !mo.tensor<[64, 128], f32>
    %12 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.mlp.up_proj.weight"} : !mo.tensor<[128, 64], f32>
    %13 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.mlp.gate_proj.weight"} : !mo.tensor<[128, 64], f32>
    %14 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.post_attention_layernorm.weight"} : !mo.tensor<[64], f32>
    %15 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.self_attn.o_proj.weight"} : !mo.tensor<[64, 64], f32>
    %16 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.self_attn.v_proj.weight"} : !mo.tensor<[32, 64], f32>
    %17 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.self_attn.k_proj.weight"} : !mo.tensor<[32, 64], f32>
    %18 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.self_attn.q_proj.weight"} : !mo.tensor<[64, 64], f32>
    %19 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "layers.0.input_layernorm.weight"} : !mo.tensor<[64], f32>
    %20 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "embed_tokens.weight"} : !mo.tensor<[256, 64], f32>
    %21 = mo.chain.create()
    %22 = rmo.mo.gather(%20, %arg0) {axis = 0 : index} : (!mo.tensor<[256, 64], f32>, !mo.tensor<[total_seq_len], si64>) -> !mo.tensor<[total_seq_len, 64], f32>
    %23 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f64>} : !mo.tensor<[], f64>
    %24 = mo.constant {value = #M.dense_array<1.600000e+01> : tensor<f64>} : !mo.tensor<[], f64>
    %25 = mo.constant {value = #M.dense_array<2.000000e+00> : tensor<f64>} : !mo.tensor<[], f64>
    %26 = rmo.mo.range(%23, %24, %25) : (!mo.tensor<[], f64>, !mo.tensor<[], f64>, !mo.tensor<[], f64>) -> !mo.tensor<[8], f64>
    %27 = mo.constant {value = #M.dense_array<1.600000e+01> : tensor<f64>} : !mo.tensor<[], f64>
    %28 = rmo.div(%26, %27) : (!mo.tensor<[8], f64>, !mo.tensor<[], f64>) -> !mo.tensor<[8], f64>
    %29 = mo.constant {value = #M.dense_array<1.000000e+04> : tensor<f64>} : !mo.tensor<[], f64>
    %30 = rmo.pow(%29, %28) : (!mo.tensor<[], f64>, !mo.tensor<[8], f64>) -> !mo.tensor<[8], f64>
    %31 = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f64>} : !mo.tensor<[], f64>
    %32 = rmo.div(%31, %30) : (!mo.tensor<[], f64>, !mo.tensor<[8], f64>) -> !mo.tensor<[8], f64>
    %33 = mo.cast(%32) : (!mo.tensor<[8], f64>) -> !mo.tensor<[8], f32>
    %34 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %35 = mo.constant {value = #M.dense_array<2.560000e+02> : tensor<f32>} : !mo.tensor<[], f32>
    %36 = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %37 = rmo.mo.range(%34, %35, %36) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[256], f32>
    %38 = rmo.reshape(%37) {newShape = #mosh<ape[256, 1]> : !mosh.ape} : (!mo.tensor<[256], f32>) -> !mo.tensor<[256, 1], f32>
    %39 = rmo.reshape(%33) {newShape = #mosh<ape[1, 8]> : !mosh.ape} : (!mo.tensor<[8], f32>) -> !mo.tensor<[1, 8], f32>
    %40 = rmo.mul(%38, %39) : (!mo.tensor<[256, 1], f32>, !mo.tensor<[1, 8], f32>) -> !mo.tensor<[256, 8], f32>
    %41 = rmo.mo.cos(%40) : (!mo.tensor<[256, 8], f32>) -> !mo.tensor<[256, 8], f32>
    %42 = rmo.mo.sin(%40) : (!mo.tensor<[256, 8], f32>) -> !mo.tensor<[256, 8], f32>
    %43 = rmo.reshape(%41) {newShape = #mosh<ape[256, 8, 1]> : !mosh.ape} : (!mo.tensor<[256, 8], f32>) -> !mo.tensor<[256, 8, 1], f32>
    %44 = rmo.reshape(%42) {newShape = #mosh<ape[256, 8, 1]> : !mosh.ape} : (!mo.tensor<[256, 8], f32>) -> !mo.tensor<[256, 8, 1], f32>
    %45 = rmo.concat(%43, %44) {axis = -1 : index} : (!mo.tensor<[256, 8, 1], f32>, !mo.tensor<[256, 8, 1], f32>) -> !mo.tensor<[256, 8, 2], f32>
    %46 = rmo.reshape(%45) {newShape = #mosh<ape[256, 16]> : !mosh.ape} : (!mo.tensor<[256, 8, 2], f32>) -> !mo.tensor<[256, 16], f32>
    %47 = mo.constant {value = #M.dense_array<0> : tensor<ui32>} : !mo.tensor<[], ui32>
    %48 = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %49 = mo.constant {value = #M.dense_array<9.99999974E-6> : tensor<f32>} : !mo.tensor<[], f32>
    %50 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %51 = mo.reduce.rms_norm(%22, %19, %49, %50) {multiply_before_cast = false} : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %52 = rmo.concat(%18, %17, %16) {axis = 0 : index} : (!mo.tensor<[64, 64], f32>, !mo.tensor<[32, 64], f32>, !mo.tensor<[32, 64], f32>) -> !mo.tensor<[128, 64], f32>
    %53 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %54 = rmo.mo.transpose(%52, %53) : (!mo.tensor<[128, 64], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[64, 128], f32>
    %55 = rmo.matmul(%51, %54) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64, 128], f32>) -> !mo.tensor<[total_seq_len, 128], f32>
    %56:2 = mo.custom {parameters = {interleaved = false}, symbol = "mo.rope_split_store.ragged.paged"}(%55, %arg1, %46, %arg3, %arg4, %arg5, %arg6, %arg7, %47, %21) : (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[input_row_offsets_len], ui32>, !mo.tensor<[256, 16], f32>, !mo.buffer<[total_num_pages, 2, 2, 128, 2, 16], f32>, !mo.tensor<[replica_0_batch_size], ui32>, !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[], ui32>, !mo.chain) -> (!mo.tensor<[total_seq_len, 64], f32>, !mo.chain)
    %57 = rmo.reshape(%56#0) {newShape = #mosh<ape[total_seq_len, 4, 16]> : !mosh.ape} : (!mo.tensor<[total_seq_len, 64], f32>) -> !mo.tensor<[total_seq_len, 4, 16], f32>
    %58 = mo.constant {value = #M.dense_array<2.500000e-01> : tensor<f32>} : !mo.tensor<[], f32>
    %59:2 = mo.custom {parameters = {local_window_size = -1 : index, mask_str = "causal"}, symbol = "mo.mha.ragged.paged"}(%57, %arg1, %arg3, %arg4, %arg5, %arg6, %arg7, %47, %58, %arg8, %56#1) : (!mo.tensor<[total_seq_len, 4, 16], f32>, !mo.tensor<[input_row_offsets_len], ui32>, !mo.buffer<[total_num_pages, 2, 2, 128, 2, 16], f32>, !mo.tensor<[replica_0_batch_size], ui32>, !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[], ui32>, !mo.tensor<[], f32>, !mo.tensor<[4], si64>, !mo.chain) -> (!mo.tensor<[total_seq_len, 4, 16], f32>, !mo.chain)
    %60 = rmo.reshape(%59#0) {newShape = #mosh<ape[total_seq_len, 64]> : !mosh.ape} : (!mo.tensor<[total_seq_len, 4, 16], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %61 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %62 = rmo.mo.transpose(%15, %61) : (!mo.tensor<[64, 64], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[64, 64], f32>
    %63 = rmo.matmul(%60, %62) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64, 64], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %64 = rmo.add(%22, %63) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[total_seq_len, 64], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %65 = mo.constant {value = #M.dense_array<9.99999974E-6> : tensor<f32>} : !mo.tensor<[], f32>
    %66 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %67 = mo.reduce.rms_norm(%64, %14, %65, %66) {multiply_before_cast = false} : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %68 = rmo.concat(%13, %12) {axis = 0 : index} : (!mo.tensor<[128, 64], f32>, !mo.tensor<[128, 64], f32>) -> !mo.tensor<[256, 64], f32>
    %69 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %70 = rmo.mo.transpose(%68, %69) : (!mo.tensor<[256, 64], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[64, 256], f32>
    %71 = rmo.matmul(%67, %70) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64, 256], f32>) -> !mo.tensor<[total_seq_len, 256], f32>
    %72 = mo.constant {value = #M.dense_array<128, 128> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %73:2 = mo.split(%71, %72) {axis = 1 : index} : (!mo.tensor<[total_seq_len, 256], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[total_seq_len, 128], f32>)
    %74 = rmo.mo.silu(%73#0) : (!mo.tensor<[total_seq_len, 128], f32>) -> !mo.tensor<[total_seq_len, 128], f32>
    %75 = rmo.mul(%74, %73#1) : (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[total_seq_len, 128], f32>) -> !mo.tensor<[total_seq_len, 128], f32>
    %76 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %77 = rmo.mo.transpose(%11, %76) : (!mo.tensor<[64, 128], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[128, 64], f32>
    %78 = rmo.matmul(%75, %77) : (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[128, 64], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %79 = rmo.add(%64, %78) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[total_seq_len, 64], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %80 = mo.constant {value = #M.dense_array<1> : tensor<ui32>} : !mo.tensor<[], ui32>
    %81 = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %82 = mo.constant {value = #M.dense_array<9.99999974E-6> : tensor<f32>} : !mo.tensor<[], f32>
    %83 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %84 = mo.reduce.rms_norm(%79, %10, %82, %83) {multiply_before_cast = false} : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %85 = rmo.concat(%9, %8, %7) {axis = 0 : index} : (!mo.tensor<[64, 64], f32>, !mo.tensor<[32, 64], f32>, !mo.tensor<[32, 64], f32>) -> !mo.tensor<[128, 64], f32>
    %86 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %87 = rmo.mo.transpose(%85, %86) : (!mo.tensor<[128, 64], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[64, 128], f32>
    %88 = rmo.matmul(%84, %87) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64, 128], f32>) -> !mo.tensor<[total_seq_len, 128], f32>
    %89:2 = mo.custom {parameters = {interleaved = false}, symbol = "mo.rope_split_store.ragged.paged"}(%88, %arg1, %46, %arg3, %arg4, %arg5, %arg6, %arg7, %80, %59#1) : (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[input_row_offsets_len], ui32>, !mo.tensor<[256, 16], f32>, !mo.buffer<[total_num_pages, 2, 2, 128, 2, 16], f32>, !mo.tensor<[replica_0_batch_size], ui32>, !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[], ui32>, !mo.chain) -> (!mo.tensor<[total_seq_len, 64], f32>, !mo.chain)
    %90 = rmo.reshape(%89#0) {newShape = #mosh<ape[total_seq_len, 4, 16]> : !mosh.ape} : (!mo.tensor<[total_seq_len, 64], f32>) -> !mo.tensor<[total_seq_len, 4, 16], f32>
    %91 = mo.constant {value = #M.dense_array<2.500000e-01> : tensor<f32>} : !mo.tensor<[], f32>
    %92:2 = mo.custom {parameters = {local_window_size = -1 : index, mask_str = "causal"}, symbol = "mo.mha.ragged.paged"}(%90, %arg1, %arg3, %arg4, %arg5, %arg6, %arg7, %80, %91, %arg8, %89#1) : (!mo.tensor<[total_seq_len, 4, 16], f32>, !mo.tensor<[input_row_offsets_len], ui32>, !mo.buffer<[total_num_pages, 2, 2, 128, 2, 16], f32>, !mo.tensor<[replica_0_batch_size], ui32>, !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[], ui32>, !mo.tensor<[], f32>, !mo.tensor<[4], si64>, !mo.chain) -> (!mo.tensor<[total_seq_len, 4, 16], f32>, !mo.chain)
    %93 = rmo.reshape(%92#0) {newShape = #mosh<ape[total_seq_len, 64]> : !mosh.ape} : (!mo.tensor<[total_seq_len, 4, 16], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %94 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %95 = rmo.mo.transpose(%6, %94) : (!mo.tensor<[64, 64], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[64, 64], f32>
    %96 = rmo.matmul(%93, %95) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64, 64], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %97 = rmo.add(%79, %96) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[total_seq_len, 64], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %98 = mo.constant {value = #M.dense_array<9.99999974E-6> : tensor<f32>} : !mo.tensor<[], f32>
    %99 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %100 = mo.reduce.rms_norm(%97, %5, %98, %99) {multiply_before_cast = false} : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %101 = rmo.concat(%4, %3) {axis = 0 : index} : (!mo.tensor<[128, 64], f32>, !mo.tensor<[128, 64], f32>) -> !mo.tensor<[256, 64], f32>
    %102 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %103 = rmo.mo.transpose(%101, %102) : (!mo.tensor<[256, 64], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[64, 256], f32>
    %104 = rmo.matmul(%100, %103) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[64, 256], f32>) -> !mo.tensor<[total_seq_len, 256], f32>
    %105 = mo.constant {value = #M.dense_array<128, 128> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %106:2 = mo.split(%104, %105) {axis = 1 : index} : (!mo.tensor<[total_seq_len, 256], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[total_seq_len, 128], f32>)
    %107 = rmo.mo.silu(%106#0) : (!mo.tensor<[total_seq_len, 128], f32>) -> !mo.tensor<[total_seq_len, 128], f32>
    %108 = rmo.mul(%107, %106#1) : (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[total_seq_len, 128], f32>) -> !mo.tensor<[total_seq_len, 128], f32>
    %109 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %110 = rmo.mo.transpose(%2, %109) : (!mo.tensor<[64, 128], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[128, 64], f32>
    %111 = rmo.matmul(%108, %110) : (!mo.tensor<[total_seq_len, 128], f32>, !mo.tensor<[128, 64], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %112 = rmo.add(%97, %111) : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[total_seq_len, 64], f32>) -> !mo.tensor<[total_seq_len, 64], f32>
    %113 = rmo.slice(%arg1) {starts = #mosh<ape[1]> : !mosh.ape, steps = #mosh<ape[1]> : !mosh.ape, stops = #mosh<ape[input_row_offsets_len]> : !mosh.ape} : (!mo.tensor<[input_row_offsets_len], ui32>) -> !mo.tensor<[add(input_row_offsets_len, -1)], ui32>
    %114 = mo.constant {value = #M.dense_array<1> : tensor<ui32>} : !mo.tensor<[], ui32>
    %115 = rmo.sub(%113, %114) : (!mo.tensor<[add(input_row_offsets_len, -1)], ui32>, !mo.tensor<[], ui32>) -> !mo.tensor<[add(input_row_offsets_len, -1)], ui32>
    %116 = rmo.mo.gather(%112, %115) {axis = 0 : index} : (!mo.tensor<[total_seq_len, 64], f32>, !mo.tensor<[add(input_row_offsets_len, -1)], ui32>) -> !mo.tensor<[add(input_row_offsets_len, -1), 64], f32>
    %117 = mo.constant {value = #M.dense_array<9.99999974E-6> : tensor<f32>} : !mo.tensor<[], f32>
    %118 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %119 = mo.reduce.rms_norm(%116, %1, %117, %118) {multiply_before_cast = false} : (!mo.tensor<[add(input_row_offsets_len, -1), 64], f32>, !mo.tensor<[64], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[add(input_row_offsets_len, -1), 64], f32>
    %120 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %121 = rmo.mo.transpose(%0, %120) : (!mo.tensor<[256, 64], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[64, 256], f32>
    %122 = rmo.matmul(%119, %121) : (!mo.tensor<[add(input_row_offsets_len, -1), 64], f32>, !mo.tensor<[64, 256], f32>) -> !mo.tensor<[add(input_row_offsets_len, -1), 256], f32>
    mo.output %122 : !mo.tensor<[add(input_row_offsets_len, -1), 256], f32>
  } {counter = 68 : i64}
}
