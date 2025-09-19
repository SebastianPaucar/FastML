#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 784
#define N_INPUT_1_1 784
#define N_LAYER_3 64
#define N_LAYER_3 64
#define N_LAYER_3 64
#define N_LAYER_8 10


// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<43,23> h_dense_result_t;
typedef ap_fixed<16,6> h_dense_weight_t;
typedef ap_fixed<16,6> h_dense_bias_t;
typedef ap_uint<1> layer3_index;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<18,8> re_lu_table_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<39,19> result_t;
typedef ap_fixed<16,6> h_dense_1_weight_t;
typedef ap_fixed<16,6> h_dense_1_bias_t;
typedef ap_uint<1> layer8_index;


#endif
