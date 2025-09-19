#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t input_1[N_INPUT_1_1],
    result_t layer8_out[N_LAYER_8]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_1,layer8_out 
    #pragma HLS PIPELINE

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<h_dense_weight_t, 50176>(w3, "w3.txt");
        nnet::load_weights_from_txt<h_dense_bias_t, 64>(b3, "b3.txt");
        nnet::load_weights_from_txt<h_dense_1_weight_t, 640>(w8, "w8.txt");
        nnet::load_weights_from_txt<h_dense_1_bias_t, 10>(b8, "b8.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_INPUT_1_1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::h_quantize<input_t, layer2_t>(input_1, layer2_out); // h_quantize

    h_dense_result_t layer3_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::dense<layer2_t, h_dense_result_t, config3>(layer2_out, layer3_out, w3, b3); // h_dense

    layer6_t layer6_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::relu<h_dense_result_t, layer6_t, ReLU_config6>(layer3_out, layer6_out); // re_lu

    layer7_t layer7_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::h_dense_relu_quantizer<layer6_t, layer7_t>(layer6_out, layer7_out); // h_dense_relu_quantizer

    nnet::dense<layer7_t, result_t, config8>(layer7_out, layer8_out, w8, b8); // h_dense_1

}

