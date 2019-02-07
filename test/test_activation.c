#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>

#include "test.h"

#define SIZE 3

int main(int argc, char* argv[])
{
    int id = 0;
    int size;
    union CNN_LAYER layer[3];

    float src[SIZE] = {-1, 0, 1};
    float gradIn[SIZE] = {-1, 0, 1};

    cnn_config_t cfg = NULL;

    // Parse argument
    if (argc < 2)
    {
        printf("Usage: %s <activ_id>\n", argv[0]);
        return -1;
    }

    id = atoi(argv[1]);
    printf("Using activation id: %d\n\n", id);

    // Create cnn
    test(cnn_init());

    test(cnn_config_create(&cfg));
    test(cnn_config_set_input_size(cfg, SIZE, 1, 1));

    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_activation(cfg, id));

    // Print information
    printf("src:\n");
    print_img(src, SIZE, 1, 1);
    printf("\n");

    printf("gradIn:\n");
    print_img(gradIn, SIZE, 1, 1);
    printf("\n");

    // Allocate cnn layer
    test(cnn_layer_activ_alloc(
        &layer[1].activ, (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[1],
        SIZE, 1, 1, cfg->batch));
    test(cnn_layer_activ_alloc(
        &layer[2].activ, (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[2],
        SIZE, 1, 1, cfg->batch));

    // Copy memory
    size =
        sizeof(float) * layer[1].outMat.data.rows * layer[1].outMat.data.cols;
#ifdef CNN_WITH_CUDA
    cudaMemcpy(layer[1].outMat.data.mat, src, size, cudaMemcpyHostToDevice);
#else
    memcpy(layer[1].outMat.data.mat, src, size);
#endif

    size =
        sizeof(float) * layer[2].outMat.data.rows * layer[2].outMat.data.cols;
#ifdef CNN_WITH_CUDA
    cudaMemcpy(layer[2].outMat.data.grad, gradIn, size, cudaMemcpyHostToDevice);
#else
    memcpy(layer[2].outMat.data.grad, gradIn, size);
#endif

    // Forward
    printf("***** Forward *****\n");
    cnn_forward_activ(layer, cfg, 2);

    printf("Activation output:\n");
#ifdef CNN_WITH_CUDA
    print_img_cu(layer[2].outMat.data.mat, layer[2].outMat.width,
                 layer[2].outMat.height, layer[2].outMat.channel);
#else
    print_img(layer[2].outMat.data.mat, layer[2].outMat.width,
              layer[2].outMat.height, layer[2].outMat.channel);
#endif
    printf("\n");

    printf("***** Forward #2 *****\n");
    cnn_forward_activ(layer, cfg, 2);

    printf("Activation output:\n");
#ifdef CNN_WITH_CUDA
    print_img_cu(layer[2].outMat.data.mat, layer[2].outMat.width,
                 layer[2].outMat.height, layer[2].outMat.channel);
#else
    print_img(layer[2].outMat.data.mat, layer[2].outMat.width,
              layer[2].outMat.height, layer[2].outMat.channel);
#endif
    printf("\n");

    // BP
    printf("***** BP *****\n");
    cnn_backward_activ(layer, cfg, 2);

    printf("Activation layer gradient:\n");
#ifdef CNN_WITH_CUDA
    print_img_cu(layer[2].outMat.data.grad, layer[2].outMat.width,
                 layer[2].outMat.height, layer[2].outMat.channel);
#else
    print_img(layer[2].outMat.data.grad, layer[2].outMat.width,
              layer[2].outMat.height, layer[2].outMat.channel);
#endif
    printf("\n");

    printf("Previous layer gradient:\n");
#ifdef CNN_WITH_CUDA
    print_img_cu(layer[1].outMat.data.grad, layer[1].outMat.width,
                 layer[1].outMat.height, layer[1].outMat.channel);
#else
    print_img(layer[1].outMat.data.grad, layer[1].outMat.width,
              layer[1].outMat.height, layer[1].outMat.channel);
#endif
    printf("\n");

    printf("***** BP #2 *****\n");
    cnn_backward_activ(layer, cfg, 2);

    printf("Activation layer gradient:\n");
#ifdef CNN_WITH_CUDA
    print_img_cu(layer[2].outMat.data.grad, layer[2].outMat.width,
                 layer[2].outMat.height, layer[2].outMat.channel);
#else
    print_img(layer[2].outMat.data.grad, layer[2].outMat.width,
              layer[2].outMat.height, layer[2].outMat.channel);
#endif
    printf("\n");

    printf("Previous layer gradient:\n");
#ifdef CNN_WITH_CUDA
    print_img_cu(layer[1].outMat.data.grad, layer[1].outMat.width,
                 layer[1].outMat.height, layer[1].outMat.channel);
#else
    print_img(layer[1].outMat.data.grad, layer[1].outMat.width,
              layer[1].outMat.height, layer[1].outMat.channel);
#endif
    printf("\n");

    return 0;
}
