#include <stdio.h>

#include <cnn_config.h>

#include <cnn.h>
#include <cnn_private.h>
#include <cnn_types.h>

#include "test.h"

#define INPUT_WIDTH 32
#define INPUT_HEIGHT 32
#define BATCH 7

#define EXPORT_PATH "test.xml"

#define print_mat_info(str, matVar)                                          \
    printf("%s: %dx%d, %p, %p\n", str, matVar.rows, matVar.cols, matVar.mat, \
           matVar.grad)

#ifdef CNN_WITH_CUDA
#define test_mat(matVar)                                                      \
    {                                                                         \
        size_t size = matVar.rows * matVar.cols * sizeof(float);              \
        cudaError_t cuRet = cudaMemset(matVar.mat, -1, size);                 \
        if (cuRet != cudaSuccess)                                             \
        {                                                                     \
            fprintf(                                                          \
                stderr,                                                       \
                "%s: cudaMemset(matVar.mat, 0, %lu) failed with error: %d\n", \
                #matVar, size, cuRet);                                        \
        }                                                                     \
                                                                              \
        if (matVar.grad != NULL)                                              \
        {                                                                     \
            cuRet = cudaMemset(matVar.grad, -1, size);                        \
            if (cuRet != cudaSuccess)                                         \
            {                                                                 \
                fprintf(stderr,                                               \
                        "%s: cudaMemset(matVar.grad, 0, %lu) failed with "    \
                        "error: %d\n",                                        \
                        #matVar, size, cuRet);                                \
            }                                                                 \
        }                                                                     \
    }
#else
#define test_mat(matVar)                                   \
    {                                                      \
        int _i;                                            \
        for (_i = 0; _i < matVar.rows * matVar.cols; _i++) \
        {                                                  \
            matVar.mat[_i] = -1;                           \
            if (matVar.grad != NULL)                       \
            {                                              \
                matVar.grad[_i] = 1;                       \
            }                                              \
        }                                                  \
    }
#endif

void check_cnn_arch(cnn_t cnn)
{
    int i;

    printf("Input image size: %dx%d\n\n", INPUT_WIDTH, INPUT_HEIGHT);

    for (i = 0; i < cnn->cfg.layers; i++)
    {
        printf("Layer %d config:\n", i);
        switch (cnn->cfg.layerCfg[i].type)
        {
            case CNN_LAYER_INPUT:
                printf("The layer is input\n");
                printf("Output size: %dx%d\n", cnn->layerList[i].outMat.width,
                       cnn->layerList[i].outMat.height);
                printf("Output channel: %d\n",
                       cnn->layerList[i].outMat.channel);

                print_mat_info("Output mat", cnn->layerList[i].outMat.data);
                test_mat(cnn->layerList[i].outMat.data);
                break;

            case CNN_LAYER_FC:
                printf("The layer is fully connected\n");
                printf("Size: %d\n", cnn->cfg.layerCfg[i].fc.size);
                printf("Output size: %dx%d\n", cnn->layerList[i].outMat.width,
                       cnn->layerList[i].outMat.height);
                printf("Output channel: %d\n",
                       cnn->layerList[i].outMat.channel);

                print_mat_info("Output mat", cnn->layerList[i].outMat.data);
                print_mat_info("Weight mat", cnn->layerList[i].fc.weight);
                print_mat_info("Bias mat", cnn->layerList[i].fc.bias);

                test_mat(cnn->layerList[i].outMat.data);
                test_mat(cnn->layerList[i].fc.weight);
                test_mat(cnn->layerList[i].fc.bias);
                break;

            case CNN_LAYER_ACTIV:
                printf("The layer is activation function\n");
                printf("ID: %d\n", cnn->cfg.layerCfg[i].activ.id);
                printf("Output size: %dx%d\n", cnn->layerList[i].outMat.width,
                       cnn->layerList[i].outMat.height);
                printf("Output channel: %d\n",
                       cnn->layerList[i].outMat.channel);

                print_mat_info("Output mat", cnn->layerList[i].outMat.data);
                print_mat_info("Grad mat", cnn->layerList[i].activ.gradMat);

                test_mat(cnn->layerList[i].outMat.data);
                test_mat(cnn->layerList[i].activ.gradMat);
                break;

            case CNN_LAYER_CONV:
                printf("The layer is convolution\n");
                printf("Dimension: %d\n", cnn->cfg.layerCfg[i].conv.dim);
                printf("Filter: %d\n", cnn->cfg.layerCfg[i].conv.filter);
                printf("Size: %d\n", cnn->cfg.layerCfg[i].conv.size);
                printf("Output size: %dx%d\n", cnn->layerList[i].outMat.width,
                       cnn->layerList[i].outMat.height);
                printf("Output channel: %d\n",
                       cnn->layerList[i].outMat.channel);

                print_mat_info("Output mat", cnn->layerList[i].outMat.data);
                print_mat_info("Kernel mat", cnn->layerList[i].conv.kernel);

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
                print_mat_info("Bias mat", cnn->layerList[i].conv.bias);
#endif

                test_mat(cnn->layerList[i].outMat.data);
                test_mat(cnn->layerList[i].conv.kernel);

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
                test_mat(cnn->layerList[i].conv.bias);
#endif
                break;

            case CNN_LAYER_POOL:
                printf("The layer is pooling\n");
                printf("Pooling type: %d\n",
                       cnn->cfg.layerCfg[i].pool.poolType);
                printf("Dimension: %d\n", cnn->cfg.layerCfg[i].pool.dim);
                printf("Size: %d\n", cnn->cfg.layerCfg[i].pool.size);
                printf("Output size: %dx%d\n", cnn->layerList[i].outMat.width,
                       cnn->layerList[i].outMat.height);
                printf("Output channel: %d\n",
                       cnn->layerList[i].outMat.channel);

                print_mat_info("Output mat", cnn->layerList[i].outMat.data);

                test_mat(cnn->layerList[i].outMat.data);
                break;

            case CNN_LAYER_DROP:
                printf("The layer is dropout\n");
                printf("Rate: %g\n", cnn->cfg.layerCfg[i].drop.rate);
                printf("Output size: %dx%d\n", cnn->layerList[i].outMat.width,
                       cnn->layerList[i].outMat.height);
                printf("Output channel: %d\n",
                       cnn->layerList[i].outMat.channel);
                print_mat_info("Output mat", cnn->layerList[i].outMat.data);
                test_mat(cnn->layerList[i].outMat.data);
                break;

            case CNN_LAYER_BN:
                printf("The layer is batchnorm\n");
                printf("rInit: %g\n", cnn->cfg.layerCfg[i].bn.rInit);
                printf("bInit: %g\n", cnn->cfg.layerCfg[i].bn.bInit);
                printf("Output size: %dx%d\n", cnn->layerList[i].outMat.width,
                       cnn->layerList[i].outMat.height);
                printf("Output channel: %d\n",
                       cnn->layerList[i].outMat.channel);

                print_mat_info("Output mat", cnn->layerList[i].outMat.data);

                test_mat(cnn->layerList[i].outMat.data);
                test_mat(cnn->layerList[i].bn.bnVar);
                test_mat(cnn->layerList[i].bn.srcShift);
                test_mat(cnn->layerList[i].bn.srcNorm);
                break;

            default:
                printf("Not a cnn layer config?!\n");
        }
        printf("\n");
    }
}

int main()
{
    cnn_config_t cfg = NULL;
    cnn_t cnn = NULL;

    test(cnn_init());
    test(cnn_config_create(&cfg));

    // Set config
    test(cnn_config_set_input_size(cfg, INPUT_WIDTH, INPUT_HEIGHT, 1));
    test(cnn_config_set_batch_size(cfg, BATCH));

    test(cnn_config_append_texture(cfg, CNN_SIGMOID, 9, 2.7183));
    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, 2, 3, 3));
    test(cnn_config_append_batchnorm(cfg, 1.0, 0.0));
    test(cnn_config_append_pooling(cfg, 2, CNN_POOL_MAX, 2));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, 2, 6, 3));
    test(cnn_config_append_batchnorm(cfg, 1.0, 0.0));
    test(cnn_config_append_pooling(cfg, 2, CNN_POOL_MAX, 2));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_full_connect(cfg, 16));
    test(cnn_config_append_dropout(cfg, 0.5));
    test(cnn_config_append_full_connect(cfg, 2));
    test(cnn_config_append_activation(cfg, CNN_SOFTMAX));

    // Create cnn
    test(cnn_create(&cnn, cfg));

    // Check cnn arch
    check_cnn_arch(cnn);

    cnn_zero_network(cnn);
    cnn_rand_network(cnn);

    printf("Press enter to continue...");
    getchar();

    test(cnn_export(cnn, EXPORT_PATH));

    // Cleanup
    cnn_delete(cnn);
    cnn_config_delete(cfg);

    cnn_deinit();

    return 0;
}
