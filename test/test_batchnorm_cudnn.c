#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>
#include <cnn_types.h>

#include "test.h"

#define cuda_run(func)                                              \
    {                                                               \
        cudaError_t __ret = func;                                   \
        if (__ret != cudaSuccess)                                   \
        {                                                           \
            fprintf(stderr, "%s(), %d: %s failed with error: %d\n", \
                    __FUNCTION__, __LINE__, #func, __ret);          \
            return -1;                                              \
        }                                                           \
    }

#define cudnn_run(func)                                             \
    {                                                               \
        cudnnStatus_t __ret = func;                                 \
        if (__ret != CUDNN_STATUS_SUCCESS)                          \
        {                                                           \
            fprintf(stderr, "%s(), %d: %s failed with error: %d\n", \
                    __FUNCTION__, __LINE__, #func, __ret);          \
            return -1;                                              \
        }                                                           \
    }

#define CH_IN 2
#define CH_OUT 2
#define IMG_WIDTH 4
#define IMG_HEIGHT 1
#define BATCH 2

int main()
{
    int size;
    union CNN_LAYER layer[3] = {0};

    // cuDNN init
    cudnnHandle_t cudnn;
    cudnn_run(cudnnCreate(&cudnn));

    float src[IMG_WIDTH * IMG_HEIGHT * CH_IN * BATCH] = {
        /* batch 1 */
        0.0, 0.1, 0.2, 0.3,  //
        0.2, 0.4, 0.6, 0.8,  //

        /* batch 2 */
        0.1, 0.2, 0.4, 0.8,   //
        -0.3, -0.2, 0.0, 0.4  //
    };

    float gradIn[IMG_WIDTH * IMG_HEIGHT * CH_IN * BATCH] = {
        /* batch 1 */
        0.3, 0.1, 0.4, 0.2,  //
        0.4, 0.2, 0.1, 0.3,  //

        /* batch 2 */
        -0.1, 0.3, -0.2, 0.4,  //
        -0.2, -0.1, 0.1, 0.2   //
    };

    cudnnTensorDescriptor_t srcTen;
    cudnn_run(cudnnCreateTensorDescriptor(&srcTen));
    cudnn_run(cudnnSetTensor4dDescriptor(             //
        srcTen, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,  //
        BATCH, CH_IN, IMG_HEIGHT, IMG_WIDTH));

    cudnnTensorDescriptor_t bnTen;
    cudnn_run(cudnnCreateTensorDescriptor(&bnTen));
    cudnn_run(
        cudnnDeriveBNTensorDescriptor(bnTen, srcTen, CUDNN_BATCHNORM_SPATIAL));

    cnn_config_t cfg = NULL;

    // Create cnn
    test(cnn_init());

    test(cnn_config_create(&cfg));
    test(cnn_config_set_batch_size(cfg, BATCH));
    test(cnn_config_set_input_size(cfg, IMG_WIDTH, IMG_HEIGHT, CH_IN));

    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_batchnorm(cfg, 0.87, 0.03, 0.001));

    // Print information
    print_img_msg("src:", src, IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch);
    print_img_msg("gradIn:", gradIn, IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch);

    // Allocate cnn layer
    test(cnn_layer_activ_alloc(
        &layer[1].activ, (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[1],
        IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch));
    test(cnn_layer_bn_alloc(&layer[2].bn,
                            (struct CNN_CONFIG_LAYER_BN*)&cfg->layerCfg[2],
                            IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch));

    // Copy memory
    size =
        sizeof(float) * layer[1].outMat.data.rows * layer[1].outMat.data.cols;
    memcpy_net(layer[1].outMat.data.mat, src, size);

    size =
        sizeof(float) * layer[2].outMat.data.rows * layer[2].outMat.data.cols;
    memcpy_net(layer[2].outMat.data.grad, gradIn, size);

    // Forward Training
    for (int i = 0; i < 2; i++)
    {
        float alpha = 1.0;
        float beta = 0.0;

        printf("***** Forward Training %d *****\n", i);
        // cnn_forward_bn(layer, cfg, 2);
        cudnn_run(cudnnBatchNormalizationForwardTraining(            //
            cudnn, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,           //
            srcTen, layer[1].outMat.data.mat,                        //
            srcTen, layer[2].outMat.data.mat,                        //
            bnTen, layer[2].bn.bnScale.mat, layer[2].bn.bnBias.mat,  //
            cfg->layerCfg[2].bn.expAvgFactor,                        //
            layer[2].bn.runMean.mat, layer[2].bn.runVar.mat,         //
            CNN_BN_EPS,                                              //
            layer[2].bn.saveMean.mat, layer[2].bn.saveVar.mat));

        print_img_net_msg("BatchNorm output:", layer[2].outMat.data.mat,
                          layer[2].outMat.width, layer[2].outMat.height,
                          layer[2].outMat.channel, cfg->batch);
    }

    // BP
    for (int i = 0; i < 2; i++)
    {
        float alpha = 1.0;
        float betaParam = 1.0;
        float betaGrad = 0.0;

        printf("***** BP %d *****\n", i);
        // cnn_backward_bn(layer, cfg, 2);
        cudnn_run(cudnnBatchNormalizationBackward(  //
            cudnn, CUDNN_BATCHNORM_SPATIAL,         //
            &alpha, &betaGrad, &alpha, &betaParam,  //
            srcTen, layer[1].outMat.data.mat,       //
            srcTen, layer[2].outMat.data.grad,      //
            srcTen, layer[1].outMat.data.grad,      //
            bnTen,                                  //
            layer[2].bn.bnScale.mat,                //
            layer[2].bn.bnScale.grad,               //
            layer[2].bn.bnBias.grad,                //
            CNN_BN_EPS,                             //
            layer[2].bn.saveMean.mat, layer[2].bn.saveVar.mat));

        print_img_net_msg(
            "BatchNorm layer gradient:", layer[2].outMat.data.grad,
            layer[2].outMat.width, layer[2].outMat.height,
            layer[2].outMat.channel, cfg->batch);

        print_img_net_msg("Previous layer gradient:", layer[1].outMat.data.grad,
                          layer[1].outMat.width, layer[1].outMat.height,
                          layer[1].outMat.channel, cfg->batch);

        print_img_net_msg("BatchNorm variable gradient:",
                          layer[2].bn.bnVar.grad, 2, CH_IN, 1, 1);

        print_img_net_msg("BatchNorm scale gradient:", layer[2].bn.bnScale.grad,
                          1, CH_IN, 1, 1);
        print_img_net_msg("BatchNorm bias gradient:", layer[2].bn.bnBias.grad,
                          1, CH_IN, 1, 1);
    }

    return 0;
}
