#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cudnn.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>

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

#define KERNEL_SIZE 3
#define CH_IN 3
#define CH_OUT 2
#define IMG_WIDTH 4
#define IMG_HEIGHT 4
#define BATCH 2

int main()
{
    int size;
    union CNN_LAYER layer[3];

    // cuDNN init
    cudnnHandle_t cudnn;
    cudnn_run(cudnnCreate(&cudnn));

    float src[IMG_WIDTH * IMG_HEIGHT * CH_IN * BATCH] = {
        /* batch 1 */
        1, 0, 3, 0,  //
        0, 3, 0, 4,  //
        4, 0, 3, 0,  //
        3, 1, 0, 4,  //

        2, 3, 2, 2,  //
        0, 3, 3, 0,  //
        0, 2, 2, 4,  //
        2, 1, 4, 0,  //

        4, 0, 0, 4,  //
        3, 2, 0, 2,  //
        3, 1, 4, 3,  //
        4, 1, 4, 1,  //

        /* batch 2 */
        3, 1, 3, 1,  //
        1, 2, 4, 2,  //
        4, 2, 2, 4,  //
        2, 1, 3, 3,  //

        0, 1, 4, 4,  //
        0, 1, 3, 3,  //
        1, 2, 1, 1,  //
        0, 4, 2, 0,  //

        0, 1, 2, 1,  //
        4, 3, 4, 4,  //
        1, 1, 3, 0,  //
        3, 4, 4, 2   //
    };

    cudnnTensorDescriptor_t srcTen;
    cudnn_run(cudnnCreateTensorDescriptor(&srcTen));
    cudnn_run(cudnnSetTensor4dDescriptor(             //
        srcTen, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,  //
        BATCH, CH_IN, IMG_HEIGHT, IMG_WIDTH));

    float kernel[CH_OUT * CH_IN * KERNEL_SIZE * KERNEL_SIZE] = {
        0, 3, 4, 0, 1, 2, 2, 4, 1,  //
        1, 3, 0, 0, 2, 0, 4, 4, 2,  //
        0, 4, 4, 1, 3, 4, 2, 4, 4,  //

        3, 1, 0, 4, 4, 3, 0, 1, 1,  //
        3, 2, 3, 0, 4, 2, 1, 2, 0,  //
        2, 0, 2, 1, 0, 3, 1, 4, 4   //
    };

    cudnnFilterDescriptor_t kernelTen;
    cudnn_run(cudnnCreateFilterDescriptor(&kernelTen));
    cudnn_run(cudnnSetFilter4dDescriptor(                //
        kernelTen, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,  //
        CH_OUT, CH_IN, KERNEL_SIZE, KERNEL_SIZE));

    float bias[CH_OUT] = {
        1, 2  //
    };

    cudnnTensorDescriptor_t biasTen;
    cudnn_run(cudnnCreateTensorDescriptor(&biasTen));
    cudnn_run(cudnnSetTensor4dDescriptor(              //
        biasTen, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,  //
        1, CH_OUT, 1, 1));

    float gradIn[IMG_WIDTH * IMG_HEIGHT * CH_OUT * BATCH] = {
        /* batch 1 */
        0, 0, 0, 0,  //
        0, 3, 1, 0,  //
        0, 4, 2, 0,  //
        0, 0, 0, 0,  //

        0, 0, 0, 0,  //
        0, 4, 4, 0,  //
        0, 4, 4, 0,  //
        0, 0, 0, 0,  //

        /* batch 2 */
        2, 4, 1, 1,  //
        4, 4, 3, 1,  //
        4, 2, 3, 4,  //
        0, 1, 4, 3,  //

        2, 1, 4, 0,  //
        2, 3, 2, 0,  //
        3, 1, 1, 1,  //
        0, 2, 2, 3   //
    };

    // cuDNN convolution descriptor
    cudnnConvolutionDescriptor_t convDesc;
    cudnn_run(cudnnCreateConvolutionDescriptor(&convDesc));
    cudnn_run(cudnnSetConvolution2dDescriptor(         //
        convDesc,                                      //
        KERNEL_SIZE / 2, KERNEL_SIZE / 2, 1, 1, 1, 1,  //
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // cuDNN output tensor
    int outN, outC, outH, outW;
    cudnn_run(cudnnGetConvolution2dForwardOutputDim(
        convDesc, srcTen, kernelTen, &outN, &outC, &outH, &outW));
    printf("outN: %d, outC: %d, outH: %d, outW: %d\n", outN, outC, outH, outW);

    cudnnTensorDescriptor_t outTen;
    cudnn_run(cudnnCreateTensorDescriptor(&outTen));
    cudnn_run(cudnnSetTensor4dDescriptor(             //
        outTen, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,  //
        outN, outC, outH, outW));

    // cuDNN convolution algorithm
    cudnnConvolutionFwdAlgo_t convAlgoFW;
    cudnn_run(cudnnGetConvolutionForwardAlgorithm(
        cudnn,                                //
        srcTen, kernelTen, convDesc, outTen,  //
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convAlgoFW));

    cudnnConvolutionBwdFilterAlgo_t convAlgoBWFilter;
    cudnn_run(cudnnGetConvolutionBackwardFilterAlgorithm(
        cudnn,                                //
        srcTen, outTen, convDesc, kernelTen,  //
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &convAlgoBWFilter));

    cudnnConvolutionBwdDataAlgo_t convAlgoBWGrad;
    cudnn_run(cudnnGetConvolutionBackwardDataAlgorithm(
        cudnn,                                //
        kernelTen, outTen, convDesc, srcTen,  //
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &convAlgoBWGrad));

    // cuDNN convolution workspace
    size_t wsSize = 0;
    size_t sizeTmp;

    cudnn_run(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, srcTen, kernelTen, convDesc, outTen, convAlgoFW, &sizeTmp));
    if (sizeTmp > wsSize) wsSize = sizeTmp;

    cudnn_run(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn, srcTen, outTen, convDesc, kernelTen, convAlgoBWFilter,
        &sizeTmp));
    if (sizeTmp > wsSize) wsSize = sizeTmp;

    cudnn_run(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn, kernelTen, outTen, convDesc, srcTen, convAlgoBWGrad, &sizeTmp));
    if (sizeTmp > wsSize) wsSize = sizeTmp;

    float* wsData = NULL;
    cuda_run(cudaMalloc((void**)&wsData, wsSize));
    printf("Convolution workspace size: %lu\n", wsSize);

    cnn_config_t cfg = NULL;

    // Create cnn
    test(cnn_init());

    test(cnn_config_create(&cfg));
    test(cnn_config_set_batch_size(cfg, BATCH));
    test(cnn_config_set_input_size(cfg, IMG_WIDTH, IMG_HEIGHT, CH_IN));

    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_convolution(cfg, CNN_PAD_SAME, CNN_DIM_2D, CH_OUT,
                                       KERNEL_SIZE));

    // Print information
    print_img_msg("src:", src, IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch);
    print_img_msg("kernel:", kernel, KERNEL_SIZE, KERNEL_SIZE * CH_IN, CH_OUT,
                  1);
    print_img_msg("bias:", bias, 1, 1, CH_OUT, 1);
    print_img_msg("gradIn", gradIn, IMG_WIDTH, IMG_HEIGHT, CH_OUT, cfg->batch);

    // Allocate cnn layer
    test(cnn_layer_activ_alloc(
        &layer[1].activ, (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[1],
        IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch));
    test(cnn_layer_conv_alloc(&layer[2].conv,
                              (struct CNN_CONFIG_LAYER_CONV*)&cfg->layerCfg[2],
                              IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch));

    // Copy memory
    size =
        sizeof(float) * layer[1].outMat.data.rows * layer[1].outMat.data.cols;
    memcpy_net(layer[1].outMat.data.mat, src, size);

    size =
        sizeof(float) * layer[2].conv.kernel.rows * layer[2].conv.kernel.cols;
    memcpy_net(layer[2].conv.kernel.mat, kernel, size);

    size = sizeof(float) * layer[2].conv.bias.rows * layer[2].conv.bias.cols;
    memcpy_net(layer[2].conv.bias.mat, bias, size);

    size =
        sizeof(float) * layer[2].outMat.data.rows * layer[2].outMat.data.cols;
    memcpy_net(layer[2].outMat.data.grad, gradIn, size);

    // Forward
    for (int i = 0; i < 2; i++)
    {
        float alpha = 1.0;
        float beta = 0.0;

        printf("***** Forward #%d *****\n", i + 1);
        // cnn_forward_conv(layer, cfg, 2);

        cudnn_run(
            cudnnConvolutionForward(cudnn,                                 //
                                    &alpha,                                //
                                    srcTen, layer[1].outMat.data.mat,      //
                                    kernelTen, layer[2].conv.kernel.mat,   //
                                    convDesc, convAlgoFW, wsData, wsSize,  //
                                    &beta,                                 //
                                    outTen, layer[2].outMat.data.mat));

        beta = 1.0;
        cudnn_run(cudnnAddTensor(cudnn,                            //
                                 &alpha,                           //
                                 biasTen, layer[2].conv.bias.mat,  //
                                 &beta,                            //
                                 outTen, layer[2].outMat.data.mat));

        print_img_net_msg("Convolution output:", layer[2].outMat.data.mat,
                          layer[2].outMat.width, layer[2].outMat.height,
                          layer[2].outMat.channel, cfg->batch);
    }

    // BP
    for (int i = 0; i < 2; i++)
    {
        float alpha = 1.0;
        float beta = 1.0;

        printf("***** BP #%d *****\n", i + 1);
        // cnn_backward_conv(layer, cfg, 2);

        cudnn_run(cudnnConvolutionBackwardFilter(
            cudnn,                                       //
            &alpha,                                      //
            srcTen, layer[1].outMat.data.mat,            //
            outTen, layer[2].outMat.data.grad,           //
            convDesc, convAlgoBWFilter, wsData, wsSize,  //
            &beta,                                       //
            kernelTen, layer[2].conv.kernel.grad));

        cudnn_run(
            cudnnConvolutionBackwardBias(cudnn,                              //
                                         &alpha,                             //
                                         outTen, layer[2].outMat.data.grad,  //
                                         &beta,                              //
                                         biasTen, layer[2].conv.bias.grad));

        beta = 0.0;
        cudnn_run(cudnnConvolutionBackwardData(
            cudnn,                                     //
            &alpha,                                    //
            kernelTen, layer[2].conv.kernel.mat,       //
            outTen, layer[2].outMat.data.grad,         //
            convDesc, convAlgoBWGrad, wsData, wsSize,  //
            &beta,                                     //
            srcTen, layer[1].outMat.data.grad));

        print_img_net_msg(
            "Convolution layer gradient:", layer[2].outMat.data.grad,
            layer[2].outMat.width, layer[2].outMat.height,
            layer[2].outMat.channel, cfg->batch);
        print_img_net_msg("Previous layer gradient:", layer[1].outMat.data.grad,
                          layer[1].outMat.width, layer[1].outMat.height,
                          layer[1].outMat.channel, cfg->batch);
        print_img_net_msg("Bias gradient:", layer[2].conv.bias.grad, 1, 1,
                          CH_OUT, 1);
        print_img_net_msg("Kernel gradient:", layer[2].conv.kernel.grad,
                          KERNEL_SIZE, KERNEL_SIZE * CH_IN, CH_OUT, 1);
    }

    return 0;
}
