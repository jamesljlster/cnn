#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>

#include "test.h"

#define KERNEL_SIZE 5

#define BATCH 1
#define ITER 10000
#define L_RATE 0.001
#define DECAY 1.0
#define GRAD_LIMIT 30

#define MODEL_PATH "test.xml"

#define STOP_ACCURACY 97
#define FORCE 5

struct DATASET
{
    int imgWidth;
    int imgHeight;
    uint8_t imgChannel;
    uint8_t classNum;

    int batch;
    int instances;

    float* input;
    float* output;
};

int make_dataset(struct DATASET* ptr, int batch, const char* binPath);
int parse_class(float* out, int len);

int main(int argc, char* argv[])
{
    int iter;
    int i, j;
    int tmpIndex;
    int hit, force;
    float mse, accu;
    float lRate = L_RATE;

    cnn_config_t cfg = NULL;
    cnn_t cnn = NULL;

    struct DATASET data;
    int dataCols;
    int labelCols;

    float* output;
    float* err;

    // Checking argument
    if (argc < 2)
    {
        printf("Usage: <in_bin>\n");
        printf("\n");
        printf("=== Input Binary Format ===\n");
        printf("<Image Width: 4 Bytes>\n");
        printf("<Image Height: 4 Bytes>\n");
        printf("<Image Channel: 1 Byte>\n");
        printf("<Classes: 1 Byte>\n");
        printf("\n");
        printf("< <Class ID: 1 Byte> <Image Data ...> ...>\n");
        return -1;
    }

    // Make dataset
    test(make_dataset(&data, BATCH, argv[1]));
    dataCols = data.imgWidth * data.imgHeight * data.imgChannel;
    labelCols = data.classNum;

    printf("Image width: %d\n", data.imgWidth);
    printf("Image height: %d\n", data.imgHeight);
    printf("Image channel: %d\n", data.imgChannel);
    printf("Instances: %d\n", data.instances);

    // Memory allocation
    output = calloc(labelCols * BATCH, sizeof(float));
    if (output == NULL)
    {
        printf("Memory allocation failed!\n");
        return -1;
    }

    err = calloc(labelCols * BATCH, sizeof(float));
    if (err == NULL)
    {
        printf("Memory allocation failed!\n");
        return -1;
    }

    test(cnn_init());

    // Set config
    test(cnn_config_create(&cfg));
    test(cnn_config_set_input_size(cfg, data.imgWidth, data.imgHeight,
                                   data.imgChannel));
    test(cnn_config_set_batch_size(cfg, BATCH));

    test(cnn_config_append_convolution(cfg, CNN_PAD_SAME, 2, 3, KERNEL_SIZE));
    test(cnn_config_append_batchnorm(cfg, 1.0, 0.0, 0.001));
    test(cnn_config_append_pooling(cfg, 2, CNN_POOL_MAX, 2));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_convolution(cfg, CNN_PAD_SAME, 2, 3, KERNEL_SIZE));
    test(cnn_config_append_batchnorm(cfg, 1.0, 0.0, 0.001));
    test(cnn_config_append_pooling(cfg, 2, CNN_POOL_MAX, 2));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_texture(cfg, CNN_RELU, 3, 2.7183));
    test(cnn_config_append_full_connect(cfg, 128));
    test(cnn_config_append_dropout(cfg, 0.5));
    test(cnn_config_append_full_connect(cfg, 64));
    test(cnn_config_append_dropout(cfg, 0.5));
    test(cnn_config_append_full_connect(cfg, labelCols));
    test(cnn_config_append_activation(cfg, CNN_SOFTMAX));

    // Create cnn
    test(cnn_create(&cnn, cfg));
    cnn_rand_network(cnn);

    // Training
    force = 0;
    for (iter = 0; iter < ITER; iter++)
    {
        cnn_set_dropout_enabled(cnn, 1);

        for (i = 0; i < data.instances; i += BATCH)
        {
            cnn_forward(cnn, &data.input[i * dataCols], output);

            for (j = 0; j < labelCols * BATCH; j++)
            {
                err[j] = data.output[i * labelCols + j] - output[j];
            }

            cnn_backward(cnn, err);
            cnn_update(cnn, lRate, GRAD_LIMIT);
        }

        cnn_set_dropout_enabled(cnn, 0);

        mse = 0;
        hit = 0;

        for (i = 0; i < data.instances; i += BATCH)
        {
            cnn_forward(cnn, &data.input[i * dataCols], output);

            for (j = 0; j < labelCols * BATCH; j++)
            {
                float tmp = data.output[i * labelCols + j] - output[j];
                mse += tmp * tmp;
            }

            for (j = 0; j < BATCH; j++)
            {
                tmpIndex = parse_class(&output[j * labelCols], labelCols);
                if (data.output[(i + j) * labelCols + tmpIndex] > 0)
                {
                    hit++;
                }
            }
        }

        mse /= (float)(labelCols * data.instances);
        accu = (float)hit * 100.0 / (float)data.instances;
        printf("Iter %d, mse: %f, accuracy: %.2f %%\n", iter, mse, accu);

        if (accu > STOP_ACCURACY)
        {
            force++;
            if (force > FORCE)
            {
                break;
            }
        }
        else
        {
            force = 0;
        }

        // lRate = L_RATE * sqrt(mse);
        lRate *= DECAY;
    }

    // Export
    test(cnn_export(cnn, MODEL_PATH));

    return 0;
}

int parse_class(float* out, int len)
{
    int i;
    int index = 0;
    float hold = out[0];

    for (i = 1; i < len; i++)
    {
        if (hold < out[i])
        {
            hold = out[i];
            index = i;
        }
    }

    return index;
}

void byte_reverse(void* ptr, size_t size)
{
    unsigned char* tmp = malloc(size);
    assert(tmp != NULL);

    for (size_t i = 0; i < size; i++)
    {
        tmp[i] = ((unsigned char*)ptr)[size - i - 1];
    }

    memcpy(ptr, tmp, size);
    free(tmp);
}

int make_dataset(struct DATASET* ptr, int batch, const char* binPath)
{
    int i;
    int ret = 0;
    int rows, dataCols, labelCols;
    int tmpRows;
    size_t fSize;
    size_t fReadSize;
    char tmp;
    FILE* fRead = NULL;
    void* allocTmp;
    char* imgBuf = NULL;

    // Zero memory
    memset(ptr, 0, sizeof(struct DATASET));

    // Open file
    fRead = fopen(binPath, "rb");
    if (fRead == NULL)
    {
        printf("Failed to open %s\n", binPath);
        ret = -1;
        goto RET;
    }

    fseek(fRead, 0, SEEK_END);
    fSize = ftell(fRead);
    fReadSize = 0;
    fseek(fRead, 0, SEEK_SET);

#define __fread(ptr, size)                   \
    ret = fread((char*)ptr, 1, size, fRead); \
    if (ret != size)                         \
    {                                        \
        printf("Broken file\n");             \
        ret = -1;                            \
        goto ERR;                            \
    }                                        \
    else                                     \
    {                                        \
        fReadSize += size;                   \
    }

    // Parse header
    __fread(&ptr->imgWidth, 4);
    __fread(&ptr->imgHeight, 4);
    __fread(&ptr->imgChannel, 1);
    __fread(&ptr->classNum, 1);

    byte_reverse(&ptr->imgWidth, 4);
    byte_reverse(&ptr->imgHeight, 4);

    // Set columns
    dataCols = ptr->imgWidth * ptr->imgHeight * ptr->imgChannel;
    labelCols = ptr->classNum;

    // Memory allocation
    imgBuf = (char*)calloc(dataCols, 1);
    if (imgBuf == NULL)
    {
        printf("Memory allocation failed!\n");
        goto ERR;
    }

#define __realloc(ptr, len)                       \
    allocTmp = realloc(ptr, len * sizeof(float)); \
    if (allocTmp == NULL)                         \
    {                                             \
        printf("Memory allocation failed!\n");    \
        goto ERR;                                 \
    }                                             \
    else                                          \
    {                                             \
        ptr = (float*)allocTmp;                   \
    }

    // Read images
    rows = 0;
    while (fReadSize < fSize)
    {
        // Reserve memory
        rows++;
        __realloc(ptr->input, rows * dataCols);
        __realloc(ptr->output, rows * labelCols);

        // Read label
        __fread(&tmp, 1);
        for (i = 0; i < labelCols; i++)
        {
            if (i == tmp)
            {
                ptr->output[(rows - 1) * labelCols + i] = 1;
            }
            else
            {
                ptr->output[(rows - 1) * labelCols + i] = 0;
            }
        }

        // Read image
        __fread(imgBuf, dataCols);
        for (i = 0; i < dataCols; i++)
        {
            ptr->input[(rows - 1) * dataCols + i] = (float)imgBuf[i] / 255.0;
        }
    }

    // Check batch
    if (rows % batch != 0)
    {
        tmpRows = (rows / batch + 1) * batch;
        __realloc(ptr->input, tmpRows * dataCols);
        __realloc(ptr->output, tmpRows * labelCols);

        for (i = rows; i < tmpRows; i++)
        {
            memcpy(&ptr->input[i * dataCols],
                   &ptr->input[(i % rows) * dataCols],
                   sizeof(float) * dataCols);
            memcpy(&ptr->output[i * labelCols],
                   &ptr->output[(i % rows) * labelCols],
                   sizeof(float) * labelCols);
        }

        rows = tmpRows;
    }

    // Assign value
    ptr->batch = batch;
    ptr->instances = rows;

    goto RET;

ERR:
    free(ptr->input);
    free(ptr->output);
    free(imgBuf);

RET:
    if (fRead != NULL)
    {
        fclose(fRead);
    }

    return ret;
}
