#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "cnn.h"
#include "cnn_builtin_math.h"
#include "cnn_private.h"

#define CNN_DEFAULT_INPUT_WIDTH 32
#define CNN_DEFAULT_INPUT_HEIGHT 32
#define CNN_DEFAULT_INPUT_CHANNEL 3
#define CNN_DEFAULT_BATCH 1
#define CNN_DEFAULT_LAYERS 1
#define CNN_DEFAULT_FC_SIZE 16
#define CNN_DEFAULT_LRATE 0.001

int cnn_config_compare(const cnn_config_t src1, const cnn_config_t src2)
{
    int i;
    int tmpRet;
    int ret = CNN_NO_ERROR;

#ifdef DEBUG
#define __cmp(elem)                                                         \
    if (src1->elem != src2->elem)                                           \
    {                                                                       \
        fprintf(stderr, "%s(), %d: %s not match\n", __FUNCTION__, __LINE__, \
                #elem);                                                     \
        ret = CNN_CONFLICT;                                                 \
        goto RET;                                                           \
    }
#else
#define __cmp(elem)               \
    if (src1->elem != src2->elem) \
    {                             \
        ret = CNN_CONFLICT;       \
        goto RET;                 \
    }
#endif

    // Compare
    __cmp(width);
    __cmp(height);
    __cmp(channel);
    __cmp(batch);
    __cmp(layers);

    for (i = 0; i < src1->layers; i++)
    {
        // Compare type
        __cmp(layerCfg[i].type);

#ifdef DEBUG
#define __cmp_mem(elem, type)                                                  \
    tmpRet = memcmp(&src1->layerCfg[i].elem, &src2->layerCfg[i].elem,          \
                    sizeof(type));                                             \
    if (tmpRet != 0)                                                           \
    {                                                                          \
        fprintf(stderr,                                                        \
                "%s(), %d: 'cfg->layerCfg[%d].%s' not match with type '%s'\n", \
                __FUNCTION__, __LINE__, i, #elem, #type);                      \
        ret = CNN_CONFLICT;                                                    \
        goto RET;                                                              \
    }
#else
#define __cmp_mem(elem, type)                                         \
    tmpRet = memcmp(&src1->layerCfg[i].elem, &src2->layerCfg[i].elem, \
                    sizeof(type));                                    \
    if (tmpRet != 0)                                                  \
    {                                                                 \
        ret = CNN_CONFLICT;                                           \
        goto RET;                                                     \
    }
#endif

        // Compare detail
        switch (src1->layerCfg[i].type)
        {
            case CNN_LAYER_INPUT:
                break;

            case CNN_LAYER_FC:
                __cmp_mem(fc, struct CNN_CONFIG_LAYER_FC);
                break;

            case CNN_LAYER_ACTIV:
                __cmp_mem(activ, struct CNN_CONFIG_LAYER_ACTIV);
                break;

            case CNN_LAYER_CONV:
                __cmp_mem(conv, struct CNN_CONFIG_LAYER_CONV);
                break;

            case CNN_LAYER_POOL:
                __cmp_mem(pool, struct CNN_CONFIG_LAYER_POOL);
                break;

            case CNN_LAYER_DROP:
                __cmp_mem(drop, struct CNN_CONFIG_LAYER_DROP);
                break;

            case CNN_LAYER_BN:
                __cmp_mem(bn, struct CNN_CONFIG_LAYER_BN);
                break;

            default:
                assert(!"Invalid layer type");
        }
    }

RET:
    return ret;
}

void cnn_set_opmode(cnn_t cnn, cnn_opmode_t opMode) { cnn->opMode = opMode; }

cnn_config_t cnn_get_config(cnn_t cnn) { return &cnn->cfg; }

int cnn_config_clone(cnn_config_t* cfgPtr, const cnn_config_t src)
{
    int ret = CNN_NO_ERROR;
    struct CNN_CONFIG* tmpCfg;

    // Memory allocation
    cnn_alloc(tmpCfg, 1, struct CNN_CONFIG, ret, RET);

    // Clone config struct
    cnn_run(cnn_config_struct_clone(tmpCfg, src), ret, ERR);

    // Assing value
    *cfgPtr = tmpCfg;

    goto RET;

ERR:
    cnn_config_delete(tmpCfg);

RET:
    return ret;
}

int cnn_config_struct_clone(struct CNN_CONFIG* dst,
                            const struct CNN_CONFIG* src)
{
    int ret = CNN_NO_ERROR;

    // Copy setting
    memcpy(dst, src, sizeof(struct CNN_CONFIG));

    // Memory allocation
    cnn_alloc(dst->layerCfg, dst->layers, union CNN_CONFIG_LAYER, ret, ERR);

    // Copy layer setting
    memcpy(dst->layerCfg, src->layerCfg,
           dst->layers * sizeof(union CNN_CONFIG_LAYER));

    goto RET;

ERR:
    cnn_config_struct_delete(dst);

RET:
    return ret;
}

int cnn_config_create(cnn_config_t* cfgPtr)
{
    int ret = CNN_NO_ERROR;
    struct CNN_CONFIG* tmpCfg;

    // Memory allocation
    cnn_alloc(tmpCfg, 1, struct CNN_CONFIG, ret, RET);

    // Initial config
    cnn_run(cnn_config_init(tmpCfg), ret, ERR);

    // Assign value
    *cfgPtr = tmpCfg;

    goto RET;

ERR:
    cnn_config_delete(tmpCfg);

RET:
    return ret;
}

int cnn_config_init(cnn_config_t cfg)
{
    int ret = CNN_NO_ERROR;

    // Set default cnn settings
    // Set input size
    cnn_run(cnn_config_set_input_size(cfg, CNN_DEFAULT_INPUT_WIDTH,
                                      CNN_DEFAULT_INPUT_HEIGHT,
                                      CNN_DEFAULT_INPUT_CHANNEL),
            ret, RET);

    // Set batch
    cnn_run(cnn_config_set_batch_size(cfg, CNN_DEFAULT_BATCH), ret, RET);

    // Set total layers
    cnn_run(cnn_config_set_layers(cfg, CNN_DEFAULT_LAYERS), ret, RET);

RET:
    return ret;
}

void cnn_config_struct_delete(struct CNN_CONFIG* cfg)
{
    cnn_free(cfg->layerCfg);
    memset(cfg, 0, sizeof(struct CNN_CONFIG));
}

void cnn_config_delete(cnn_config_t cfg)
{
    if (cfg != NULL)
    {
        cnn_config_struct_delete(cfg);
        cnn_free(cfg);
    }
}

int cnn_config_set_input_size(cnn_config_t cfg, int width, int height,
                              int channel)
{
    int ret = CNN_NO_ERROR;

    // Checking
    if (width <= 0 || height <= 0)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Assign value
    cfg->width = width;
    cfg->height = height;
    cfg->channel = channel;

RET:
    return ret;
}

void cnn_config_get_input_size(cnn_config_t cfg, int* wPtr, int* hPtr,
                               int* cPtr)
{
    if (wPtr != NULL)
    {
        *wPtr = cfg->width;
    }

    if (hPtr != NULL)
    {
        *hPtr = cfg->height;
    }

    if (cPtr != NULL)
    {
        *cPtr = cfg->channel;
    }
}

int cnn_config_find_layer_outsize(int* outWPtr, int* outHPtr, int* outCPtr,
                                  int inWidth, int inHeight, int inChannel,
                                  union CNN_CONFIG_LAYER* layerCfg)
{
    int outWidth, outHeight, outChannel;

    // Find output size
    switch (layerCfg->type)
    {
        case CNN_LAYER_INPUT:
        case CNN_LAYER_ACTIV:
        case CNN_LAYER_DROP:
        case CNN_LAYER_BN:
            outWidth = inWidth;
            outHeight = inHeight;
            outChannel = inChannel;
            break;

        case CNN_LAYER_FC:
            outWidth = layerCfg->fc.size;
            outHeight = 1;
            outChannel = 1;
            break;

        case CNN_LAYER_CONV:
            switch (layerCfg->conv.pad)
            {
                case CNN_PAD_VALID:
                    outWidth = inWidth - layerCfg->conv.size + 1;
                    outHeight = inHeight - layerCfg->conv.size + 1;
                    break;

                case CNN_PAD_SAME:
                    outWidth = inWidth;
                    outHeight = inHeight;
                    break;

                default:
                    assert(!"Invalid padding type");
            }

            outChannel = layerCfg->conv.filter;
            break;

        case CNN_LAYER_POOL:
            outWidth = inWidth / layerCfg->pool.size;
            outHeight = inHeight / layerCfg->pool.size;
            outChannel = inChannel;
            break;

        default:
            assert(!"Invalid layer type");
    }

    // Check output shape
    if (outWidth <= 0 || outHeight <= 0 || outChannel <= 0)
    {
        return CNN_INVALID_SHAPE;
    }

    // Assign value
    *outWPtr = outWidth;
    *outHPtr = outHeight;
    *outCPtr = outChannel;

    return CNN_NO_ERROR;
}

void cnn_config_get_output_size(cnn_config_t cfg, int* wPtr, int* hPtr,
                                int* cPtr)
{
    int i;
    int inWidth, inHeight, inChannel;
    int outWidth, outHeight, outChannel;

    // Set initial size
    inWidth = cfg->width;
    inHeight = cfg->height;
    inChannel = cfg->channel;

    outWidth = inWidth;
    outHeight = inHeight;
    outChannel = inChannel;

    // Find output size
    for (i = 1; i < cfg->layers; i++)
    {
        cnn_config_find_layer_outsize(&outWidth, &outHeight, &outChannel,
                                      inWidth, inHeight, inChannel,
                                      &cfg->layerCfg[i]);

        inWidth = outWidth;
        inHeight = outHeight;
        inChannel = outChannel;
    }

    // Assign value
    if (wPtr != NULL)
    {
        *wPtr = outWidth;
    }

    if (hPtr != NULL)
    {
        *hPtr = outHeight;
    }

    if (cPtr != NULL)
    {
        *cPtr = outChannel;
    }
}

int cnn_config_set_batch_size(cnn_config_t cfg, int batchSize)
{
    int ret = CNN_NO_ERROR;

    // Checking
    if (batchSize <= 0)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Assign value
    cfg->batch = batchSize;

RET:
    return ret;
}

void cnn_config_get_batch_size(cnn_config_t cfg, int* batchPtr)
{
    if (batchPtr != NULL)
    {
        *batchPtr = cfg->batch;
    }
}

int cnn_config_set_layers(cnn_config_t cfg, int layers)
{
    int ret = CNN_NO_ERROR;
    int i, preLayers;

    union CNN_CONFIG_LAYER* tmpLayerCfg;

    // Checking
    if (layers <= 0)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    preLayers = cfg->layers;
    if (preLayers == layers)
    {
        ret = CNN_NO_ERROR;
        goto RET;
    }

    // Reallocate layer config list
    tmpLayerCfg =
        realloc(cfg->layerCfg, layers * sizeof(union CNN_CONFIG_LAYER));
    if (tmpLayerCfg == NULL)
    {
        ret = CNN_MEM_FAILED;
        goto RET;
    }
    else
    {
        cfg->layerCfg = tmpLayerCfg;
        cfg->layers = layers;
    }

    // Set default layer config
    if (preLayers <= 0)
    {
        i = 1;
        cfg->layerCfg[0].type = CNN_LAYER_INPUT;
    }
    else
    {
        i = preLayers;
    }

    for (; i < layers; i++)
    {
        ret = cnn_config_set_full_connect(cfg, i, CNN_DEFAULT_FC_SIZE);
        assert(ret == CNN_NO_ERROR);
    }

RET:
    return ret;
}

void cnn_config_get_layers(cnn_config_t cfg, int* layersPtr)
{
    if (layersPtr != NULL)
    {
        *layersPtr = cfg->layers;
    }
}

int cnn_config_get_layer_type(cnn_config_t cfg, int layerIndex,
                              cnn_layer_t* typePtr)
{
    int ret = CNN_NO_ERROR;

    // Checking
    if (layerIndex < 0 || layerIndex >= cfg->layers)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    if (typePtr != NULL)
    {
        *typePtr = cfg->layerCfg[layerIndex].type;
    }

RET:
    return ret;
}

int cnn_config_append_full_connect(cnn_config_t cfg, int size)
{
    int ret = CNN_NO_ERROR;
    int layers;

    // Checking
    if (size <= 0)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Append fully connected layer
    cnn_config_get_layers(cfg, &layers);
    layers++;
    cnn_run(cnn_config_set_layers(cfg, layers), ret, RET);
    cnn_run(cnn_config_set_full_connect(cfg, layers - 1, size), ret, RET);

RET:
    return ret;
}

int cnn_config_set_full_connect(cnn_config_t cfg, int layerIndex, int size)
{
    int ret = CNN_NO_ERROR;

    // Checking
    if (size <= 0 || layerIndex <= 0 || layerIndex >= cfg->layers)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Set config
    cfg->layerCfg[layerIndex].type = CNN_LAYER_FC;
    cfg->layerCfg[layerIndex].fc.size = size;

RET:
    return ret;
}

int cnn_config_get_full_connect(cnn_config_t cfg, int layerIndex, int* sizePtr)
{
    int ret = CNN_NO_ERROR;

    // Checking
    if (layerIndex < 0 || layerIndex >= cfg->layers)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    if (cfg->layerCfg[layerIndex].type != CNN_LAYER_FC)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Assign value
    if (sizePtr != NULL)
    {
        *sizePtr = cfg->layerCfg[layerIndex].fc.size;
    }

RET:
    return ret;
}

int cnn_config_append_activation(cnn_config_t cfg, cnn_activ_t activID)
{
    int ret = CNN_NO_ERROR;
    int layers;

    // Checking
    if (activID < 0 || activID >= CNN_ACTIV_AMOUNT)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Append activation layer
    cnn_config_get_layers(cfg, &layers);
    layers++;
    cnn_run(cnn_config_set_layers(cfg, layers), ret, RET);
    cnn_run(cnn_config_set_activation(cfg, layers - 1, activID), ret, RET);

RET:
    return ret;
}

int cnn_config_set_activation(cnn_config_t cfg, int layerIndex,
                              cnn_activ_t activID)
{
    int ret = CNN_NO_ERROR;

    // Checking
    if (layerIndex <= 0 || layerIndex >= cfg->layers || activID < 0 ||
        activID >= CNN_ACTIV_AMOUNT)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Set config
    cfg->layerCfg[layerIndex].type = CNN_LAYER_ACTIV;
    cfg->layerCfg[layerIndex].activ.id = activID;

RET:
    return ret;
}

int cnn_config_get_activation(cnn_config_t cfg, int layerIndex,
                              cnn_activ_t* idPtr)
{
    int ret = CNN_NO_ERROR;

    // Checking
    if (layerIndex < 0 || layerIndex >= cfg->layers)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    if (cfg->layerCfg[layerIndex].type != CNN_LAYER_ACTIV)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Assing value
    if (idPtr != NULL)
    {
        *idPtr = cfg->layerCfg[layerIndex].activ.id;
    }

RET:
    return ret;
}

int cnn_config_append_convolution(cnn_config_t cfg, cnn_pad_t padding,
                                  cnn_dim_t convDim, int filter, int size)
{
    int ret = CNN_NO_ERROR;
    int layers;

    // Checking
    if (convDim <= 0 || convDim > 2 || size <= 0 || filter <= 0)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Append convolution layer
    cnn_config_get_layers(cfg, &layers);
    layers++;
    cnn_run(cnn_config_set_layers(cfg, layers), ret, RET);
    cnn_run(cnn_config_set_convolution(cfg, layers - 1, padding, convDim,
                                       filter, size),
            ret, RET);

RET:
    return ret;
}

int cnn_config_set_convolution(cnn_config_t cfg, int layerIndex,
                               cnn_pad_t padding, cnn_dim_t convDim, int filter,
                               int size)
{
    int ret = CNN_NO_ERROR;

    // Checking
    if (layerIndex <= 0 || layerIndex >= cfg->layers ||             //
        convDim <= 0 || convDim > 2 || size <= 0 || filter <= 0 ||  //
        padding < CNN_PAD_VALID || padding > CNN_PAD_SAME)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Set config
    cfg->layerCfg[layerIndex].type = CNN_LAYER_CONV;
    cfg->layerCfg[layerIndex].conv.pad = padding;
    cfg->layerCfg[layerIndex].conv.dim = convDim;
    cfg->layerCfg[layerIndex].conv.size = size;
    cfg->layerCfg[layerIndex].conv.filter = filter;

RET:
    return ret;
}

int cnn_config_get_convolution(cnn_config_t cfg, int layerIndex,
                               cnn_pad_t* padPtr, cnn_dim_t* dimPtr,
                               int* filterPtr, int* sizePtr)
{
    int ret = CNN_NO_ERROR;

    // Checking
    if (layerIndex < 0 || layerIndex >= cfg->layers)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    if (cfg->layerCfg[layerIndex].type != CNN_LAYER_CONV)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Assign value
    if (padPtr != NULL)
    {
        *padPtr = cfg->layerCfg[layerIndex].conv.pad;
    }

    if (dimPtr != NULL)
    {
        *dimPtr = cfg->layerCfg[layerIndex].conv.dim;
    }

    if (filterPtr != NULL)
    {
        *filterPtr = cfg->layerCfg[layerIndex].conv.filter;
    }

    if (sizePtr != NULL)
    {
        *sizePtr = cfg->layerCfg[layerIndex].conv.size;
    }

RET:
    return ret;
}

int cnn_config_append_pooling(cnn_config_t cfg, cnn_dim_t dim, cnn_pool_t type,
                              int size)
{
    int ret = CNN_NO_ERROR;
    int layers;

    // Checking
    if (dim <= 0 || dim > 2 || type < CNN_POOL_MAX || type > CNN_POOL_AVG ||
        size <= 0)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Append pooling layer
    cnn_config_get_layers(cfg, &layers);
    layers++;
    cnn_run(cnn_config_set_layers(cfg, layers), ret, RET);
    cnn_run(cnn_config_set_pooling(cfg, layers - 1, dim, type, size), ret, RET);

RET:
    return ret;
}

int cnn_config_set_pooling(cnn_config_t cfg, int layerIndex, cnn_dim_t dim,
                           cnn_pool_t type, int size)
{
    int ret = CNN_NO_ERROR;

    // Checking
    if (layerIndex <= 0 || layerIndex >= cfg->layers || type < CNN_POOL_MAX ||
        type > CNN_POOL_AVG || dim <= 0 || dim > 2 || size <= 0)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Set config
    cfg->layerCfg[layerIndex].type = CNN_LAYER_POOL;
    cfg->layerCfg[layerIndex].pool.poolType = type;
    cfg->layerCfg[layerIndex].pool.dim = dim;
    cfg->layerCfg[layerIndex].pool.size = size;

RET:
    return ret;
}

int cnn_config_get_pooling(cnn_config_t cfg, int layerIndex, cnn_dim_t* dimPtr,
                           cnn_pool_t* typePtr, int* sizePtr)
{
    int ret = CNN_NO_ERROR;

    // Checking
    if (layerIndex < 0 || layerIndex >= cfg->layers)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    if (cfg->layerCfg[layerIndex].type != CNN_LAYER_POOL)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Assign value
    if (dimPtr != NULL)
    {
        *dimPtr = cfg->layerCfg[layerIndex].pool.dim;
    }

    if (typePtr != NULL)
    {
        *typePtr = cfg->layerCfg[layerIndex].pool.poolType;
    }

    if (sizePtr != NULL)
    {
        *sizePtr = cfg->layerCfg[layerIndex].pool.size;
    }

RET:
    return ret;
}

int cnn_config_append_dropout(cnn_config_t cfg, float rate)
{
    int ret = CNN_NO_ERROR;
    int layers;

    // Checking
    if (rate < 0.0 || rate > 1.0)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Append dropout layer
    cnn_config_get_layers(cfg, &layers);
    layers++;
    cnn_run(cnn_config_set_layers(cfg, layers), ret, RET);
    cnn_run(cnn_config_set_dropout(cfg, layers - 1, rate), ret, RET);

RET:
    return ret;
}

int cnn_config_set_dropout(cnn_config_t cfg, int layerIndex, float rate)
{
    int ret = CNN_NO_ERROR;

    // Checking
    if (layerIndex <= 0 || layerIndex >= cfg->layers || rate < 0.0 ||
        rate > 1.0)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Set config
    cfg->layerCfg[layerIndex].type = CNN_LAYER_DROP;
    cfg->layerCfg[layerIndex].drop.rate = rate;
    cfg->layerCfg[layerIndex].drop.scale = 1.0 / (1.0 - rate);

RET:
    return ret;
}

int cnn_config_get_dropout(cnn_config_t cfg, int layerIndex, float* ratePtr)
{
    int ret = CNN_NO_ERROR;

    // Checking
    if (layerIndex < 0 || layerIndex >= cfg->layers)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    if (cfg->layerCfg[layerIndex].type != CNN_LAYER_DROP)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Assign value
    if (ratePtr != NULL)
    {
        *ratePtr = cfg->layerCfg[layerIndex].drop.rate;
    }

RET:
    return ret;
}

int cnn_config_append_batchnorm(cnn_config_t cfg, float rInit, float bInit,
                                float expAvgFactor)
{
    int ret = CNN_NO_ERROR;
    int layers;

    // Append batch normalization layer
    cnn_config_get_layers(cfg, &layers);
    layers++;
    cnn_run(cnn_config_set_layers(cfg, layers), ret, RET);
    cnn_run(
        cnn_config_set_batchnorm(cfg, layers - 1, rInit, bInit, expAvgFactor),
        ret, RET);

RET:
    return ret;
}

int cnn_config_set_batchnorm(cnn_config_t cfg, int layerIndex, float rInit,
                             float bInit, float expAvgFactor)
{
    int ret = CNN_NO_ERROR;

    // Checking
    if (layerIndex <= 0 || layerIndex >= cfg->layers)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Set config
    cfg->layerCfg[layerIndex].type = CNN_LAYER_BN;
    cfg->layerCfg[layerIndex].bn.rInit = rInit;
    cfg->layerCfg[layerIndex].bn.bInit = bInit;
    cfg->layerCfg[layerIndex].bn.expAvgFactor = expAvgFactor;

RET:
    return ret;
}

int cnn_config_get_batchnorm(cnn_config_t cfg, int layerIndex, float* rInitPtr,
                             float* bInitPtr, float* expAvgFactorPtr)
{
    int ret = CNN_NO_ERROR;

    // Checking
    if (layerIndex < 0 || layerIndex >= cfg->layers)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    if (cfg->layerCfg[layerIndex].type != CNN_LAYER_BN)
    {
        ret = CNN_INVALID_ARG;
        goto RET;
    }

    // Assign value
    if (rInitPtr != NULL)
    {
        *rInitPtr = cfg->layerCfg[layerIndex].bn.rInit;
    }

    if (bInitPtr != NULL)
    {
        *bInitPtr = cfg->layerCfg[layerIndex].bn.bInit;
    }

    if (expAvgFactorPtr != NULL)
    {
        *expAvgFactorPtr = cfg->layerCfg[layerIndex].bn.expAvgFactor;
    }

RET:
    return ret;
}
