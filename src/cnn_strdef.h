#ifndef __CNN_STRDEF_H__
#define __CNN_STRDEF_H__

// CNN reserve words enumeration
enum CNN_STR_LIST
{
    CNN_STR_CONFIG,     //
    CNN_STR_MODEL,      //
    CNN_STR_NETWORK,    //
    CNN_STR_LAYER,      //
    CNN_STR_INDEX,      //
    CNN_STR_TYPE,       //
    CNN_STR_DIM,        //
    CNN_STR_SIZE,       //
    CNN_STR_CHANNEL,    //
    CNN_STR_BATCH,      //
    CNN_STR_VALUE,      //
    CNN_STR_INPUT,      //
    CNN_STR_WIDTH,      //
    CNN_STR_HEIGHT,     //
    CNN_STR_FC,         //
    CNN_STR_ACTIV,      //
    CNN_STR_CONV,       //
    CNN_STR_POOL,       //
    CNN_STR_DROP,       //
    CNN_STR_BN,         //
    CNN_STR_TEXT,       //
    CNN_STR_ID,         //
    CNN_STR_POOL_TYPE,  //
    CNN_STR_MAX,        //
    CNN_STR_AVG,        //
    CNN_STR_KERNEL,     //
    CNN_STR_BIAS,       //
    CNN_STR_WEIGHT,     //
    CNN_STR_1D,         //
    CNN_STR_2D,         //
    CNN_STR_RATE,       //
    CNN_STR_FILTER,     //
    CNN_STR_PAD,        //
    CNN_STR_VALID,      //
    CNN_STR_SAME,       //
    CNN_STR_GAMMA,      //
    CNN_STR_BETA,       //
    CNN_STR_PARAM,      //
    CNN_STR_ALPHA,      //
    CNN_STR_MEAN,       //
    CNN_STR_VAR,        //
    CNN_STR_EAF,        // Exponential average factor

    CNN_STR_AMOUNT
};

// CNN reserve words list
extern const char* cnn_str_list[];

#ifdef __cplusplus
extern "C"
{
#endif

    int cnn_strdef_get_id(const char* str);

#ifdef __cplusplus
}
#endif

#endif
