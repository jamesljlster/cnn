#include <string.h>

#include "cnn.h"
#include "cnn_strdef.h"

const char* cnn_str_list[] = {
    "config",     //
    "cnn_model",  //
    "network",    //
    "layer",      //
    "index",      //
    "type",       //
    "dim",        //
    "size",       //
    "channel",    //
    "batch",      //
    "value",      //
    "input",      //
    "width",      //
    "height",     //
    "fc",         //
    "activ",      //
    "conv",       //
    "pool",       //
    "drop",       //
    "bn",         //
    "text",       //
    "id",         //
    "pool_type",  //
    "max",        //
    "avg",        //
    "kernel",     //
    "bias",       //
    "weight",     //
    "1d",         //
    "2d",         //
    "rate",       //
    "filter",     //
    "padding",    //
    "valid",      //
    "same",       //
    "gamma",      //
    "beta",       //
    "param",      //
    "alpha",      //
    "mean",       //
    "var",        //
    "expAvgF",    //
};

int cnn_strdef_get_id(const char* str)
{
    int i;
    int strId = CNN_PARSE_FAILED;
    int ret;

    if (str != NULL)
    {
        for (i = 0; i < CNN_STR_AMOUNT; i++)
        {
            ret = strcmp(str, cnn_str_list[i]);
            if (ret == 0)
            {
                strId = i;
                goto RET;
            }
        }
    }

RET:
    return strId;
}
