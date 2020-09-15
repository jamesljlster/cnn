#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "cnn.h"
#include "cnn_init.h"
#include "cnn_private.h"

int cnn_create(cnn_t* cnnPtr, const cnn_config_t cfg)
{
    int ret = CNN_NO_ERROR;
    struct CNN* tmpCnn;

    // Memory allocation
    cnn_alloc(tmpCnn, 1, struct CNN, ret, RET);

    // Clone config
    cnn_run(cnn_config_struct_clone(&tmpCnn->cfg, cfg), ret, ERR);

    // Allocate network
    cnn_run(cnn_network_alloc(tmpCnn), ret, ERR);

#ifdef CNN_WITH_CUDA
    // Allocate workspace size
    cnn_run(cnn_cudnn_ws_alloc(), ret, ERR);
#endif

    // Assing value
    *cnnPtr = tmpCnn;

    goto RET;

ERR:
    cnn_delete(tmpCnn);

RET:
    return ret;
}
