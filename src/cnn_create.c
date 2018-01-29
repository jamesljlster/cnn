#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cnn.h"
#include "cnn_private.h"

int cnn_create(cnn_t* cnnPtr, const cnn_config_t cfg)
{
	int ret = CNN_NO_ERROR;
	struct CNN* tmpCnn;

	// Memory allocation
	cnn_alloc(tmpCnn, 1, struct CNN, ret, RET);

	// Clone config
	cnn_run(cnn_config_struct_clone(&tmpCnn->cfg, cfg), ret, ERR);

	// Assing value
	*cnnPtr = tmpCnn;

	goto RET;

ERR:
	cnn_delete(tmpCnn);

RET:
	return ret;
}
