#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cnn.h"
#include "cnn_private.h"

int cnn_create(cnn_t* cnnPtr)
{
	int ret = CNN_NO_ERROR;
	struct CNN* tmpCnn;

	// Memory allocation
	cnn_alloc(tmpCnn, 1, struct CNN, ret, RET);

	// Assing value
	*cnnPtr = tmpCnn;

	goto RET;

ERR:
	cnn_delete(tmpCnn);

RET:
	return ret;
}
