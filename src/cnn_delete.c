#include <stdlib.h>
#include <string.h>

#include "cnn.h"
#include "cnn_private.h"

void cnn_mat_delete(struct CNN_MAT* matPtr)
{
	// Free memory
	cnn_free(matPtr->mat);
	cnn_free(matPtr->grad);

	// Zero memory
	memset(matPtr, 0, sizeof(struct CNN_MAT));
}

void cnn_layer_afunc_delete(struct CNN_LAYER_AFUNC* layerPtr)
{
	// Free memory
	cnn_mat_delete(&layerPtr->outMat.data);
	cnn_mat_delete(&layerPtr->gradMat);

	// Zero memory
	memset(layerPtr, 0, sizeof(struct CNN_LAYER_AFUNC));
}

void cnn_layer_fc_delete(struct CNN_LAYER_FC* layerPtr)
{
	// Free memory
	cnn_mat_delete(&layerPtr->outMat.data);
	cnn_mat_delete(&layerPtr->weight);
	cnn_mat_delete(&layerPtr->bias);

	// Zero memory
	memset(layerPtr, 0, sizeof(struct CNN_LAYER_FC));
}

