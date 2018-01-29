#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cnn.h"
#include "cnn_private.h"
#include "cnn_builtin_math.h"

int cnn_mat_alloc(struct CNN_MAT* matPtr, int rows, int cols, int needGrad)
{
	int ret = CNN_NO_ERROR;

	// Checking
	if(rows <= 0 || cols <= 0)
	{
		ret = CNN_INVALID_SHAPE;
		goto RET;
	}

	// Memory allocation
	cnn_alloc(matPtr->mat, rows * cols, float, ret, ERR);
	if(needGrad > 0)
	{
		cnn_alloc(matPtr->grad, rows * cols, float, ret, ERR);
	}
	else
	{
		matPtr->grad = NULL;
	}

	// Assign value
	matPtr->rows = rows;
	matPtr->cols = cols;

	goto RET;

ERR:
	cnn_mat_delete(matPtr);

RET:
	return ret;
}

int cnn_layer_afunc_alloc(struct CNN_LAYER_AFUNC* layerPtr, int outSize, int batch)
{
	int ret = CNN_NO_ERROR;
	int outRows, outCols;
	int gradRows, gradCols;

	// Find allocate size
	outRows = batch;
	outCols = outSize;

	gradRows = outSize * batch;
	gradCols = outSize;

	// Allocate memory
	cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret, ERR);
	cnn_run(cnn_mat_alloc(&layerPtr->gradMat, gradRows, gradCols, 0), ret, ERR);

	goto RET;

ERR:
	cnn_layer_afunc_delete(layerPtr);

RET:
	return ret;
}

int cnn_layer_fc_alloc(struct CNN_LAYER_FC* layerPtr, int inSize, int outSize, int batch)
{
	int ret = CNN_NO_ERROR;
	int outRows, outCols; // Output matrix size
	int wRows, wCols; // Weight matrix size
	int bRows, bCols; // Bias matrix size

	// Find allocate size
	outRows = batch;
	outCols = outSize;

	wRows = inSize;
	wCols = outSize;

	bRows = 1;
	bCols = outSize;

	// Allocate memory
	cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret, ERR);
	cnn_run(cnn_mat_alloc(&layerPtr->weight, wRows, wCols, 1), ret, ERR);
	cnn_run(cnn_mat_alloc(&layerPtr->bias, bRows, bCols, 1), ret, ERR);

	goto RET;

ERR:
	cnn_layer_fc_delete(layerPtr);

RET:
	return ret;
}

