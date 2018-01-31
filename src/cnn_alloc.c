#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cnn.h"
#include "cnn_private.h"
#include "cnn_builtin_math.h"

int cnn_network_alloc(struct CNN* cnn, const struct CNN_CONFIG* cfg)
{
	int i;
	int ret = CNN_NO_ERROR;
	int tmpWidth, tmpHeight;

	// Memory allocation
	cnn_alloc(cnn->layerList, cfg->layers, union CNN_LAYER, ret, ERR);

	// Allocate CNN layers
	tmpWidth = cfg->width;
	tmpHeight = cfg->height;
	for(i = 0; i < cfg->layers; i++)
	{
		switch(cfg->layerCfg[i].type)
		{
			case CNN_LAYER_INPUT:
				cnn_run(cnn_layer_input_alloc(&cnn->layerList[i],
							tmpWidth, tmpHeight, cfg->batch),
						ret, ERR);
				break;

			case CNN_LAYER_FC:
				cnn_run(cnn_layer_fc_alloc(&cnn->layerList[i].fc,
							tmpWidth, tmpHeight, cfg->layerCfg[i].fc.size, cfg->batch),
						ret, ERR);
				break;

			case CNN_LAYER_AFUNC:
				cnn_run(cnn_layer_afunc_alloc(&cnn->layerList[i].aFunc,
							tmpWidth, tmpHeight, cfg->batch),
						ret, ERR);
				break;

			case CNN_LAYER_CONV:
				cnn_run(cnn_layer_conv_alloc(&cnn->layerList[i].conv,
							tmpWidth, tmpHeight, cfg->layerCfg[i].conv.size, cfg->batch),
						ret, ERR);
				break;
		}

		// Find layer output image size
		tmpWidth = cnn->layerList[i].outMat.width;
		tmpHeight = cnn->layerList[i].outMat.height;
	}

	goto RET;

ERR:
	cnn_network_delete(cnn);

RET:
	return ret;
}

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

int cnn_layer_input_alloc(union CNN_LAYER* layerPtr,
		int inWidth, int inHeight, int batch)
{
	int ret = CNN_NO_ERROR;
	int outRows, outCols;

	// Find allocate size
	outRows = batch;
	outCols = inWidth * inHeight;

	// Allocate memory
	cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 0), ret, ERR);

	// Assign value
	layerPtr->outMat.width = inWidth;
	layerPtr->outMat.height = inHeight;

	goto RET;

ERR:
	cnn_layer_input_delete(layerPtr);

RET:
	return ret;
}

int cnn_layer_afunc_alloc(struct CNN_LAYER_AFUNC* layerPtr,
		int inWidth, int inHeight, int batch)
{
	int ret = CNN_NO_ERROR;
	int outRows, outCols;
	int gradRows, gradCols;

	// Find allocate size
	outRows = batch;
	outCols = inWidth * inHeight;

	gradRows = outCols * batch;
	gradCols = outCols;

	// Allocate memory
	cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret, ERR);
	cnn_run(cnn_mat_alloc(&layerPtr->gradMat, gradRows, gradCols, 0), ret, ERR);
	cnn_run(cnn_mat_alloc(&layerPtr->buf, outRows, outCols, 0), ret, ERR);

	// Assign value
	layerPtr->outMat.width = inWidth;
	layerPtr->outMat.height = inHeight;

	goto RET;

ERR:
	cnn_layer_afunc_delete(layerPtr);

RET:
	return ret;
}

int cnn_layer_fc_alloc(struct CNN_LAYER_FC* layerPtr,
		int inWidth, int inHeight, int outSize, int batch)
{
	int ret = CNN_NO_ERROR;
	int outRows, outCols; // Output matrix size
	int wRows, wCols; // Weight matrix size
	int bRows, bCols; // Bias matrix size

	// Find allocate size
	outRows = batch;
	outCols = outSize;

	wRows = inWidth * inHeight;
	wCols = outSize;

	bRows = 1;
	bCols = outSize;

	// Allocate memory
	cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret, ERR);
	cnn_run(cnn_mat_alloc(&layerPtr->weight, wRows, wCols, 1), ret, ERR);
	cnn_run(cnn_mat_alloc(&layerPtr->bias, bRows, bCols, 1), ret, ERR);

	// Assign value
	layerPtr->outMat.width = outCols;
	layerPtr->outMat.height = 1;

	goto RET;

ERR:
	cnn_layer_fc_delete(layerPtr);

RET:
	return ret;
}

int cnn_layer_conv_alloc(struct CNN_LAYER_CONV* layerPtr,
		int inWidth, int inHeight, int size, int batch)
{
	int ret = CNN_NO_ERROR;
	int outRows, outCols; // Output matrix size
	int outWidth, outHeight; // Valid convolution output size
	int bRows, bCols; // Bias matrix size

	// Find output image size
	outWidth = inWidth - size + 1;
	outHeight = inHeight - size + 1;

	// Checking
	if(outWidth <= 0 || outHeight <= 0)
	{
		ret = CNN_INVALID_SHAPE;
		goto RET;
	}

	// Find allocate size
	outRows = batch;
	outCols = outWidth * outHeight;

	bRows = 1;
	bCols = outCols;

	// Allocate memory
	cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret, ERR);
	cnn_run(cnn_mat_alloc(&layerPtr->kernel, size, size, 1), ret, ERR);
	cnn_run(cnn_mat_alloc(&layerPtr->bias, bRows, bCols, 1), ret, ERR);

	// Assing value
	layerPtr->outMat.width = outWidth;
	layerPtr->outMat.height = outHeight;

	goto RET;

ERR:
	cnn_layer_conv_delete(layerPtr);

RET:
	return ret;
}

