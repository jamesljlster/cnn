#include <stdio.h>

#include <cnn.h>
#include <cnn_private.h>
#include <cnn_types.h>

#define INPUT_WIDTH 180
#define INPUT_HEIGHT 150

#define print_mat_info(str, matVar) \
	printf("%s: %dx%d, %p, %p\n", str, matVar.rows, matVar.cols, matVar.mat, matVar.grad)

#define test_mat(matVar) \
{ \
	int _i; \
	for(_i = 0; _i < matVar.rows * matVar.cols; _i++) \
	{ \
		matVar.mat[_i] = -1; \
		if(matVar.grad != NULL) \
		{ \
			matVar.grad[_i] = 1; \
		} \
	} \
}

void check_cnn_arch(cnn_t cnn)
{
	int i;

	printf("Input image size: %dx%d\n\n", INPUT_WIDTH, INPUT_HEIGHT);

	for(i = 0; i < cnn->cfg.layers; i++)
	{
		printf("Layer %d config:\n", i);
		switch(cnn->cfg.layerCfg[i].type)
		{
			case CNN_LAYER_INPUT:
				printf("The layer is input\n");
				printf("Output size: %dx%d\n", cnn->layerList[i].outMat.width,
						cnn->layerList[i].outMat.height);

				print_mat_info("Output mat", cnn->layerList[i].outMat.data);
				test_mat(cnn->layerList[i].outMat.data);
				break;

			case CNN_LAYER_FC:
				printf("The layer is fully connected\n");
				printf("Size: %d\n", cnn->cfg.layerCfg[i].fc.size);
				printf("Output size: %dx%d\n", cnn->layerList[i].outMat.width,
						cnn->layerList[i].outMat.height);

				print_mat_info("Output mat", cnn->layerList[i].outMat.data);
				print_mat_info("Weight mat", cnn->layerList[i].fc.weight);
				print_mat_info("Bias mat", cnn->layerList[i].fc.bias);

				test_mat(cnn->layerList[i].outMat.data);
				test_mat(cnn->layerList[i].fc.weight);
				test_mat(cnn->layerList[i].fc.bias);
				break;

			case CNN_LAYER_AFUNC:
				printf("The layer is activation function\n");
				printf("ID: %d\n", cnn->cfg.layerCfg[i].aFunc.id);
				printf("Output size: %dx%d\n", cnn->layerList[i].outMat.width,
						cnn->layerList[i].outMat.height);

				print_mat_info("Output mat", cnn->layerList[i].outMat.data);
				print_mat_info("Grad mat", cnn->layerList[i].aFunc.gradMat);

				test_mat(cnn->layerList[i].outMat.data);
				test_mat(cnn->layerList[i].aFunc.gradMat);
				break;

			case CNN_LAYER_CONV:
				printf("The layer is convolution\n");
				printf("Dimension: %d\n", cnn->cfg.layerCfg[i].conv.dim);
				printf("Size: %d\n", cnn->cfg.layerCfg[i].conv.size);
				printf("Output size: %dx%d\n", cnn->layerList[i].outMat.width,
						cnn->layerList[i].outMat.height);

				print_mat_info("Output mat", cnn->layerList[i].outMat.data);
				print_mat_info("Kernel mat", cnn->layerList[i].conv.kernel);
				print_mat_info("Bias mat", cnn->layerList[i].conv.bias);

				test_mat(cnn->layerList[i].outMat.data);
				test_mat(cnn->layerList[i].conv.kernel);
				test_mat(cnn->layerList[i].conv.bias);
				break;

			default:
				printf("Not a cnn layer config?!\n");
		}
		printf("\n");
	}
}

int main()
{
	int ret;
	cnn_config_t cfg = NULL;
	cnn_t cnn = NULL;

	ret = cnn_config_create(&cfg);
	if(ret < 0)
	{
		printf("cnn_config_create() failed with error: %d\n", ret);
		return -1;
	}

	// Set config
	cnn_config_set_input_size(cfg, INPUT_WIDTH, INPUT_HEIGHT);

	cnn_config_set_layers(cfg, 5);
	cnn_config_set_convolution(cfg, 0, 1, 3);
	cnn_config_set_activation(cfg, 1, CNN_RELU);
	cnn_config_set_full_connect(cfg, 2, 128);
	cnn_config_set_full_connect(cfg, 3, 128);
	cnn_config_set_activation(cfg, 4, CNN_SOFTMAX);

	// Create cnn
	ret = cnn_create(&cnn, cfg);
	if(ret < 0)
	{
		printf("cnn_create() failed with error: %d\n", ret);
		return -1;
	}

	// Check cnn arch
	check_cnn_arch(cnn);

	printf("Press enter to continue...");
	getchar();

	// Cleanup
	cnn_delete(cnn);
	cnn_config_delete(cfg);

	return 0;
}
