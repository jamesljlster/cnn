#include <stdio.h>

#include <cnn.h>
#include <cnn_types.h>

#define INPUT_WIDTH 180
#define INPUT_HEIGHT 150

void check_cnn_arch(cnn_t cnn)
{
	int i;

	printf("Input image size: %dx%d\n\n", INPUT_WIDTH, INPUT_HEIGHT);

	for(i = 0; i < cnn->cfg.layers; i++)
	{
		printf("Layer %d config:\n", i);
		switch(cnn->cfg.layerCfg[i].type)
		{
			case CNN_LAYER_FC:
				printf("The layer is fully connected\n");
				printf("Size: %d\n", cnn->cfg.layerCfg[i].fc.size);
				printf("Output size: %dx%d\n", cnn->layerList[i].outMat.width,
						cnn->layerList[i].outMat.height);
				break;

			case CNN_LAYER_AFUNC:
				printf("The layer is activation function\n");
				printf("ID: %d\n", cnn->cfg.layerCfg[i].aFunc.id);
				printf("Output size: %dx%d\n", cnn->layerList[i].outMat.width,
						cnn->layerList[i].outMat.height);
				break;

			case CNN_LAYER_CONV:
				printf("The layer is convolution\n");
				printf("Dimension: %d\n", cnn->cfg.layerCfg[i].conv.dim);
				printf("Size: %d\n", cnn->cfg.layerCfg[i].conv.size);
				printf("Output size: %dx%d\n", cnn->layerList[i].outMat.width,
						cnn->layerList[i].outMat.height);
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
	cnn_config_set_outputs(cfg, 3);

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

	printf("Press any key to continue...");
	getchar();

	// Cleanup
	cnn_delete(cnn);
	cnn_config_delete(cfg);

	return 0;
}
