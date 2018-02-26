#include <stdio.h>

#include <cnn.h>
#include <cnn_private.h>
#include <cnn_types.h>

#define CFG_PATH "test.xml"

#define test(func) \
	ret = func; \
	if(ret < 0) \
	{ \
		printf("%s failed with error: %d\n", #func, ret); \
		return -1; \
	}

void check_cnn_layer_config(cnn_config_t cfg, int layerIndex)
{
	printf("Layer %d config:\n", layerIndex);
	switch(cfg->layerCfg[layerIndex].type)
	{
		case CNN_LAYER_INPUT:
			printf("The layer is input\n");
			break;

		case CNN_LAYER_FC:
			printf("The layer is fully connected\n");
			printf("Size: %d\n", cfg->layerCfg[layerIndex].fc.size);
			break;

		case CNN_LAYER_AFUNC:
			printf("The layer is activation function\n");
			printf("ID: %d\n", cfg->layerCfg[layerIndex].aFunc.id);
			break;

		case CNN_LAYER_CONV:
			printf("The layer is convolution\n");
			printf("Dimension: %d\n", cfg->layerCfg[layerIndex].conv.dim);
			printf("Filter: %d\n", cfg->layerCfg[layerIndex].conv.filter);
			printf("Size: %d\n", cfg->layerCfg[layerIndex].conv.size);
			break;

		case CNN_LAYER_POOL:
			printf("The layer is pooling\n");
			printf("Dimension: %d\n", cfg->layerCfg[layerIndex].pool.dim);
			printf("Size: %d\n", cfg->layerCfg[layerIndex].pool.size);
			break;

		case CNN_LAYER_DROP:
			printf("The layer is dropout\n");
			printf("Rate: %g\n", cfg->layerCfg[layerIndex].drop.rate);
			break;

		default:
			printf("Not a cnn layer config?!\n");
	}
	printf("\n");
}

int main()
{
	int i;
	int ret;
	cnn_config_t cfg = NULL;

	ret = cnn_config_create(&cfg);
	if(ret < 0)
	{
		printf("cnn_config_create() failed with error: %d\n", ret);
		return -1;
	}

	// Check default setting
	printf("=== Default CNN Config ===\n");
	for(i = 0; i < cfg->layers; i++)
	{
		check_cnn_layer_config(cfg, i);
	}

	// Test set config
	test(cnn_config_set_input_size(cfg, 640, 480, 1));
	test(cnn_config_set_layers(cfg, 11));

	i = 1;
	test(cnn_config_set_convolution	(cfg, i++, 1, 32, 3));
	test(cnn_config_set_pooling		(cfg, i++, 2, CNN_POOL_MAX, 2));
	test(cnn_config_set_activation	(cfg, i++, CNN_RELU));
	test(cnn_config_set_convolution	(cfg, i++, 1, 64, 3));
	test(cnn_config_set_pooling		(cfg, i++, 2, CNN_POOL_MAX, 2));
	test(cnn_config_set_activation	(cfg, i++, CNN_RELU));
	test(cnn_config_set_full_connect(cfg, i++, 128));
	test(cnn_config_set_dropout		(cfg, i++, 0.5));
	test(cnn_config_set_full_connect(cfg, i++, 2));
	test(cnn_config_set_activation	(cfg, i++, CNN_SOFTMAX));

	// Check default setting
	printf("=== Modify CNN Config ===\n");
	for(i = 0; i < cfg->layers; i++)
	{
		check_cnn_layer_config(cfg, i);
	}

	// Export config
	ret = cnn_config_export(cfg, CFG_PATH);
	if(ret != CNN_NO_ERROR)
	{
		printf("cnn_config_export() failed with error: %d\n", ret);
	}

	// Cleanup
	cnn_config_delete(cfg);
	return 0;
}
