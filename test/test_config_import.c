#include <cnn.h>
#include <cnn_parse.h>

#define EXPORT_PATH "test_config_import.xml"

#define test(func) \
	ret = func; \
	if(ret < 0) \
	{ \
		printf("%s failed with error: %d\n", #func, ret); \
		return -1; \
	}

int main(int argc, char* argv[])
{
	int i;
	int ret;
	cnn_config_t cfg;

	int layers;
	int size;
	int filter;
	cnn_layer_t layerType;
	cnn_pool_t poolType;
	cnn_dim_t dim;
	cnn_activ_t id;
	int width, height, channel;
	int batch;
	float rate;

	if(argc < 2)
	{
		printf("Assign a config xml to run the program\n");
		return -1;
	}

	test(cnn_config_import(&cfg, argv[1]));

	// Show cnn arch
	cnn_config_get_input_size(cfg, &width, &height, &channel);
	cnn_config_get_batch_size(cfg, &batch);
	printf("Width: %d, Height: %d, Channel: %d\n", width, height, channel);
	printf("Batch: %d\n", batch);
	printf("\n");

	cnn_config_get_layers(cfg, &layers);
	for(i = 0; i < layers; i++)
	{
		printf("=== Layer %d Config ===\n", i);
		test(cnn_config_get_layer_type(cfg, i, &layerType));
		switch(layerType)
		{
			case CNN_LAYER_INPUT:
				printf("Type: Input\n");
				break;

			case CNN_LAYER_FC:
				test(cnn_config_get_full_connect(cfg, i, &size));
				printf("Type: Fully Connected\n");
				printf("Size: %d\n", size);
				break;

			case CNN_LAYER_ACTIV:
				test(cnn_config_get_activation(cfg, i, &id));
				printf("Type: Activation\n");
				printf("ID: %d\n", id);
				break;

			case CNN_LAYER_CONV:
				test(cnn_config_get_convolution(cfg, i, &dim, &filter, &size));
				printf("Type: Convolution\n");
				printf("Dimension: %d\n", dim);
				printf("filter: %d\n", filter);
				printf("Size: %d\n", size);
				break;

			case CNN_LAYER_POOL:
				test(cnn_config_get_pooling(cfg, i, &dim, &poolType, &size));
				printf("Type: Pooling\n");
				printf("Dimension: %d\n", dim);
				printf("Pooling Type: %d\n", poolType);
				printf("Size: %d\n", size);
				break;

			case CNN_LAYER_DROP:
				test(cnn_config_get_dropout(cfg, i, &rate));
				printf("Type: Dropout\n");
				printf("Rate: %g\n", rate);
				break;
		}

		printf("\n");
	}

	test(cnn_config_export(cfg, EXPORT_PATH));

	return 0;
}
