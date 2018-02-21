#include <stdio.h>

#include <cnn.h>
#include <cnn_types.h>

#define INPUT_WIDTH 32
#define INPUT_HEIGHT 32
#define BATCH 7

#define test(func) \
	ret = func; \
	if(ret < 0) \
	{ \
		printf("%s failed with error: %d\n", #func, ret); \
		return -1; \
	}

int main()
{
	int i;
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
	test(cnn_config_set_input_size(cfg, INPUT_WIDTH, INPUT_HEIGHT, 1));
	test(cnn_config_set_batch_size(cfg, BATCH));
	test(cnn_config_set_layers(cfg, 11));

	i = 1;
	test(cnn_config_set_convolution (cfg, i++, 2, 3));
	test(cnn_config_set_pooling     (cfg, i++, 2, CNN_POOL_MAX, 2));
	test(cnn_config_set_activation  (cfg, i++, CNN_RELU));
	test(cnn_config_set_convolution (cfg, i++, 2, 3));
	test(cnn_config_set_pooling     (cfg, i++, 2, CNN_POOL_MAX, 2));
	test(cnn_config_set_activation  (cfg, i++, CNN_RELU));
	test(cnn_config_set_full_connect(cfg, i++, 16));
	test(cnn_config_set_dropout		(cfg, i++, 0.5));
	test(cnn_config_set_full_connect(cfg, i++, 2));
	test(cnn_config_set_activation  (cfg, i++, CNN_SOFTMAX));

	while(1)
	{
		// Create cnn
		ret = cnn_create(&cnn, cfg);
		if(ret < 0)
		{
			printf("cnn_create() failed with error: %d\n", ret);
			return -1;
		}

		// Cleanup
		cnn_delete(cnn);
	}

	cnn_config_delete(cfg);

	return 0;
}
