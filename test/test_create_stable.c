#include <stdio.h>

#include <cnn.h>
#include <cnn_types.h>

#define INPUT_WIDTH 180
#define INPUT_HEIGHT 150

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
