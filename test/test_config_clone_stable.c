#include <stdio.h>

#include <cnn.h>
#include <cnn_types.h>

int main()
{
	int ret;
	cnn_config_t cfg = NULL;
	cnn_config_t cpy = NULL;

	ret = cnn_config_create(&cfg);
	if(ret < 0)
	{
		printf("cnn_config_create() failed with error: %d\n", ret);
		return -1;
	}

	// Test set config
	cnn_config_set_input_size(cfg, 640, 480);
	cnn_config_set_outputs(cfg, 3);

	cnn_config_set_layers(cfg, 5);
	cnn_config_set_convolution(cfg, 0, 1, 3);
	cnn_config_set_full_connect(cfg, 1, 32);
	cnn_config_set_activation(cfg, 2, CNN_RELU);
	cnn_config_set_full_connect(cfg, 3, 128);
	cnn_config_set_activation(cfg, 4, CNN_SOFTMAX);

	// Clone cnn setting
	while(1)
	{
		ret = cnn_config_clone(&cpy, cfg);
		if(ret < 0)
		{
			printf("cnn_config_clone() failed with error: %d\n", ret);
			return -1;
		}

		cnn_config_delete(cpy);
	}

	// Cleanup
	cnn_config_delete(cfg);
	return 0;
}
