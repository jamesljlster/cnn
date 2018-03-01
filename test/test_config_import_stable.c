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
	int ret;
	cnn_config_t cfg;

	if(argc < 2)
	{
		printf("Assign a config xml to run the program\n");
		return -1;
	}

	while(1)
	{
		test(cnn_config_import(&cfg, argv[1]));
		cnn_config_delete(cfg);
	}

	return 0;
}