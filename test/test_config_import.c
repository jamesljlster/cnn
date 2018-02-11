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

	test(cnn_config_create(&cfg));
	test(cnn_import_root(cfg, NULL, argv[1]));
	test(cnn_config_export(cfg, EXPORT_PATH));

	return 0;
}
