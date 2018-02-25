#include <cnn.h>
#include <cnn_parse.h>

#define EXPORT_PATH "test_clone.xml"

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
	cnn_t cnn;
	cnn_t cpy;

	if(argc < 2)
	{
		printf("Assign a config xml to run the program\n");
		return -1;
	}

	test(cnn_import(&cnn, argv[1]));
	test(cnn_clone(&cpy, cnn));
	test(cnn_export(cpy, EXPORT_PATH));

	return 0;
}
