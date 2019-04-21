#include <cnn.h>
#include <cnn_parse.h>

#include "test.h"

#define EXPORT_PATH "test_config_import.xml"

int main(int argc, char* argv[])
{
    cnn_config_t cfg;

    if (argc < 2)
    {
        printf("Assign a config xml to run the program\n");
        return -1;
    }

    while (1)
    {
        test(cnn_config_import(&cfg, argv[1]));
        cnn_config_delete(cfg);
    }

    return 0;
}
