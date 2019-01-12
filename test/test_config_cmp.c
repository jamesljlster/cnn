#include <cnn.h>
#include <stdio.h>

#include "test.h"

int main(int argc, char* argv[])
{
    cnn_config_t src = NULL;
    cnn_config_t cmp = NULL;

    // Check argument
    if (argc < 3)
    {
        printf("Usage: test_config_cmp <cfg1> <cfg2>\n");
        return -1;
    }

    test(cnn_config_import(&src, argv[1]));
    test(cnn_config_import(&cmp, argv[2]));
    test(cnn_config_compare(src, cmp));

    return 0;
}
