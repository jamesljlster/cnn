#include <stdio.h>

#include <cnn.h>
#include <cnn_types.h>

#include "test.h"

int main()
{
    cnn_config_t cfg = NULL;

    while (1)
    {
        test(cnn_config_create(&cfg));

        // Cleanup
        cnn_config_delete(cfg);
    }

    return 0;
}
