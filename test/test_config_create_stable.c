#include <stdio.h>

#include <cnn.h>
#include <cnn_types.h>

int main()
{
    int ret;
    cnn_config_t cfg = NULL;

    while (1)
    {
        ret = cnn_config_create(&cfg);
        if (ret < 0)
        {
            printf("cnn_config_create() failed with error: %d\n", ret);
            return -1;
        }

        // Cleanup
        cnn_config_delete(cfg);
    }

    return 0;
}
