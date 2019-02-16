#include <cnn.h>
#include <cnn_parse.h>

#include "test.h"

#define EXPORT_PATH "test_import.xml"

int main(int argc, char* argv[])
{
    cnn_t cnn;

    if (argc < 2)
    {
        printf("Assign a config xml to run the program\n");
        return -1;
    }

    test(cnn_init());

    test(cnn_import(&cnn, argv[1]));
    test(cnn_export(cnn, EXPORT_PATH));

    cnn_deinit();

    return 0;
}
