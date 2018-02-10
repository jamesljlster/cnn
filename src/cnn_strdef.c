
#include <string.h>

#include "cnn.h"
#include "cnn_strdef.h"

const char* cnn_str_list[] = {
	"config",
	"cnn_model",
	"arch",
	"layer",
	"type",
	"dim",
	"size",
	"channel",
	"batch",
	"learning_rate",
	"value",
	"input",
	"width",
	"height",
};

int cnn_strdef_get_id(const char* str)
{
	int i;
	int ret = CNN_PARSE_FAILED;

	for(i = 0; i < CNN_STR_AMOUNT; i++)
	{
		ret = strcmp(str, cnn_str_list[i]);
		if(ret == 0)
		{
			ret = i;
			goto RET;
		}
	}

RET:
	return ret;
}
