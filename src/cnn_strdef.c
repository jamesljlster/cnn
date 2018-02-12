
#include <string.h>

#include "cnn.h"
#include "cnn_strdef.h"

const char* cnn_str_list[] = {
	"config",
	"cnn_model",
	"network",
	"layer",
	"index",
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
	"fc",
	"activ",
	"conv",
	"pool",
	"id",
	"pool_type",
	"max",
	"min",
	"kernel",
	"bias",
	"weight",
	"1d",
	"2d"
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
