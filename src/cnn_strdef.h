#ifndef __CNN_STRDEF_H__
#define __CNN_STRDEF_H__

// CNN reserve words enumeration
enum CNN_STR_LIST
{
	CNN_STR_CONFIG,
	CNN_STR_MODEL,

	CNN_STR_AMOUNT
};

// CNN reserve words list
extern const char* cnn_str_list[];

#ifdef __cplusplus
extern "C" {
#endif

int cnn_strdef_get_id(const char* str);

#ifdef __cplusplus
}
#endif

#endif
