#ifndef __TEST_H__
#define __TEST_H__

#define test(func) \
{ \
	int __retVal = func; \
	if(__retVal != CNN_NO_ERROR) \
	{ \
		fprintf(stderr, "%s(), %d: %s failed with error: %d\n", __FUNCTION__, __LINE__, \
				#func, __retVal); \
		return -1; \
	} \
}

#endif
