#ifndef _DEBUG_H_
#define _DEBUG_H_

#ifdef DEBUG
#include <stdio.h>
#define LOG(msg, ...) 							\
	fprintf(stderr, "%s(): ", __FUNCTION__);	\
	fprintf(stderr, msg, ##__VA_ARGS__);		\
	fprintf(stderr, "\n");

#define LOG_MAT(msg, src, rows, cols, ...) \
	fprintf(stderr, "%s(): ", __FUNCTION__); \
	fprintf(stderr, msg, ##__VA_ARGS__); \
	fprintf(stderr, "\n"); \
	for(int _i = 0; _i < rows; _i++) \
	{ \
		for(int _j = 0; _j < cols; _j++) \
		{ \
			fprintf(stderr, "%+5g", src[_i * cols + _j]); \
			if(_j < cols - 1) \
			{ \
				fprintf(stderr, "  "); \
			} \
			else \
			{ \
				fprintf(stderr, "\n"); \
			} \
		} \
	} \
	fprintf(stderr, "\n");

#else
#define	LOG(msg, ...)
#define LOG_MAT(msg, src, rows, cols, ...)
#endif

#endif
