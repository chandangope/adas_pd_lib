#ifndef LIB_IVADASPD_H_
#define LIB_IVADASPD_H_

#include <vector>
#include <string>

void* ivAdasPDEngine_Init(std::string modelFilePath);

int ivAdasPDEngine_classifyimage_bbox(void* p_engine,
	const unsigned char* pixelData, int numrows, int numcols, int* classOut, float* confOut);

int ivAdasPDEngine_Close(void* p_engine);

#endif