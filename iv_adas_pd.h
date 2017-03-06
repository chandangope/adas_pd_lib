#ifndef LIB_IVADASPD_H_
#define LIB_IVADASPD_H_

#include <vector>
#include <string>

#define MAX_IVADASPD_BBOXES (100)

typedef struct _ivAdasPDBbox
{
  int width;
  int height;
  int topLeftX;
  int topLeftY;
  float detectionConf;
}ivAdasPDBbox;

void* ivAdasPDEngine_Init(std::string modelFilePath);

int ivAdasPDEngine_classifyimage_bbox(void* p_engine,
	const unsigned char* pixelData, int numrows, int numcols, int* classOut, float* confOut);

int ivAdasPDEngine_classifyBboxesInImage(void* p_engine, const unsigned char* pixelData, int numrows, int numcols,
	ivAdasPDBbox* boxes, int numboxes);

int ivAdasPDEngine_detectInImage(void* p_engine, const unsigned char* intensityData, int numrows, int numcols,
	ivAdasPDBbox* boxes, int* numboxes, ivAdasPDBbox* roiWindow);

int ivAdasPDEngine_Close(void* p_engine);

#endif