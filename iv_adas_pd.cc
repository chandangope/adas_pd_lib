
//
// Reference:
// 1. load mnist data is reference by https://github.com/krck/MNIST_Loader
// 2. how to use tensorflow c++ api to load the graph is reference by 
// https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f#.chz3r27xt
// ---------------------------------------------------------------------------------
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "iv_adas_pd.h"

#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;
using namespace chrono;
using namespace tensorflow;

typedef struct _ivAdasPDTFEngine
{
  Session* m_session;
  unsigned char* m_bboxData; //Buffer for one bounding box 72x36
  unsigned char* m_bboxData2; //Buffer for one bounding box 96x48
  unsigned char* m_bboxData3; //Buffer for one bounding box 128x64
  unsigned char* m_bboxData4; //Buffer for one bounding box 256x128
  unsigned char* m_bboxData5; //Buffer for one bounding box 192x96

  unsigned char** m_boxesData; //Buffer for all bounding boxes 72x36 which will be sent for classification
  int* m_boxClasses; //Buffer for boxes classification results
  float* m_boxConfs; //Buffer for boxes classification confidence

  //std::vector<float> m_pixelDataFloat;
  float* m_pixelDataFloat; //Float buffer for all bounding boxes 72x36 which will be sent for classification

}ivAdasPDTFEngine;




int ivAdasPDEngine_classify_bboxes(void* p_engine, const unsigned char** boxesData, int numboxes, 
  int bboxheight, int bboxwidth, int* classOut, float* confOut);
int getROIBoxes(int xOffset, int yOffset, ivAdasPDBbox* roiBoxes, int* numBoxes, ivAdasPDBbox* roiWindow);





// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(string graph_file_name, Session** session)
{
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(ReadBinaryProto(Env::Default(), graph_file_name, &graph_def));
  TF_RETURN_IF_ERROR(NewSession(SessionOptions(), session));
  TF_RETURN_IF_ERROR((*session)->Create(graph_def));
  return Status::OK();
}

int resize_nn(const unsigned char* imgDataIn, int widthIn, int heightIn,
              unsigned char* imgDataOut, int widthOut, int heightOut)
{
  unsigned char* ptrOut = imgDataOut;
  float x_scaleFactor = (widthIn-1)/(float)widthOut;
  float y_scaleFactor = (heightIn-1)/(float)heightOut;

  for(int row = 0; row < heightOut; row++){
    for(int col = 0; col < widthOut; col++){
      int xIn = (int)(x_scaleFactor*col);
      int yIn = (int)(y_scaleFactor*row);
      int offset = yIn*widthIn + xIn;
      *ptrOut = *(imgDataIn + offset);
      ptrOut++;
    }
  }

  return 0;
}


void* ivAdasPDEngine_Init(std::string modelFilePath)
{
  cout << "Creating session and adding graph" << "\n";
  ivAdasPDTFEngine* p_engine = new ivAdasPDTFEngine;
  p_engine->m_bboxData = new unsigned char[72*36];
  p_engine->m_bboxData2 = new unsigned char[96*48];
  p_engine->m_bboxData3 = new unsigned char[128*64];
  p_engine->m_bboxData4 = new unsigned char[256*128];
  p_engine->m_bboxData5 = new unsigned char[192*96];

  p_engine->m_boxesData = new unsigned char*[MAX_IVADASPD_BBOXES];
  p_engine->m_boxClasses = new int[MAX_IVADASPD_BBOXES];
  p_engine->m_boxConfs = new float[MAX_IVADASPD_BBOXES];
  for(int i=0; i<MAX_IVADASPD_BBOXES; i++){
    p_engine->m_boxesData[i] = new unsigned char[72*36];
  }

  //p_engine->m_pixelDataFloat.reserve(72*36*MAX_IVADASPD_BBOXES);
  p_engine->m_pixelDataFloat = new float[72*36*MAX_IVADASPD_BBOXES];

  Status status = LoadGraph(modelFilePath, &(p_engine->m_session));
  if (!status.ok())
  {
    cout << status.ToString() << "\n";
    delete p_engine;
    return 0;
  }
  else
  {
    return p_engine;
  }
}

int ivAdasPDEngine_Close(void* p_engine)
{
  if(p_engine == 0)
  {
    cout << "Null p_engine" << "\n";
    return 1;
  }

  // Close the session to release the resources associated with this session
  ivAdasPDTFEngine* engine = reinterpret_cast<ivAdasPDTFEngine*>(p_engine);
  delete [](engine->m_bboxData);
  delete [](engine->m_bboxData2);
  delete [](engine->m_bboxData3);
  delete [](engine->m_bboxData4);
  delete [](engine->m_bboxData5);

  for(int i=0; i<MAX_IVADASPD_BBOXES; i++){
    delete []engine->m_boxesData[i];
  }
  delete []engine->m_boxesData;
  delete []engine->m_boxClasses;
  delete []engine->m_boxConfs;

  //engine->m_pixelDataFloat.clear();
  delete []engine->m_pixelDataFloat;

  Session* m_session = engine->m_session;
  m_session->Close();
  cout << "Closing the session" << endl;
  return 0;
}

int ivAdasPDEngine_detectInImage(void* p_engine, const unsigned char* intensityData, int numrows, int numcols,
  ivAdasPDBbox* boxes, int* numboxes, ivAdasPDBbox* roiWindow)
{
  if(p_engine == 0)
  {
    cout << "Null p_engine" << "\n"; return 1;
  }
  if(intensityData == 0)
  {
    cout << "Null intensityData" << "\n"; return 1;
  }
  if(boxes == 0)
  {
    cout << "Null boxes" << "\n"; return 1;
  }
  if(numrows != 720 || numcols != 1280)
  {
    cout << "Unexpected number of rows or cols: Image dimension should be 720p" << "\n"; return 1;
  }


  getROIBoxes(320, 180, boxes, numboxes, roiWindow);
  int res = ivAdasPDEngine_classifyBboxesInImage(p_engine, intensityData, 720, 1280, boxes, *numboxes);
  if(res != 0)
  {
    cout << "ivAdasPDEngine_classifyBboxesInImage() failed" << "\n";
    return 1;
  }

  return 0;
}

int getROIBoxes(int xOffset, int yOffset, ivAdasPDBbox* roiBoxes, int* numBoxes, ivAdasPDBbox* roiWindow)
{
  int roiWindow_bottomRightX = -1;
  int roiWindow_bottomRightY = -1;

  int marginX, topLeftX, topLeftY, width, height;

  *numBoxes = 0;
  roiWindow->topLeftX = 100000;
  roiWindow->topLeftY = 100000;

  marginX = 200;
  topLeftX = marginX+xOffset;
  topLeftY = 140+yOffset; //130
  width = 64;
  height = 128;
  while(true){
    if(topLeftX+width > 640+xOffset-marginX) break;
    roiBoxes[*numBoxes].width = width;
    roiBoxes[*numBoxes].height = height;
    roiBoxes[*numBoxes].topLeftY = topLeftY;
    roiBoxes[*numBoxes].topLeftX = topLeftX;

    if(roiWindow->topLeftX > roiBoxes[*numBoxes].topLeftX)
      roiWindow->topLeftX = roiBoxes[*numBoxes].topLeftX;
    if(roiWindow->topLeftY > roiBoxes[*numBoxes].topLeftY)
      roiWindow->topLeftY = roiBoxes[*numBoxes].topLeftY;
    if(roiWindow_bottomRightX < roiBoxes[*numBoxes].topLeftX + roiBoxes[*numBoxes].width)
      roiWindow_bottomRightX = roiBoxes[*numBoxes].topLeftX + roiBoxes[*numBoxes].width;
    if(roiWindow_bottomRightY < roiBoxes[*numBoxes].topLeftY + roiBoxes[*numBoxes].height)
      roiWindow_bottomRightY = roiBoxes[*numBoxes].topLeftY + roiBoxes[*numBoxes].height;

    (*numBoxes)++;
    topLeftX += width/3;
  }

  /*marginX = 200;
  topLeftX = marginX+xOffset;
  topLeftY = 100+yOffset;
  width = 96;
  height = 192;
  while(true){
    if(topLeftX+width > 640+xOffset-marginX) break;
    roiBoxes[*numBoxes].width = width;
    roiBoxes[*numBoxes].height = height;
    roiBoxes[*numBoxes].topLeftY = topLeftY;
    roiBoxes[*numBoxes].topLeftX = topLeftX;

    if(roiWindow->topLeftX > roiBoxes[*numBoxes].topLeftX)
      roiWindow->topLeftX = roiBoxes[*numBoxes].topLeftX;
    if(roiWindow->topLeftY > roiBoxes[*numBoxes].topLeftY)
      roiWindow->topLeftY = roiBoxes[*numBoxes].topLeftY;
    if(roiWindow_bottomRightX < roiBoxes[*numBoxes].topLeftX + roiBoxes[*numBoxes].width)
      roiWindow_bottomRightX = roiBoxes[*numBoxes].topLeftX + roiBoxes[*numBoxes].width;
    if(roiWindow_bottomRightY < roiBoxes[*numBoxes].topLeftY + roiBoxes[*numBoxes].height)
      roiWindow_bottomRightY = roiBoxes[*numBoxes].topLeftY + roiBoxes[*numBoxes].height;

    (*numBoxes)++;
    topLeftX += width/2;
  }*/


  roiWindow->width = roiWindow_bottomRightX - roiWindow->topLeftX;
  roiWindow->height = roiWindow_bottomRightY - roiWindow->topLeftY;
  roiWindow->detectionConf = -1;

  return 0;
}


int ivAdasPDEngine_classifyBboxesInImage(void* p_engine, const unsigned char* pixelData, int numrows, int numcols,
  ivAdasPDBbox* boxes, int numboxes)
{
  if(p_engine == 0)
  {
    cout << "Null p_engine" << "\n"; return 1;
  }
  if(numrows != 720 || numcols != 1280)
  {
    cout << "Unexpected number of rows or cols (Expected 720p)" << "\n"; return 1;
  }

  cout << "Numboxes to classify: " << numboxes << "\n";
  ivAdasPDTFEngine* engine = reinterpret_cast<ivAdasPDTFEngine*>(p_engine);
  assert(numboxes <= MAX_IVADASPD_BBOXES);
  const unsigned char* rowStart = 0;
  unsigned char* databuffer = 0;
  for(int i=0; i<numboxes; i++){
    boxes[i].detectionConf = 0;

    if(boxes[i].width == 128 && boxes[i].height == 256)
      databuffer = engine->m_bboxData4; //buffer to hold 256x128
    else if(boxes[i].width == 96 && boxes[i].height == 192)
      databuffer = engine->m_bboxData5; //buffer to hold 192x96
    else if(boxes[i].width == 64 && boxes[i].height == 128)
      databuffer = engine->m_bboxData3; //buffer to hold 128x64
    else if(boxes[i].width == 48 && boxes[i].height == 96)
      databuffer = engine->m_bboxData2; //buffer to hold 96x48
    else if(boxes[i].width == 36 && boxes[i].height == 72)
      databuffer = engine->m_bboxData; //buffer to hold 72x36
    else{
      cout << "Invalid bbox size" << "\n";
      return -1;
    }
    
    for(int row = 0; row < boxes[i].height; row++){
      rowStart = pixelData + numcols*(boxes[i].topLeftY+row) + boxes[i].topLeftX;
      for(int col=0; col<boxes[i].width; col++){
        *databuffer = *rowStart;
        databuffer++;
        rowStart++;
      }
    }

    //engine->m_bboxData is populated, now classify it
    if(boxes[i].width == 48 && boxes[i].height == 96) //if 48x96, then resize to 36x72
      resize_nn(engine->m_bboxData2, 48, 96, engine->m_bboxData, 36, 72);
    else if(boxes[i].width == 64 && boxes[i].height == 128) //if 64x128, then resize to 36x72
      resize_nn(engine->m_bboxData3, 64, 128, engine->m_bboxData, 36, 72);
    else if(boxes[i].width == 96 && boxes[i].height == 192) //if 96x192, then resize to 36x72
      resize_nn(engine->m_bboxData5, 96, 192, engine->m_bboxData, 36, 72);
    else if(boxes[i].width == 128 && boxes[i].height == 256) //if 128x256, then resize to 36x72
      resize_nn(engine->m_bboxData4, 128, 256, engine->m_bboxData, 36, 72);
    else{
      //do nothing, bounding box is already 36x72
    }

    std::copy_n(engine->m_bboxData, 36*72, engine->m_boxesData[i]);

    /*int classOut; float confOut;
    int res = ivAdasPDEngine_classifyimage_bbox(p_engine,engine->m_bboxData, 72,36, &classOut, &confOut);
    if(res != 0) return res;
    //cout << "classOut=" << classOut << ", confOut=" << confOut << "\n";
    if(classOut == 1 && confOut > 0.5){
        boxes[i].detectionConf = confOut;
    }*/
  }
  int res = ivAdasPDEngine_classify_bboxes(p_engine, (const unsigned char**)(engine->m_boxesData), numboxes, 72, 36, 
    engine->m_boxClasses, engine->m_boxConfs);
  if(res != 0) return res;
  for(int i=0; i<numboxes; i++){
    if(engine->m_boxClasses[i] == 1 && engine->m_boxConfs[i] > 0.5){
      boxes[i].detectionConf = engine->m_boxConfs[i];
    }
  }

  return 0;
}

int ivAdasPDEngine_classify_bboxes(void* p_engine, const unsigned char** boxesData, int numboxes, 
  int bboxheight, int bboxwidth, int* classOut, float* confOut)
  {
    if(p_engine == 0)
    {
      cout << "Null p_engine" << "\n";
      return 1;
    }

    if(bboxheight != 72 || bboxwidth != 36)
    {
      cout << "Unexpected bboxheight or bboxwidth" << "\n";
      return 1;
    }

    ivAdasPDTFEngine* engine = reinterpret_cast<ivAdasPDTFEngine*>(p_engine);
    Session* m_session = engine->m_session;

    int imageDim = bboxheight*bboxwidth;
    int nTests = numboxes;
    //engine->m_pixelDataFloat.clear(); //Dont forget this!!
    float* ptrpixelDataFloat = engine->m_pixelDataFloat;
    for(int n = 0; n < nTests; n++){
      const unsigned char* boxesDataPtr = boxesData[n];
      for(int k=0; k<imageDim; k++){
        //engine->m_pixelDataFloat.push_back(boxesDataPtr[k]/255.0);
        *ptrpixelDataFloat = boxesDataPtr[k]/255.0;
        ptrpixelDataFloat++;
      }
    }
    // Setup inputs and outputs:
    Tensor x(DT_FLOAT, TensorShape({nTests, imageDim}));
    auto dst = x.flat<float>().data();
    //std::copy_n(engine->m_pixelDataFloat.begin(), imageDim*nTests, dst);
    std::copy_n(engine->m_pixelDataFloat, imageDim*nTests, dst);

    /*Tensor keep_prob(DT_FLOAT, TensorShape({1}));
    auto dst2 = keep_prob.flat<float>().data();
    std::vector<float> keep_probs;
    keep_probs.push_back(1.0);
    std::copy_n(keep_probs.begin(), 1, dst2);*/

    Tensor keep_prob(DT_FLOAT, TensorShape());
    keep_prob.scalar<float>()() = 1.0;

    vector<pair<string, Tensor>> inputs = {
      { "input", x}, {"keep_prob", keep_prob}
    };

    // The session will initialize the outputs
    vector<Tensor> outputs;
    // Run the session, evaluating our "softmax" operation from the graph
    Status status = m_session->Run(inputs, {"softmax"}, {}, &outputs);
    if (!status.ok()) {
      cout << status.ToString() << "\n";
      return 1;
    }else{
      //cout << "Successfully ran session " << "\n";
    }

    int numClasses = 2;
    for (vector<Tensor>::iterator it = outputs.begin() ; it != outputs.end(); ++it){
      auto items = it->shaped<float, 2>({nTests, numClasses});
      for(int i = 0 ; i < nTests ; i++){
        int arg_max = 0;
        float val_max = items(i, 0);
        for (int j = 0; j < numClasses; j++){
          if (items(i, j) > val_max){
            arg_max = j;
            val_max = items(i, j);
          }
        }
        classOut[i] = arg_max;
        confOut[i] = val_max;
      }
    }

    return 0;
  }


int ivAdasPDEngine_classifyimage_bbox(void* p_engine,
  const unsigned char* pixelData, int numrows, int numcols, int* classOut, float* confOut)
{
  if(p_engine == 0)
  {
    cout << "Null p_engine" << "\n";
    return 1;
  }

  if(numrows != 72 || numcols != 36)
  {
    cout << "Unexpected number of rows or cols" << "\n";
    return 1;
  }

  ivAdasPDTFEngine* engine = reinterpret_cast<ivAdasPDTFEngine*>(p_engine);
  Session* m_session = engine->m_session;

  /*cout << "Create session and add graph" << "\n";
  Session* m_session;

  string graph_file_name = "./frozen_graph.pb";
  Status status = LoadGraph(graph_file_name, &m_session);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }*/

  /*Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }*/

  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  /*GraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), "./frozen_graph.pb", &graph_def);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }*/

  // Add the graph to the session
  /*status = session->Create(graph_def);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }*/
  
  //cout << "preparing input data..." << endl;
  // config setting
  int imageDim = numrows*numcols;
  int nTests = 1;

  std::vector<float> pixelDataFloat;
  pixelDataFloat.reserve(imageDim);
  for(int k=0; k<imageDim; k++){
    pixelDataFloat.push_back(pixelData[k]/255.0);
  }
  // Setup inputs and outputs:
  Tensor x(DT_FLOAT, TensorShape({nTests, imageDim}));
  auto dst = x.flat<float>().data();
  //std::copy_n(pixelData.begin(), imageDim, dst);
  std::copy_n(pixelDataFloat.begin(), imageDim, dst);

  Tensor keep_prob(DT_FLOAT, TensorShape({1}));
  auto dst2 = keep_prob.flat<float>().data();
  std::vector<float> keep_probs;
  keep_probs.push_back(1.0);
  std::copy_n(keep_probs.begin(), 1, dst2);

  //cout << "data is ready" << endl;
  vector<pair<string, Tensor>> inputs = {
    { "input", x}, {"keep_prob",keep_prob}
  };

  // The session will initialize the outputs
  vector<Tensor> outputs;
  // Run the session, evaluating our "softmax" operation from the graph
  Status status = m_session->Run(inputs, {"softmax"}, {}, &outputs);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }else{
    //cout << "Successfully ran session " << "\n";
  }

  vector<Tensor>::iterator it = outputs.begin();
  int numClasses = 2;
  auto items = it->shaped<float, 2>({nTests, numClasses});
  int arg_max = 0;
  float val_max = items(0, 0);
  for (int j = 0; j < numClasses; j++)
  {
    if (items(0, j) > val_max)
    {
      arg_max = j;
      val_max = items(0, j);
    }
  }
  *classOut = arg_max;
  *confOut = val_max;

  return 0;
}


#if 0
int ivAdasPDEngine_detectInImage(void* p_engine, const unsigned char* pixelData, int numrows, int numcols,
  ivAdasPDBbox* detectedBoxes, int* numDetectedBoxes, float* detectedConfs)
{
  if(p_engine == 0)
  {
    cout << "Null p_engine" << "\n";
    return 1;
  }

  if(numrows != 360 || numcols != 640)
  {
    cout << "Unexpected number of rows or cols" << "\n";
    return 1;
  }

  ivAdasPDTFEngine* engine = reinterpret_cast<ivAdasPDTFEngine*>(p_engine);
  int numBoxes = engine->m_num_bboxes;
  assert(numBoxes <= MAX_IVADASPD_BBOXES);
  ivAdasPDBbox* boxes = engine->m_bboxes;
  int numDetected = 0;
  const unsigned char* rowStart = 0;
  unsigned char* databuffer = 0;
  for(int i=0; i<numBoxes; i++){
    databuffer = engine->m_bboxData; //buffer to hold 72x36
    for(int row = 0; row < boxes[i].height; row++){
      rowStart = pixelData + numcols*(boxes[i].topLeftY+row) + boxes[i].topLeftX;
      for(int col=0; col<boxes[i].width; col++){
        *databuffer = *rowStart;
        databuffer++;
        rowStart++;
      }
    }

    /*ofstream fout("boxdata.bin", ios::binary);
    if(fout.is_open())
    {
      fout.write(reinterpret_cast<char*>(engine->m_bboxData),
        boxes[i].height*boxes[i].width*sizeof(unsigned char));
      fout.close();
      cout <<  "bbox data written to boxdata.bin" << std::endl;
    }
    else{
      cout <<  "Could not open boxdata.bin for writing" << std::endl;
      return -1;
    }*/


    //engine->m_bboxData is populated, now classify it
    int classOut; float confOut;
    int res = ivAdasPDEngine_classifyimage_bbox(p_engine,engine->m_bboxData,
      boxes[i].height,boxes[i].width, &classOut, &confOut);
    if(res != 0) return res;
    //cout << "classOut=" << classOut << ", confOut=" << confOut << "\n";
    if(classOut == 1 && confOut > 0.5){
        detectedBoxes[numDetected] = boxes[i];
        detectedConfs[numDetected] = confOut;
        numDetected++;
    }
  }
  *numDetectedBoxes = numDetected;

  return 0;
}
#endif

/*int main(int argc, char* argv[]) {

  // Initialize a tensorflow session
  cout << "start initalize session" << "\n";
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }

  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  GraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), "./frozen_graph.pb", &graph_def);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }

  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }
  
  cout << "preparing input data..." << endl;
  // config setting
  int imageDim = 784;
  int nTests = 10000;
  
  // Setup inputs and outputs:
  Tensor x(DT_FLOAT, TensorShape({nTests, imageDim}));

  MNIST mnist = MNIST("./MNIST_data/");
  auto dst = x.flat<float>().data();
  for (int i = 0; i < nTests; i++) {
    auto img = mnist.testData.at(i).pixelData;
    std::copy_n(img.begin(), imageDim, dst);
    dst += imageDim;
  }

  cout << "data is ready" << endl;
  vector<pair<string, Tensor>> inputs = {
    { "input", x}
  };

  // The session will initialize the outputs
  vector<Tensor> outputs;
  // Run the session, evaluating our "softmax" operation from the graph
  status = session->Run(inputs, {"softmax"}, {}, &outputs);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }else{
  	cout << "Success load graph !! " << "\n";
  }

  // start compute the accuracy,
  // arg_max is to record which index is the largest value after 
  // computing softmax, and if arg_max is equal to testData.label,
  // means predict correct.
  int nHits = 0;
  for (vector<Tensor>::iterator it = outputs.begin() ; it != outputs.end(); ++it) {
  	auto items = it->shaped<float, 2>({nTests, 10}); // 10 represent number of class
	for(int i = 0 ; i < nTests ; i++){
	     int arg_max = 0;
      	     float val_max = items(i, 0);
      	     for (int j = 0; j < 10; j++) {
        	if (items(i, j) > val_max) {
          	    arg_max = j;
          	    val_max = items(i, j);
                }
	     }
	     if (arg_max == mnist.testData.at(i).label) {
        	 nHits++;
      	     } 
	}
  }
  float accuracy = (float)nHits/nTests;
  cout << "accuracy is : " << accuracy << ", and Done!!" << "\n";
  return 0;
}*/
