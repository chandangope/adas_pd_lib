
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
}ivAdasPDTFEngine;

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


void* ivAdasPDEngine_Init(std::string modelFilePath)
{
  cout << "Creating session and adding graph" << "\n";
  ivAdasPDTFEngine* p_engine = new ivAdasPDTFEngine;
  
  //Session* m_session;
  //string graph_file_name = "./frozen_graph.pb";
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
  Session* m_session = engine->m_session;
  m_session->Close();
  cout << "Closing the session" << endl;
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
  
  cout << "preparing input data..." << endl;
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

  cout << "data is ready" << endl;
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
    cout << "Successfully ran session " << "\n";
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
