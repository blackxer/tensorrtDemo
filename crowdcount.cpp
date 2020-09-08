#ifndef CROWDCOUNT
#define CROWDCOUNT

#endif // CROWDCOUNT

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "argsParser.h"
#include "logger.h"
#include "common.h"

#include <opencv2/opencv.hpp>

using namespace nvinfer1;

class CrowdCount{
public:
    CrowdCount(int input_c, int input_h, int input_w, int output_c, int output_h, int output_w);
    bool onnx2trt(std::string onnxModelName, std::string trtModelName, int maxBatchSize);
    void initEngine(std::string trtModelName);
    void doInference(float* input, float* output, int batchSize);
    void preprocess(const std::string& fileName, float* preInput);

    ~CrowdCount(){
        // destroy the engine
        if(!context_){
            context_->destroy();
            engine_->destroy();
            runtime_->destroy();
        }
    }

private:
    int input_c_;
    int input_h_;
    int input_w_;

    int output_c_;
    int output_h_;
    int output_w_;

    int batchSize_;

    IRuntime* runtime_;
    ICudaEngine* engine_;
    IExecutionContext* context_;
};

CrowdCount::CrowdCount(int input_c, int input_h, int input_w, int output_c, int output_h, int output_w){
    input_c_ = input_c;
    input_h_ = input_h;
    input_w_ = input_w;

    output_c_ = output_c;
    output_h_ = output_h;
    output_w_ = output_w;
}

bool CrowdCount::onnx2trt(std::string onnxModelName, std::string trtModelName, int maxBatchSize){
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());

    if ( !parser->parseFromFile(onnxModelName.c_str(), static_cast<int>(gLogger.getReportableSeverity()) ) )
    {
       gLogError << "Failure while parsing ONNX file" << std::endl;
       return false;
    }

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    builder->setFp16Mode(false);
    builder->setInt8Mode(false);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we can destroy the parser
    parser->destroy();

    // serialize the engine, then close everything down
    IHostMemory *trtModelStream{nullptr};
    trtModelStream = engine->serialize();

    // 设置保存文件的名称为cached_model.bin
    std::ofstream serialize_output_stream;

    // 将序列化的模型结果拷贝至serialize_str字符串
    std::string serialize_str;
    serialize_str.resize( trtModelStream->size() );
    memcpy((void*)serialize_str.data(), trtModelStream->data(), trtModelStream->size());

    // 将serialize_str字符串的内容输出至cached_model.bin文件
    serialize_output_stream.open(trtModelName);
    serialize_output_stream << serialize_str;
    serialize_output_stream.close();

    engine->destroy();
    network->destroy();
    builder->destroy();

    return true;
}

void CrowdCount::initEngine(std::string trtModelName){
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(trtModelName, std::ios::binary);
    if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
    }

    // deserialize the engine
    runtime_ = createInferRuntime(gLogger);
    assert(runtime_ != nullptr);

    engine_ = runtime_->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine_ != nullptr);
    delete[] trtModelStream;
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);
}

void CrowdCount::doInference(float* input, float* output, int batchSize){
    const ICudaEngine& engine = context_->getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];
    std::cout << engine.getBindingDimensions(0) << std::endl;

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[0], batchSize * input_c_ * input_h_ * input_w_ * sizeof(float)));
    CHECK(cudaMalloc(&buffers[1], batchSize * output_c_ * output_h_ * output_w_ * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * input_c_ * input_h_ * input_w_* sizeof(float), cudaMemcpyHostToDevice, stream));
    context_->enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * output_c_ * output_h_ * output_w_* sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
}

void CrowdCount::preprocess(const std::string& fileName, float* preInput){
    cv::Mat img = cv::imread(fileName);
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
//    cv::namedWindow("test");
//    cv::imshow("test", img);
//    cv::waitKey(0);
    cv::resize(img,img,cv::Size(input_w_, input_h_));

    assert(img.channels() == input_c_);
    assert(img.rows == input_h_);
    assert(img.cols == input_w_);

    int predHeight = img.rows;
    int predWidth = img.cols;
    int size = predHeight * predWidth;
    uint8_t buffer[input_c_*input_h_*input_w_];
    // 注意imread读入的图像格式是unsigned char，如果你的网络输入要求是float的话，下面的操作就不对了。
    for (auto i=0; i<predHeight; i++) {
        //printf("+\n");
        for (auto j=0; j<predWidth; j++) {
            buffer[i * predWidth + j + 0*size] = (uint8_t)img.data[(i*predWidth + j) * 3 + 0];
            buffer[i * predWidth + j + 1*size] = (uint8_t)img.data[(i*predWidth + j) * 3 + 1];
            buffer[i * predWidth + j + 2*size] = (uint8_t)img.data[(i*predWidth + j) * 3 + 2];
        }
    }
    // mean, std
    for(int k=0; k< input_c_*input_h_*input_w_; k++){
        preInput[k] = ((float)buffer[k]/255.0 - 0.5)/0.5;
    }
}

int main(int argc, char** argv)
{
    int INPUT_C = 3;
    int INPUT_H = 512;
    int INPUT_W = 512;

    int OUTPUT_C = 1;
    int OUTPUT_H = 64;
    int OUTPUT_W = 64;

    CrowdCount crowdCount(INPUT_C, INPUT_H, INPUT_W, OUTPUT_C, OUTPUT_H, OUTPUT_W);
    const char* onnxModelName = "/media/zw/DL/ly/workspace/project10/Bayesian-Crowd-Counting-master/weights/model.onnx";
    std::string trtModelName = "../cached_model.bin";
//    crowdCount.onnx2trt(onnxModelName, trtModelName, 1);
    crowdCount.initEngine(trtModelName);

    float data[INPUT_C*INPUT_H*INPUT_W];
    string  filename = "/media/zw/DL/ly/workspace/project10/Bayesian-Crowd-Counting-master/datasets/ShanghaiTech_Crowd_Counting_Dataset/part_A_final_processed/test/IMG_11.jpg";
    crowdCount.preprocess(filename, data);

    // run inference
    float scores[OUTPUT_C*OUTPUT_H*OUTPUT_W];
    crowdCount.doInference(data, scores, 1);
    float people_count = 0;
    for(int i=0;i<OUTPUT_C*OUTPUT_H*OUTPUT_W;i++){
        people_count+=scores[i];
    }
    std::cout << "finished" << people_count << std::endl;


    return 0;
}







