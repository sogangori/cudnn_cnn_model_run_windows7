
#include "cudnn_helper.cuh"
#include <vector>
#include "cpu_func.h"
#include <npps.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include "constant.h"
//#include "variable_reader.h"


class Network{
private:
	bool isDebug = true;
	int vecIdx = 0;
	float one = 1;
	float zero = 0;
	float alpha = 1.0f;
	float beta = 0.0f;
	int variable_length;
	float *variables_h;
	float *variables_d;
	float *outData_d, *buffer_d;
	float *inData_normal;
	uchar* final_d;
	void* workSpace;

	cudnnHandle_t cudnnHandle;
	int* filterShapePtr;
	int filterCount;
	vector<float*> feature_vec;
	vector<float*> conv_vec;
	vector<float*> bias_vec;
	vector<shape> shape_vec;
	vector<cudnnTensorDescriptor_t> td_vec;	
	vector<cudnnFilterDescriptor_t > filterDescriptor_vec;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnPoolingDescriptor_t maxPoolDesc;
	cudnnActivationDescriptor_t actDesc;
	cudnnConvolutionFwdAlgo_t algo;
	cudnnOpTensorDescriptor_t opTensorDesc;
	size_t sizeInBytes = 0;

	uchar* pMeanStdBuffer; // Mean,Std 
	float* pMeanStd;

public:
	char * variablePath = "C:/ultrasound/filter/network/variable.txt";
	const int POOL_COUNT = 7;
	const int CONV_COUNT = 14;
	int filterDepth = 3;
	
	float *inData_d;
	Network()
	{
		printf("Network Constructor \n");
	}

	~Network()
	{
		printf("Network Destroyer \n");
	}

	void LoadWeight(char* path, int *filterShape, int filter_count)
	{		
		filterShapePtr = filterShape;
		filterCount = filter_count;
		variable_length = 0;
		for (int i = 0; i < filterCount/FILTER_DIM; i++)
		{
			int offset = i * FILTER_DIM;
			variable_length += filterShape[offset + 0] * filterShape[offset + 1] * filterShape[offset + 2] * filterShape[offset + 3];
		}
		printf("%s, filterCount:%d, filter_length : %d\n", path,filterCount, variable_length);//8,408,788
				
		FILE* inf = fopen(path, "rb");
		variables_h = new float[variable_length];
		int size = fread(variables_h, sizeof(float), variable_length, inf);
		printf("read %d\n", size);
		fclose(inf);		

		for (int i = 0; i < 2; i++)
		{
			printf("%d,%.3f\n", i, variables_h[i]);
		}
		for (int i = 0; i < 2; i++)
		{
			int offset = variable_length - 1 - i;
			printf("%d,%.3ff\n", offset, variables_h[offset]);
		}

		checkCUDA(cudaMalloc(&variables_d, sizeof(float)* variable_length));
		checkCUDA(cudaMemcpy(variables_d, variables_h, sizeof(float)* variable_length,cudaMemcpyHostToDevice));
	}

	void CreateTensorDescriptor(char *NetLayer,int layerCount, int inputH, int inputW, int inputC){

		int convolution_index = 0;
		printf("%s Feature Map shapes\n", CHAR_INFO);

		printf("input (%d, %d, %d)\n", inputH, inputW, inputC);
		vector<int> width_vec;
		vector<int> height_vec;
		width_vec.push_back(inputW);
		height_vec.push_back(inputH);
		cudnnTensorDescriptor_t inTensorDesc;
		checkCUDA(cudnnCreateTensorDescriptor(&inTensorDesc));
		checkCUDA(cudnnSetTensor4dDescriptor(inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, inputC, inputH, inputW));
		td_vec.push_back(inTensorDesc);

		for (int i = 0; i < layerCount; i++)
		{
			char layer = NetLayer[i];
			if (layer == CONV)
			{				
				inputC = filterShapePtr[convolution_index*FILTER_DIM + 3];
				convolution_index++;
			}
			else if (layer == BN)
			{				
				convolution_index += 2;
			}
			else if (layer == POOL){
				inputW /= 2;
				inputH /= 2;
				width_vec.push_back(inputW);
				height_vec.push_back(inputH);
			}
			else if (layer == UN_POOL){

				width_vec.pop_back();
				height_vec.pop_back();
				inputW = width_vec[width_vec.size() - 1];
				inputH = height_vec[height_vec.size() - 1];
			}

			printf("%d %c (%d, %d, %d)\n", i, layer, inputH, inputW, inputC);
			cudnnTensorDescriptor_t tensorDesc;
			checkCUDA(cudnnCreateTensorDescriptor(&tensorDesc));
			checkCUDA(cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, inputC, inputH, inputW));
			td_vec.push_back(tensorDesc);
		}
		for (int i = 0; i < td_vec.size(); i++)
		{
			PrintDescriptor(i,td_vec[i]);
		}
	}

	void InitFilterDesc()
	{
		for (int i = 0; i < filterCount / FILTER_DIM; i++)
		{
			cudnnFilterDescriptor_t filterDesc;
			checkCUDA(cudnnCreateFilterDescriptor(&filterDesc));
			int offset = i * FILTER_DIM;
			int h = filterShapePtr[offset + 0];
			int w = filterShapePtr[offset + 1];
			int c = filterShapePtr[offset + 2];
			int k = filterShapePtr[offset + 3];
			checkCUDA(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, h, w));
			filterDescriptor_vec.push_back(filterDesc);
		}
		for (int i = 0; i < filterDescriptor_vec.size(); i++)
		{
			int size = GetFilterSize(filterDescriptor_vec[i]);
			printf("[%d] size: %d", i, size);
		}
	}

	float* GetVariablePtr(int variableIndex){
		int size = GetFilterSize(filterDescriptor_vec[variableIndex]);		
		float* weightPtr = variables_d + size;
		return weightPtr;
	}

	void Init(int in_h, int in_w, int in_c)
	{
		printf("Network Init() \n");
		checkCUDA(cudnnCreate(&cudnnHandle));
		checkCUDA(cudnnCreateConvolutionDescriptor(&convDesc));
		checkCUDA(cudnnCreatePoolingDescriptor(&maxPoolDesc));
		checkCUDA(cudnnCreateActivationDescriptor(&actDesc));
		checkCUDA(cudnnCreateOpTensorDescriptor(&opTensorDesc));		

		checkCUDA(cudnnSetConvolution2dDescriptor(convDesc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));
		checkCUDA(cudnnSetPooling2dDescriptor(maxPoolDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2));
		checkCUDA(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));
		checkCUDA(cudnnSetOpTensorDescriptor(opTensorDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));		

		checkCUDA(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, td_vec[0], filterDescriptor_vec[0], convDesc, td_vec[0], CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
		cout << "Fastest algorithm for conv = " << algo << endl;
		checkCUDA(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, td_vec[0], filterDescriptor_vec[0], convDesc, td_vec[0], algo, &sizeInBytes));
		cout << "sizeInBytes " << sizeInBytes << endl;
		if (sizeInBytes > 0) checkCUDA(cudaMalloc(&workSpace, sizeInBytes));

		checkCUDA(cudaMalloc((void**)&inData_normal, GetTensorSize(td_vec[0])*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&inData_d, GetTensorSize(td_vec[0])*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&outData_d, GetTensorSize(td_vec[0])*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&buffer_d, GetTensorSize(td_vec[0])*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&final_d, GetTensorSize(td_vec[0])*sizeof(uchar)));

		int nBufferSize;
		nppsMeanStdDevGetBufferSize_32f(in_w * in_h, &nBufferSize);
		cudaMalloc(&pMeanStdBuffer, nBufferSize);
		cudaMalloc(&pMeanStd, sizeof(float)* 3);
		printf("Network Init() OK \n");
	}

	void inference()
	{
		if (isDebug) printf("Network inference() \n");
		vecIdx = 0;
		int tensorDescInx = 0;
		int variableInx = 0;
		
		Nornalize(inData_d, inData_normal, td_vec[0]);
		checkCUDA(cudnnConvolutionForward(cudnnHandle, &alpha, td_vec[0], inData_normal,
			filterDescriptor_vec[vecIdx], GetVariablePtr(0), convDesc, algo, workSpace, sizeInBytes, &beta, td_vec[1], outData_d));
		
	}

	void Nornalize(float* src, float* dst, cudnnTensorDescriptor_t descriptor)
	{
		cudnnDataType_t                    dataType; // image data type
		int                                n;        // number of inputs (batch size)
		int                                c;        // number of input feature maps
		int                                h;        // height of input section
		int                                w;        // width of input section
		int                                nStride;
		int                                cStride;
		int                                hStride;
		int                                wStride;
		checkCUDA(cudnnGetTensor4dDescriptor(descriptor, &dataType, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride));
		for (int i = 0; i < c; i++)
		{
			float* target = src + w*h*i;
			checkNPP(nppsMeanStdDev_32f(target, w * h, &pMeanStd[0], &pMeanStd[1], pMeanStdBuffer));
			math_std_normal << <h, w >> >(&dst[w*h*i], target, pMeanStd);
		}
	}
};