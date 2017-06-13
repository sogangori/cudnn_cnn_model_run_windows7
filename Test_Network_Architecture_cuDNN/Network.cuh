
#include "cudnn_helper.cuh"
#include <vector>
#include "cpu_func.h"
#include <npps.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include "constant.h"
//#include "variable_reader.h"


class Network{
private:
	bool isDebug = false;
	int vecIdx = 0;
	float one = 1;
	float zero = 0;
	float alpha = 1.0f;
	float beta = 0.0f;
	int variable_length;
	float *variables_h;
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
	vector<cudnnTensorDescriptor_t> tensorDescriptor_vec;
	vector<cudnnTensorDescriptor_t> biasDescriptor_vec;
	vector<cudnnFilterDescriptor_t > filterDescriptor_vec;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnPoolingDescriptor_t maxPoolDesc, avgPoolDesc;
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
		float* src = new float[variable_length];
		
		FILE* inf = fopen(path, "rb");

		int size = fread(src, sizeof(float), variable_length, inf);
		printf("read %d\n", size);
		fclose(inf);		

		for (int i = 0; i < 2; i++)
		{
			printf("%d,%.3f\n", i, src[i]);
		}
		for (int i = 0; i < 2; i++)
		{
			int offset = variable_length - 1 - i;
			printf("%d,%.3ff\n", offset, src[offset]);
		}
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
		tensorDescriptor_vec.push_back(inTensorDesc);

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
			tensorDescriptor_vec.push_back(tensorDesc);
		}
		for (int i = 0; i < tensorDescriptor_vec.size(); i++)
		{
			PrintDescriptor(i,tensorDescriptor_vec[i]);
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
	void Init(int in_h, int in_w, int in_c)
	{
		printf("Network Init() \n");
		checkCUDA(cudnnCreate(&cudnnHandle));
		
		/*
		checkCUDA(cudnnCreateConvolutionDescriptor(&convDesc));
		checkCUDA(cudnnCreatePoolingDescriptor(&maxPoolDesc));
		checkCUDA(cudnnCreatePoolingDescriptor(&avgPoolDesc));
		checkCUDA(cudnnCreateActivationDescriptor(&actDesc));
		checkCUDA(cudnnCreateOpTensorDescriptor(&opTensorDesc));

		shape lastShape = shape_vec[shape_vec.size() - 1];
		printf("last shape\n");
		lastShape.Print();

		checkCUDA(cudnnSetConvolution2dDescriptor(convDesc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));
		checkCUDA(cudnnSetPooling2dDescriptor(maxPoolDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2));
		checkCUDA(cudnnSetPooling2dDescriptor(avgPoolDesc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2));
		checkCUDA(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));
		checkCUDA(cudnnSetOpTensorDescriptor(opTensorDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));

		cudnnTensorDescriptor_t inTensorDesc;
		checkCUDA(cudnnCreateTensorDescriptor(&inTensorDesc));
		checkCUDA(cudnnSetTensor4dDescriptor(inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape_in.n, shape_in.c, shape_in.h, shape_in.w));
		tensorDescriptor_vec.push_back(inTensorDesc);

		for (int i = 0; i < CONV_COUNT; i++)
		{
			cudnnTensorDescriptor_t tensorDesc;
			cudnnFilterDescriptor_t filterDesc;
			cudnnTensorDescriptor_t biasTensorDesc;
			float* conv, *bias;

			checkCUDA(cudnnCreateTensorDescriptor(&tensorDesc));
			checkCUDA(cudnnCreateTensorDescriptor(&biasTensorDesc));
			checkCUDA(cudnnCreateFilterDescriptor(&filterDesc));

			checkCUDA(cudaMalloc((void**)&conv, shape_vec[i].len() * sizeof(float)));
			checkCUDA(cudaMalloc((void**)&bias, shape_vec[i].n * sizeof(float)));
			checkCUDA(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, shape_vec[i].n, shape_vec[i].c, shape_vec[i].h, shape_vec[i].w));
			checkCUDA(cudnnSetTensor4dDescriptor(biasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, shape_vec[i].n, 1, 1));

			tensorDescriptor_vec.push_back(tensorDesc);
			filterDescriptor_vec.push_back(filterDesc);
			biasDescriptor_vec.push_back(biasTensorDesc);
			conv_vec.push_back(conv);
			bias_vec.push_back(bias);
		}

		for (int i = 0; i < POOL_COUNT; i++)
		{
			int out_n, out_c, out_h, out_w;
			checkCUDA(cudnnGetPooling2dForwardOutputDim(avgPoolDesc, tensorDescriptor_vec[i], &out_n, &out_c, &out_h, &out_w));
			printf("%d pool out shape (%d x %d x %d x %d)\n", i + 1, out_n, out_c, out_h, out_w);
			checkCUDA(cudnnSetTensor4dDescriptor(tensorDescriptor_vec[i + 1], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));

			float* feature;
			checkCUDA(cudaMalloc((void**)&feature, GetTensorSize(tensorDescriptor_vec[i])* sizeof(float)));
			feature_vec.push_back(feature);
		}

		checkCUDA(cudnnSetTensor4dDescriptor(GetTensorDescriptor(-1), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape_in.n, 2, shape_in.h, shape_in.w));
		checkCUDA(cudnnSetTensor4dDescriptor(GetTensorDescriptor(-2), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape_in.n, 2, shape_in.h / 2, shape_in.w / 2));
		checkCUDA(cudnnSetTensor4dDescriptor(GetTensorDescriptor(-3), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape_in.n, 3, shape_in.h / 4, shape_in.w / 4));
		checkCUDA(cudnnSetTensor4dDescriptor(GetTensorDescriptor(-4), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape_in.n, 3, shape_in.h / 8, shape_in.w / 8));
		checkCUDA(cudnnSetTensor4dDescriptor(GetTensorDescriptor(-5), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape_in.n, 3, shape_in.h / 16, shape_in.w / 16));
		checkCUDA(cudnnSetTensor4dDescriptor(GetTensorDescriptor(-6), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape_in.n, 3, shape_in.h / 32, shape_in.w / 32));
		checkCUDA(cudnnSetTensor4dDescriptor(GetTensorDescriptor(-7), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape_in.n, 3, shape_in.h / 64, shape_in.w / 64));

		Print(tensorDescriptor_vec);

		checkCUDA(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, tensorDescriptor_vec[0], filterDescriptor_vec[0], convDesc, tensorDescriptor_vec[0], CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

		cout << "Fastest algorithm for conv = " << algo << endl;

		checkCUDA(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, tensorDescriptor_vec[0], filterDescriptor_vec[0], convDesc, tensorDescriptor_vec[0], algo, &sizeInBytes));

		cout << "sizeInBytes " << sizeInBytes << endl;
		if (sizeInBytes > 0) checkCUDA(cudaMalloc(&workSpace, sizeInBytes));

		checkCUDA(cudaMalloc((void**)&inData_normal, GetTensorSize(tensorDescriptor_vec[0])*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&inData_d, GetTensorSize(tensorDescriptor_vec[0])*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&outData_d, GetTensorSize(tensorDescriptor_vec[0])*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&buffer_d, GetTensorSize(tensorDescriptor_vec[0])*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&final_d, GetTensorSize(tensorDescriptor_vec[0])*sizeof(uchar)));

		int nBufferSize;
		nppsMeanStdDevGetBufferSize_32f(shape_in.w * shape_in.h, &nBufferSize);
		cudaMalloc(&pMeanStdBuffer, nBufferSize);
		cudaMalloc(&pMeanStd, sizeof(float)* 3);*/
		printf("Network Init() OK \n");
	}

	cudnnTensorDescriptor_t GetTensorDescriptor(int index)
	{
		//not Tested
		if (index < 0 && abs(index) >= tensorDescriptor_vec.size()) index = index%tensorDescriptor_vec.size();
		if (index < 0) index = tensorDescriptor_vec.size() + index;

		return tensorDescriptor_vec[index];
	}
};