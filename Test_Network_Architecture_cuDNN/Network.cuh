
#include "cudnn_helper.cuh"
#include <vector>
#include "cpu_func.h"
#include <npps.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include "constant.h"


class Network{
private:
	bool isDebug = true;	
	float one = 1;
	float zero = 0;
	float alpha = 1.0f;
	float beta = 0.0f;
	int variable_length;
	float *testBuffer_h;
	float *variables_h;
	float *variables_convert_h;
	float *variables_d;
	float *outData_d, *buffer1_d, *buffer2_d;	
	float *feature0, *feature1, *feature2, *feature3;
	void* workSpace;
	int tdInx = 0;
	int variableInx = 0;

	cudnnHandle_t cudnnHandle;
	int* filterShapePtr;
	int filterCount;
	vector<float*> feature_vec;
	vector<float*> conv_vec;
	vector<float*> bias_vec;
	vector<shape> shape_vec;
	vector<cudnnTensorDescriptor_t> td_vec;
	vector<cudnnTensorDescriptor_t> filter_td_vec;
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
		variables_convert_h = new float[variable_length];
		checkCUDA(cudaMalloc(&variables_d, sizeof(float)* variable_length));
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

		//필터 돌리자 HWCN -> NCHW
		int v_offset = 0;
		for (int i = 0; i < filterCount / FILTER_DIM; i++)
		{
			int offset = i * FILTER_DIM;
			int height = filterShape[offset + 0];
			int width = filterShape[offset + 1];
			int channel = filterShape[offset + 2];
			int kcount = filterShape[offset + 3];
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					for (int c = 0; c < channel; c++)
					{
						for (int k = 0; k < kcount; k++)
						{
							int index_in = v_offset 
								+ (h * width * channel * kcount) 
								+ (w * channel * kcount) 
								+ (c * kcount)
								+ k;
							int index_out = v_offset
								+ k * channel * height * width
								+ c * height * width 
								+ h * width 
								+ w;
							variables_convert_h[index_out] = variables_h[index_in];
						}
					}
				}
			}
			v_offset += height*width*channel*kcount;
		}
		
		checkCUDA(cudaMemcpy(variables_d, variables_convert_h, sizeof(float)* variable_length, cudaMemcpyHostToDevice));
	}

	void CreateTensorDescriptor(char *NetLayer,int layerCount, int inputH, int inputW, int inputC){

		int filter_index = 0;
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
				inputC = filterShapePtr[filter_index*FILTER_DIM + 3];
				filter_index++;
			}
			else if (layer == BN) filter_index += 2;			
			else if (layer == BIAS) filter_index ++;			
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
			cudnnTensorDescriptor_t filterTensorDesc;
			checkCUDA(cudnnCreateFilterDescriptor(&filterDesc));
			checkCUDA(cudnnCreateTensorDescriptor(&filterTensorDesc));

			int offset = i * FILTER_DIM;
			int h = filterShapePtr[offset + 0];
			int w = filterShapePtr[offset + 1];
			int c = filterShapePtr[offset + 2];
			int k = filterShapePtr[offset + 3];
			
			checkCUDA(cudnnSetTensor4dDescriptor(filterTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, k, c, h, w));
			checkCUDA(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, h, w));
			
			filterDescriptor_vec.push_back(filterDesc);
			filter_td_vec.push_back(filterTensorDesc);
		}
		for (int i = 0; i < filterDescriptor_vec.size(); i++)
		{
			int size = GetFilterSize(filterDescriptor_vec[i]);			
		}
	}

	float* GetVariablePtr(int variableIndex)
	{
		int offset = 0;
		for (int i = 0; i < variableIndex; i++)
			offset += GetFilterSize(filterDescriptor_vec[i]);
		
		return variables_d + offset;
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

		testBuffer_h = new float[GetTensorSize(td_vec[0])];
		
		checkCUDA(cudaMalloc((void**)&inData_d, GetTensorSize(td_vec[0])*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&outData_d, GetTensorSize(td_vec[td_vec.size()-1])*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&buffer1_d, 2 * GetTensorSize(td_vec[0])*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&buffer2_d, 2 * GetTensorSize(td_vec[0])*sizeof(float)));		
		checkCUDA(cudaMalloc((void**)&feature0, GetTensorSize(td_vec[0])*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&feature1, GetTensorSize(td_vec[0])*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&feature2, GetTensorSize(td_vec[0])*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&feature3, GetTensorSize(td_vec[0])*sizeof(float)));
		

		checkCUDA(cudaMemset(inData_d, 0, GetTensorSize(td_vec[0])*sizeof(float)));		
		checkCUDA(cudaMemset(buffer1_d, 0, 2 * GetTensorSize(td_vec[0])*sizeof(float)));
		checkCUDA(cudaMemset(buffer2_d, 0, 2 * GetTensorSize(td_vec[0])*sizeof(float)));
		checkCUDA(cudaMemset(outData_d, 0, GetTensorSize(td_vec[td_vec.size() - 1])*sizeof(float)));

		int nBufferSize;
		nppsMeanStdDevGetBufferSize_32f(in_w * in_h, &nBufferSize);
		cudaMalloc(&pMeanStdBuffer, nBufferSize);
		cudaMalloc(&pMeanStd, sizeof(float)* nBufferSize*4);//TODO. 대충 해놓음
		printf("Network Init() OK \n");
	}

	void BatchNornalize(float* src, float* dst, cudnnTensorDescriptor_t descriptor, cudnnTensorDescriptor_t filterD,
		float* bnScale, float* bnBias)
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
			checkNPP(nppsMeanStdDev_32f(target, w * h, &pMeanStd[i], &pMeanStd[c + i], pMeanStdBuffer));
		}

		Std2Var << <1, c >> >(pMeanStd + c);
		cudnnBatchNormalizationForwardInference(cudnnHandle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, descriptor,
			src, descriptor, dst, filterD, bnScale, bnBias, pMeanStd, pMeanStd + c, 0.001);
	}

	void Log(char* layer)
	{
		printf("[Log] %s\t(%d, %d)\n", layer, tdInx, variableInx);
		PrintDescriptor(tdInx, td_vec[tdInx]);
	}

	void Bias(float* dst)
	{
		Log("Bias In");

		cudnnTensorDescriptor_t biasDesc = filter_td_vec[variableInx];		
		cudnnTensorDescriptor_t tensorDesc = td_vec[tdInx];
		float * bias = GetVariablePtr(variableInx);
		checkCUDA(cudnnAddTensor(cudnnHandle, &alpha, biasDesc, bias, &alpha, tensorDesc, dst));
		tdInx++;
		variableInx++;
		Log("Bias Out");
	}

	void Conv(float* src, float* dst)
	{
		Log("convIn");

		checkCUDA(cudnnConvolutionForward(cudnnHandle, &alpha, td_vec[tdInx], src,
			filterDescriptor_vec[variableInx], GetVariablePtr(variableInx), convDesc, algo, workSpace, sizeInBytes, &beta, td_vec[tdInx + 1], dst));
		tdInx++;
		variableInx ++;
		Log("conv Out");		
	}
	
	void BatchNormalize(float* src, float* dst)
	{
		Log("BN In");
		float* bnScale = GetVariablePtr(variableInx);
		float* bnBias = GetVariablePtr(variableInx + 1);
		BatchNornalize(src, dst, td_vec[tdInx], filter_td_vec[variableInx], bnScale, bnBias);
		tdInx++;
		variableInx += 2;
		Log("BN Out");
	}

	void ConvBN(float* src, float* dst)
	{
		Conv(src, inData_d);
		BatchNormalize(inData_d, dst);
	}

	void Activate(float* src, float* dst)
	{		
		checkCUDA(cudnnActivationForward(cudnnHandle, actDesc, &alpha, td_vec[tdInx], src, &beta, td_vec[tdInx + 1], dst));
		tdInx++;
		Log("Acti");
	}

	void Pool()
	{		
		checkCUDA(cudnnPoolingForward(cudnnHandle, maxPoolDesc, &alpha, td_vec[tdInx], buffer2_d, &zero, td_vec[tdInx + 1], buffer1_d));
		tdInx++;
		Log("Pool");
	}

	void UnPool(float* src, float* dst)
	{		
		Resize(src, td_vec[tdInx], dst, td_vec[tdInx+1]);
		tdInx++;
		Log("UnPool");
	}

	void Add(float* src, float*src2, float* dst)
	{
		checkCUDA(cudnnOpTensor(cudnnHandle, opTensorDesc, &alpha, td_vec[tdInx + 1], src, &alpha, td_vec[tdInx + 1], src2, &zero, td_vec[tdInx + 1], dst));
		tdInx++;
		Log("Add");
	}

	void inference()
	{
		if (isDebug) printf("Network inference() \n");
		tdInx = variableInx = 0;		
		//mean/std : -26.7, 612
				
		NormalizeInput(inData_d, buffer2_d, -26.7f, 612);
		//0. P CBN R
		Pool();
		Conv(buffer1_d, buffer2_d);
		Bias(buffer2_d);
		Activate(buffer2_d, buffer2_d);
		
		//1. P CBN R
		Pool();
		Conv(buffer1_d, buffer2_d);
		Bias(buffer2_d);
		Activate(buffer2_d, buffer2_d);
				
		//2. P CBN R
		Pool();
		Conv(buffer1_d, buffer2_d);
		Bias(buffer2_d);
		Activate(buffer2_d, buffer2_d);					

		//checkCUDA(cudaMemcpy(outData_d, buffer2_d, GetTensorSize(td_vec[td_vec.size() - 1])*sizeof(float), cudaMemcpyDeviceToDevice));
		//3 U
		Resize(buffer2_d, td_vec[tdInx], outData_d, td_vec[0]);
		
		if (td_vec.size() != tdInx + 1)
		{
			printf("[Warn] Operation?  %d != %d\n", td_vec.size(), tdInx + 1);
		}
		if (filter_td_vec.size() != variableInx){
			printf("[Warn] filter?  %d != %d\n", filter_td_vec.size(), variableInx);
		}
		
		printf("Inference finished\n");
	}

	void GetInference(void* dst)
	{
		cudnnDataType_t                    dataType;
		int                                n;
		int                                c;
		int                                h;
		int                                w;
		int                                nStride;
		int                                cStride;
		int                                hStride;
		int                                wStride;
		
		checkCUDA(cudnnGetTensor4dDescriptor(td_vec[td_vec.size() - 1], &dataType, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride));
		if (isDebug) printf("GetInference() %d x %d x %d x %d \n", n, c, h, w);
		ArgMax << <h, w >> >((uchar*)dst, outData_d);
		//softMax2Uchar << <h, w >> >((uchar*)dst, outData_d, 1);
	}

	void TestCopyInput()
	{
		int size = GetTensorSize(td_vec[td_vec.size() - 1]);
		checkCUDA(cudaMemcpy(testBuffer_h, buffer1_d, size*sizeof(float), cudaMemcpyDeviceToHost));
		printf("TestCopyInput\n");
		for (int i = 0; i < size/10; i++)
		{
			if (testBuffer_h[i] > 0.1)
				printf("%d %f\n", i,testBuffer_h[i]);
		}
	}

	void CopyInput(void* src)
	{
		cudnnDataType_t                    dataType;
		int                                n;
		int                                c;
		int                                h;
		int                                w;
		int                                nStride;
		int                                cStride;
		int                                hStride;
		int                                wStride;

		checkCUDA(cudnnGetTensor4dDescriptor(td_vec[0], &dataType, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride));				
		checkCUDA(cudaMemcpy(inData_d, src, c*w*h*sizeof(float), cudaMemcpyDeviceToDevice));
	}

	void NornalizeMinMax(float* src, float* dst, cudnnTensorDescriptor_t descriptor)
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

	void NormalizeInput(float* src, float* dst,float mean, float std)
	{
		cudnnTensorDescriptor_t descriptor = td_vec[0];
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
		
		math_std_normal << <dim3(c,h), w >> >(dst,src, mean, std);
	}
};