
#include "cudnn_helper.cuh"
#include <vector>
#include "cpu_func.h"
#include <npps.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include "constant.h"


class Network{
private:
	bool isDebug = true;
	float alpha = 1.0f;
	float zero = 0;
	double epsilon = 0.001;
	int variable_length;
	float *testBuffer_h;
	float *variables_h;
	float *variables_convert_h;
	float *variables_d;
	float *outData_d, *buffer1_d, *buffer2_d, *buffer_conv_d;
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
		checkCUDA(cudnnCreate(&cudnnHandle));
		checkCUDA(cudnnCreateConvolutionDescriptor(&convDesc));
		checkCUDA(cudnnCreatePoolingDescriptor(&maxPoolDesc));
		checkCUDA(cudnnCreateActivationDescriptor(&actDesc));
		checkCUDA(cudnnCreateOpTensorDescriptor(&opTensorDesc));
		checkCUDA(cudnnSetConvolution2dDescriptor(convDesc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));
		checkCUDA(cudnnSetPooling2dDescriptor(maxPoolDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2));
		checkCUDA(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));
		checkCUDA(cudnnSetOpTensorDescriptor(opTensorDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
	}

	~Network()
	{
		printf("Network Destroyer \n");
	}

	void CopyFilterRotate()
	{	
		RotateFilterHWCN2NCHW(variables_h, variables_convert_h, filterShapePtr, filter_td_vec.size());
		checkCUDA(cudaMemcpy(variables_d, variables_convert_h, sizeof(float)* variable_length, cudaMemcpyHostToDevice));
	}

	void LoadWeight(char* path, int *filterShape, int filter_count)
	{
		filterShapePtr = filterShape;
		filterCount = filter_count;
		variable_length = 0;
		for (int i = 0; i < filterCount / FILTER_DIM; i++)
		{
			int offset = i * FILTER_DIM;
			int h = filterShape[offset + 0];
			int w = filterShape[offset + 1];
			int c = filterShape[offset + 2];
			int k = filterShape[offset + 3];
			variable_length += h*w*c*k;

			cudnnFilterDescriptor_t filterDesc;
			cudnnTensorDescriptor_t filterTensorDesc;
			checkCUDA(cudnnCreateFilterDescriptor(&filterDesc));
			checkCUDA(cudnnCreateTensorDescriptor(&filterTensorDesc));

			printf("InitFilterDesc(%d) %d x %d x %d x %d\n", i, k, c, h, w);
			checkCUDA(cudnnSetTensor4dDescriptor(filterTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, k, c, h, w));
			checkCUDA(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, h, w));

			filterDescriptor_vec.push_back(filterDesc);
			filter_td_vec.push_back(filterTensorDesc);
		}
		printf("%s, filterCount:%d, filter_length : %d\n", path, filterCount, variable_length);//8,408,788

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
		
		CopyFilterRotate();		
	}	

	void SetMaxConvBufferSize(char *NetLayer, int layerCount)
	{
		printf("%s SetMaxConvBuffSize\n", CHAR_INFO);
		int filter_index = 0;
		size_t maxSizeInBytes = 0;
		for (int i = 0; i < layerCount; i++)
		{
			char layer = NetLayer[i];
			if (layer == CONV)
			{	
				cudnnTensorDescriptor_t xDesc = td_vec[i];
				cudnnTensorDescriptor_t yDesc = td_vec[i + 1];
				cudnnFilterDescriptor_t wDesc = filterDescriptor_vec[filter_index];
				checkCUDA(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, xDesc, 
					wDesc, convDesc, yDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
				cout << "Fastest algorithm for conv = " << algo << endl;
				checkCUDA(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, xDesc, wDesc, convDesc, yDesc, algo, &sizeInBytes));				
				cout << "sizeInBytes " <<filter_index<<" "<<  sizeInBytes << endl;
				if (sizeInBytes > maxSizeInBytes) maxSizeInBytes = sizeInBytes;
				filter_index++;
			}
			else if (layer == BN) filter_index += 2;
			else if (layer == BIAS) filter_index++;		
		}

		sizeInBytes = maxSizeInBytes;
		if (sizeInBytes > 0) checkCUDA(cudaMalloc(&workSpace, sizeInBytes));
	}

	void CreateTensorDescriptor(char *NetLayer, int layerCount, int inputH, int inputW, int inputC)
	{		
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

		int filter_index = 0;
		for (int i = 0; i < layerCount; i++)
		{
			char layer = NetLayer[i];
			if (layer == CONV)
			{
				inputC = filterShapePtr[filter_index*FILTER_DIM + 3];
				filter_index++;
			}
			else if (layer == BN) filter_index += 2;
			else if (layer == BIAS) filter_index++;
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
		SetMaxConvBufferSize(NetLayer, layerCount);
		for (int i = 0; i < td_vec.size(); i++)
		{
			PrintDescriptor(i, td_vec[i]);
		}
	}

	float* GetVariablePtr(int variableIndex)
	{
		int offset = 0;
		for (int i = 0; i < variableIndex; i++)
			offset += GetFilterSize(filterDescriptor_vec[i]);

		return variables_d + offset;
	}

	int GetMaxTensorSize()
	{
		int maxSize = 0;
		for (int i = 0; i < td_vec.size(); i++)
		{
			int size = GetTensorSize(td_vec[i]);
			if (size>maxSize) maxSize = size;
		}
		return maxSize;
	}

	void Init(int in_h, int in_w, int in_c)
	{
		printf("Network Init() \n");


		int maxSize = GetMaxTensorSize();
		testBuffer_h = new float[maxSize];
		checkCUDA(cudaMalloc((void**)&inData_d, GetTensorSize(td_vec[0])*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&outData_d, GetTensorSize(td_vec[0])*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&buffer1_d, maxSize*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&buffer2_d, maxSize*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&buffer_conv_d, maxSize*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&feature0, maxSize*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&feature1, maxSize*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&feature2, maxSize*sizeof(float)));
		checkCUDA(cudaMalloc((void**)&feature3, maxSize*sizeof(float)));

		checkCUDA(cudaMemset(inData_d, 0, GetTensorSize(td_vec[0])*sizeof(float)));
		checkCUDA(cudaMemset(outData_d, 0, GetTensorSize(td_vec[td_vec.size() - 1])*sizeof(float)));
		checkCUDA(cudaMemset(buffer1_d, 0, maxSize*sizeof(float)));
		checkCUDA(cudaMemset(buffer2_d, 0, maxSize*sizeof(float)));

		int nBufferSize;
		nppsMeanStdDevGetBufferSize_32f(in_w * in_h, &nBufferSize);		
		cudaMalloc(&pMeanStdBuffer, nBufferSize);
		cudaMalloc(&pMeanStd, sizeof(float)* nBufferSize);
		printf("Network Init() OK \n");
	}

	void Log(char* layer)
	{
		if (isDebug)
		{
			printf("[Log] %s\t(%d, %d)\n", layer, tdInx, variableInx);
			PrintDescriptor(tdInx, td_vec[tdInx]);
		}
	}

	void Bias(float* dst)
	{
		Log("Bias In");

		cudnnTensorDescriptor_t biasDesc = filter_td_vec[variableInx];
		cudnnTensorDescriptor_t tensorDesc = td_vec[tdInx];
		float * bias = GetVariablePtr(variableInx);
		checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, biasDesc, bias, &alpha, tensorDesc, dst));
		tdInx++;
		variableInx++;
		Log("Bias Out");
	}

	void Conv(float* src, float* dst)
	{
		Log("convIn");

		int out_n, out_c, out_h, out_w;
		cudnnTensorDescriptor_t inputTensorDesc = td_vec[tdInx];
		cudnnTensorDescriptor_t outputTensorDesc = td_vec[tdInx + 1];
		checkCUDA(cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensorDesc, filterDescriptor_vec[variableInx], &out_n, &out_c, &out_h, &out_w));
		printf("OutDim Conv %d x %d x %d x %d \n", out_n, out_c, out_h, out_w);

		checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, inputTensorDesc, src,
			filterDescriptor_vec[variableInx], GetVariablePtr(variableInx), convDesc, algo, workSpace, sizeInBytes, &zero, outputTensorDesc, dst));
		tdInx++;
		variableInx++;
		Log("conv Out");
	}

	void BatchNormalize(float* src, float* dst)
	{
		Log("BN In");
		float* bnBias = GetVariablePtr(variableInx);
		float* bnScale = GetVariablePtr(variableInx + 1);
		cudnnTensorDescriptor_t descriptor = td_vec[tdInx];
		cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc = filter_td_vec[variableInx];
		cudnnDataType_t                    dataType; 
		int                                n;        
		int                                c;        
		int                                h;        
		int                                w;        
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

		float * estimatedMean = pMeanStd;
		float * estimatedVariance = pMeanStd + c;
		Std2Var << <1, c >> >(estimatedVariance);

		checkCUDNN(cudnnBatchNormalizationForwardInference(cudnnHandle, CUDNN_BATCHNORM_SPATIAL, &alpha, &zero, descriptor, src, descriptor, dst, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon));
		tdInx++;
		variableInx += 2;
		Log("BN Out");
	}

	void ConvBN(float* src)
	{
		Conv(src, buffer_conv_d);
		BatchNormalize(buffer_conv_d, src);
	}

	void ConvBN_Activate(float* src)
	{
		Conv(src, buffer_conv_d);
		BatchNormalize(buffer_conv_d, src);
		Activate(src);
	}

	void Activate(float* src)
	{
		checkCUDA(cudnnActivationForward(cudnnHandle, actDesc, &alpha, td_vec[tdInx], src, &zero, td_vec[tdInx + 1], src));
		tdInx++;
		Log("Acti");
	}

	void Pool(float* src, float* dst)
	{
		int out_n, out_c, out_h, out_w;
		checkCUDA(cudnnGetPooling2dForwardOutputDim(maxPoolDesc, td_vec[tdInx], &out_n, &out_c, &out_h, &out_w));
		printf("Predict Pool Result %d/%d/%d/%d\n", out_n, out_c, out_h, out_w);
		checkCUDA(cudnnPoolingForward(cudnnHandle, maxPoolDesc, &alpha, td_vec[tdInx], src, &zero, td_vec[tdInx + 1], dst));
		tdInx++;
		Log("Pool");
	}

	void UnPool(float* src, float* dst)
	{
		Resize(src, td_vec[tdInx], dst, td_vec[tdInx + 1]);
		tdInx++;
		Log("UnPool");
	}

	void Add(float* src, float*src2, float* dst)
	{
		checkCUDA(cudnnOpTensor(cudnnHandle, opTensorDesc, &alpha, td_vec[tdInx + 1], src, &alpha, td_vec[tdInx + 1], src2, &zero, td_vec[tdInx + 1], dst));
		tdInx++;
		Log("Add");
	}

	void Add_Active(float* src, float*src2, float* dst){
		Add(src, src2, dst);
		Activate(dst);
	}

	void inference()
	{
		if (isDebug) printf("Network inference() \n");
		tdInx = variableInx = 0;

		NormalizeInput(inData_d, buffer1_d, -26.7f, 612);
		
		//0. CBN R
		ConvBN_Activate(buffer1_d);

		//1. P CBN R
		Pool(buffer1_d, feature0);
		ConvBN_Activate(feature0);
		
		//2. P CBN R
		Pool(feature0, feature1);
		ConvBN_Activate(feature1);
				
		//3. P CBN R
		Pool(feature1, feature2);
		ConvBN_Activate(feature2);
		
		//4. P CBN R
		Pool(feature2, feature3);
		ConvBN_Activate(feature3);		

		//5. PCBN R
		Pool(feature3, buffer2_d);
		ConvBN_Activate(buffer2_d);

		//6. C BN
		ConvBN(buffer2_d);

		//7. U A R 
		UnPool(buffer2_d, buffer1_d);
		Add_Active(feature3, buffer1_d, buffer2_d);

		//C BN R		
		ConvBN_Activate(buffer2_d);

		//8 C B
		ConvBN(buffer2_d);

		//9 U A R C B
		UnPool(buffer2_d,buffer1_d);
		Add_Active(feature2, buffer1_d, buffer2_d);
		ConvBN(buffer2_d);

		//10 U A R C B
		UnPool(buffer2_d, buffer1_d);
		Add_Active(feature1, buffer1_d, buffer2_d);
		ConvBN(buffer2_d);

		//11 U A R C B 
		UnPool(buffer2_d, buffer1_d);
		Add_Active(feature0, buffer1_d, buffer2_d);				
		ConvBN_Activate(buffer2_d);		

		ConvBN_Activate(buffer2_d);		
		ConvBN_Activate(buffer2_d);

		//15 U
		UnPool(buffer2_d, outData_d);

		if (td_vec.size() != tdInx + 1)
		{
			printf("[Warn] Operation?  %d != %d\n", td_vec.size(), tdInx + 1);
		}
		if (filter_td_vec.size() != variableInx){
			printf("[Warn] filter?  %d != %d\n", filter_td_vec.size(), variableInx);
		}

		printf("[INFO] Inference finished\n");
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
		GetKnownIndex << <h, w >> >((uchar*)dst, outData_d, c);
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

	void NormalizeInput(float* src, float* dst, float mean, float std)
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

		math_std_normal << <dim3(c, h), w >> >(dst, src, mean, std,epsilon);
	}
};

