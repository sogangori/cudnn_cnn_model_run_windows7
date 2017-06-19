#include <iostream>
#include <cuda.h>
#include <cudnn.h>
#include <nppi.h>
#include <vector>

using namespace std;
typedef unsigned char uchar;
struct shape
{
	//CUDNN_TENSOR_NCHW
	int n;//갯수, 배치사이즈
	int c;//채널 수
	int h;//세로 길이
	int w;//가로 길이

	shape(){}
	shape(int nn, int cc, int hh, int ww){
		n = nn;
		c = cc;
		h = hh;
		w = ww;
	}

	int len(){
		return n*c*h*w;
	}
	void Print(){
		printf("NCHW %d/%d/%d/%d\n", n, c, h, w);
	}
};

void checkCPU(int status)
{
	if (status != 0)
	{
		cout << "[ERROR] CPU " << status << endl << endl;;
		exit(0);
	}
}

void checkCUDNN(cudnnStatus_t  status)
{
	if (status != CUDA_SUCCESS){
		cout << "[ERROR] CUDNN (" << status << ") " << cudnnGetErrorString(status) << endl << endl;;		
		exit(1);
	}
}

void checkCUDA(cudnnStatus_t status)
{
	checkCUDNN(status);
}

void checkCUDA(cudaError_t error)
{
	if (error != CUDA_SUCCESS)
		cout << "[ERROR] CUDA " << error << endl;
}

void checkNPP(NppStatus  error) {
	if (error != NPP_SUCCESS)
		cout << "[ERROR] NPP " << error << endl;
	
}

void print(char* title, float* src, int count, int c, int h, int w)
{
	printf("%s, NCHW %d/%d/%d/%d \n", title, count, c, h, w);
	for (int n = 0; n < count; n++)
	{
		for (int i = 0; i < c; i++) 
		{
			for (int y = 0; y < h; y++) 
			{
				for (int x = 0; x < w; x++) 
				{
					int index = n * c * h * w + i * h * w + y * w + x;
					printf("%.1f ", src[index]);
				}
				cout << endl;
			}
			cout << endl;
		}

	}
}

void print(char* title, float* src, int filter_num, int h, int w)
{
	cout << title << endl;
	for (int i = 0; i < filter_num; i++) {
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				printf("%.0f ", src[i*h * w + y * w + x]);
			}
			cout << endl;
		}
		cout << endl;
	}
}


float printBuffer[256*256*3];

void PrintTensor(char* title, float* tensor, cudnnTensorDescriptor_t descriptor)
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
	checkCUDA(cudaMemcpy(printBuffer, tensor, sizeof(float)* n * c * h * w, cudaMemcpyDeviceToHost));
	print(title, printBuffer, n, c, h, w);
}

void PrintDescriptor(int index, cudnnTensorDescriptor_t descriptor)
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
	printf("[Info] DescriptorShape(%d) %d x %d x %d x %d \n", index, n, c, h, w);
}

int GetTensorSize(cudnnTensorDescriptor_t descriptor)
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
	//printf("GetTensorSize() dataType : %d, %d x %d x %d x %d \n", dataType, n, c, h, w);
	return n* c* h* w;
}

int GetFilterSize(cudnnFilterDescriptor_t filterDesc)
{
	cudnnDataType_t                    dataType;
	cudnnTensorFormat_t                format;
	int                                k;        // number of inputs (batch size)
	int                                c;        // number of input feature maps
	int                                h;        // height of input section
	int                                w;        // width of input section
	cudnnGetFilter4dDescriptor(filterDesc, &dataType, &format, &k, &c, &h, &w);
	//printf("GetFilterSize()  %d x %d x %d x %d \n", k, c, h, w);
	return k* c* h* w;		
}

void Resize(float* src, cudnnTensorDescriptor_t srcDesc, float* dst, cudnnTensorDescriptor_t dstDesc)
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
	checkCUDA(cudnnGetTensor4dDescriptor(srcDesc, &dataType, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride));
	int                                dstH;
	int                                dstW;
	int                                dstC;
	checkCUDA(cudnnGetTensor4dDescriptor(dstDesc, &dataType, &n, &dstC, &dstH, &dstW, &nStride, &cStride, &hStride, &wStride));

	double nXFactor = 1.0 * dstW / (double)w;
	double nYFactor = 1.0 * dstH / (double)h;
	printf("Resize %d/%d/%d -> %d/%d/%d\, nFactor=%.2f/%.2f \n", c, w, h, dstC, dstW, dstH, nXFactor, nYFactor);

	for (int i = 0; i < c; i++)
	{
		int src_offset = i * w * h;
		int dst_offset = i * dstW * dstH;
		NppiSize oSrcSize = { w, h };
		NppiSize oDstSize = { dstW, dstH };
		NppiRect oSrcROI = { 0, 0, w, h };
		NppiRect oDstROI = { 0, 0, dstW, dstH };

		nppiResize_32f_C1R(&src[src_offset], oSrcSize, oSrcSize.width*sizeof(float), oSrcROI, &dst[dst_offset], oDstSize.width*sizeof(float), oDstSize, nXFactor, nYFactor, NPPI_INTER_LINEAR);//NPPI_INTER_NN,NPPI_INTER_LINEAR
	}
}

void Print(vector<cudnnTensorDescriptor_t> tensorDescriptor_vec){

	for (int i = 0; i < tensorDescriptor_vec.size(); i++)
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
		checkCUDA(cudnnGetTensor4dDescriptor(tensorDescriptor_vec[i], &dataType, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride));
		printf("GetTensorSize() [%d] %d x %d x %d x %d \n", i, n, c, h, w);
	}
}

__global__ void softMax2Uchar(uchar* dst, float* src,int channel)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x * channel;

	dst[idx] = src[idx + offset]*255;
}

__global__ void ConvertFloat2uchar(uchar* dst, float* src, int srcOffset)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	dst[idx] = src[idx + srcOffset] * 255;
}

__global__ void math_std_normal(float* dst, float* src, float *meanStdArray)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	float mean = meanStdArray[0];
	float deviation = meanStdArray[1];

	dst[index] = (src[index] - mean) / (deviation+0.001f);
}

__global__ void math_std_normal(float* dst, float* src, float mean, float deviation)
{	
	int index = blockIdx.x  * gridDim.y* blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;
	//dst[index] = (src[index] - mean) / (deviation + 0.001f);
	dst[index] = src[index] / (deviation + 0.001f);
}

__global__ void Std2Var(float* dst)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	dst[idx] = dst[idx] * dst[idx];
}

__global__ void batchNormal(float* dst, float* src, float *mean, float *var, float* gamma, float*betta)
{
	int index = blockIdx.x  * gridDim.y* blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;
	int c = blockIdx.x;	
	dst[index] = gamma[c] * (src[index] - mean[c]) / sqrt(var[c] + 0.001f) + betta[c];
	//y[i] = bnScale[k] * (x[i] - estimatedMean[k]) / sqrt(epsilon + estimatedVariance[k]) + bnBias[k]
}

__global__ void ArgMax(uchar* dst, float* src)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x;

	uchar v = 0;
	if (src[idx + offset] > src[idx]) v = 255;
	dst[idx] = v;

	//float v0 = src[idx + offset] * 20;
	//if (v0 > 255) v0 = 255;
	//dst[idx] = v0;
}