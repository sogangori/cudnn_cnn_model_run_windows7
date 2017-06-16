#include <stdio.h>
#include "Network.cuh"

char NetLayer[] = {
	POOL, CONV, BN, RELU,
	POOL, CONV, BN, RELU,
	POOL, CONV, BN, RELU,
	UN_POOL, UN_POOL, UN_POOL
};
int in_w = 256;
int in_h = 256;
int in_c = 12;
int label_c = 2;
int filterShape[][FILTER_DIM] = {
	{ 3, 3, in_c, in_c * 2 }, { 1, 1, 1, in_c * 2 }, { 1, 1, 1, in_c * 2 },
	{ 3, 3, in_c * 2, in_c * 4 }, { 1, 1, 1, in_c * 4 }, { 1, 1, 1, in_c * 4 },
	{ 3, 3, in_c * 4, label_c }, { 1, 1, 1, label_c }, { 1, 1, 1, label_c }
};

int CheckArchitecture(int in_h, int in_w, int inputC)
{	
	int convolution_index = 0;
	printf("%s Feature Map shapes\n", CHAR_INFO);

	printf("input (%d, %d, %d)\n", in_h, in_w, inputC);

	for (int i = 0; i < sizeof(NetLayer); i++)
	{
		char layer = NetLayer[i];
		if (layer == CONV)
		{
			int filterDepth = filterShape[convolution_index][2];
			if (inputC != filterDepth)
			{
				printf("%s data channel size (%d) is not Equal with (%d)th Filter channel (%d)\n",
					CHAR_ERROR, inputC, convolution_index, filterDepth);
				return -1;
			}
			inputC = filterShape[convolution_index][3];
			convolution_index++;
		}
		else if (layer == BN)
		{
			int filterCount = filterShape[convolution_index][3];
			if (inputC != filterCount)
			{
				printf("%s data channel size (%d) is not Equal with Filter channel (%d)\n",
					CHAR_ERROR, inputC, filterCount);
				return -1;
			}
			convolution_index += 2;
		}
		else if (layer == POOL){
			in_w /= 2;
			in_h /= 2;
		}
		else if (layer == UN_POOL){
			in_w *= 2;
			in_h *= 2;
		}
		
		printf("%d %c (%d, %d, %d)\n", i,layer, in_h, in_w, inputC);
	}

	int filter_count = sizeof(filterShape) / sizeof(int) / FILTER_DIM;
	if (filter_count != convolution_index){
		printf("%s filterCount (%d) is not Equal with convolution count in Network (%d)\n", CHAR_ERROR, filter_count, convolution_index);
		return -1;
	}

	return 0;
}

int CheckFilterCount(int in_h, int in_w, int inputC)
{
	int convolution_index = 0;
	printf("%s Check Filter Count\n", CHAR_INFO);

	printf("input (%d, %d, %d)\n", in_h, in_w, inputC);

	for (int i = 0; i < sizeof(NetLayer); i++)
	{
		char layer = NetLayer[i];
		if (layer == CONV)
		{
			int filterDepth = filterShape[convolution_index][2];
			if (inputC != filterDepth)
			{
				printf("%s data channel size (%d) is not Equal with (%d)th Filter channel (%d)\n",
					CHAR_ERROR, inputC, convolution_index, filterDepth);
				return -1;
			}
			inputC = filterShape[convolution_index][3];
			convolution_index++;
		}
		else if (layer == BN)
		{
			int filterCount = filterShape[convolution_index][3];
			if (inputC != filterCount)
			{
				printf("%s data channel size (%d) is not Equal with Filter channel (%d)\n",
					CHAR_ERROR, inputC, filterCount);
				return -1;
			}
			convolution_index += 2;
		}
	}

	int filter_count = sizeof(filterShape) / sizeof(int) / FILTER_DIM;
	if (filter_count != convolution_index){
		printf("%s filterCount (%d) is not Equal with convolution count in Network (%d)\n", CHAR_ERROR, filter_count, convolution_index);
		return -1;
	}

	return 0;
}

Network network;

int main(int argc, char* argv[])
{
	for (int i = 0; i < sizeof(filterShape) / sizeof(int) / FILTER_DIM; i++)
	{
		printf("filter %d (%d,%d,%d,%d)\n", i, filterShape[i][0], filterShape[i][1], filterShape[i][2], filterShape[i][3]);
	}
	for (int i = 0; i < sizeof(NetLayer); i++)
	{
		printf("%c", NetLayer[i]);
	}
	printf("\n");
	
	checkCPU(CheckFilterCount(in_h, in_w, in_c));

	char * variablePath = "../weights/weight_small.dat";		 
	char * dataPath = "c:/Users/pc/Documents/Visual Studio 2013/Projects/DopplerTrainPreProcess/IQApp_cuda/bin/x64/Debug/trainData/das9/das_301_05.dat";
	//char * dataPath = "c:/Users/pc/Documents/Visual Studio 2013/Projects/DopplerTrainPreProcess/IQApp_cuda/bin/x64/Debug/trainData/das9/das_301_11.dat";

	int mask_len = in_w * in_h;
	int input_len = in_c * mask_len;
	float* input = new float[input_len + mask_len];
	float* input_d;
	uchar * mask = new uchar[mask_len];
	uchar * mask_d;
	cudaMalloc(&input_d, input_len * sizeof(float));
	cudaMalloc(&mask_d, in_w*in_h);

	FILE *inf = fopen(dataPath, "rb");
	if (inf == NULL) {
		printf("ERROR Can't Read float File %s \n", dataPath);
		return 1;
	}

	size_t t = fread(input, sizeof(float), input_len + mask_len, inf);
	fclose(inf);
	printf("Read %d\n", t);
	if (t != input_len)  printf("[WARN] read count (%d) != (%d) \n", t, input_len);
		
	cudaMemcpy(input_d, input + mask_len, input_len * sizeof(float), cudaMemcpyHostToDevice);

	network.LoadWeight(variablePath, &filterShape[0][0], sizeof(filterShape) / sizeof(int));
	network.InitFilterDesc();
	network.CreateTensorDescriptor(NetLayer, sizeof(NetLayer), in_h, in_w, in_c);   
	network.Init(in_h, in_w, in_c);
	network.CopyInput(input_d);
	network.inference();
	network.GetInference(mask_d);
	cudaMemcpy(mask, mask_d, mask_len, cudaMemcpyDeviceToHost); 
	SaveImageFile("mask.bmp", mask, in_w, in_h);	

	return 0;
}