#include <stdio.h>
#include "Network.cuh"

char NetLayer[] = {
	CONV, BN, RELU,
	POOL, CONV, BN, RELU,
	POOL, CONV, BN, RELU,
	POOL, CONV, BN, RELU,
	POOL, CONV, BN, RELU,
	POOL, CONV, BN, RELU,
	CONV, BN,
	UN_POOL, ADD, RELU, CONV, BN, RELU,
	CONV, BN,
	UN_POOL, ADD, RELU, CONV, BN,
	UN_POOL, ADD, RELU, CONV, BN,
	UN_POOL, ADD, RELU,
	CONV, BN, RELU,
	CONV, BN, RELU,
	CONV, BN, RELU,
	UN_POOL
};
int in_w = 256;
int in_h = 256;
int in_c = 12;
int label_c = 2;

int filterShape[][FILTER_DIM] = {

	{ 3, 3, in_c, in_c * 2 }, { 1, 1, in_c * 2, 1 }, { 1, 1, in_c * 2, 1 },
	{ 3, 3, in_c * 2, in_c * 4 }, { 1, 1, in_c * 4, 1 }, { 1, 1, in_c * 4, 1 },
	{ 3, 3, in_c * 4, in_c * 8 }, { 1, 1, in_c * 8, 1 }, { 1, 1, in_c * 8, 1 },
	{ 3, 3, in_c * 8, in_c * 16 }, { 1, 1, in_c * 16, 1 }, { 1, 1, in_c * 16, 1 },
	{ 3, 3, in_c * 16, in_c * 32 }, { 1, 1, in_c * 32, 1 }, { 1, 1, in_c * 32, 1 },
	{ 3, 3, in_c * 32, in_c * 64 }, { 1, 1, in_c * 64, 1 }, { 1, 1, in_c * 64, 1 },
	{ 3, 3, in_c * 64, in_c * 32 }, { 1, 1, in_c * 32, 1 }, { 1, 1, in_c * 32, 1 },
	{ 3, 3, in_c * 32, in_c * 32 }, { 1, 1, in_c * 32, 1 }, { 1, 1, in_c * 32, 1 },
	{ 3, 3, in_c * 32, in_c * 16 }, { 1, 1, in_c * 16, 1 }, { 1, 1, in_c * 16, 1 },
	{ 3, 3, in_c * 16, in_c * 8 }, { 1, 1, in_c * 8, 1 }, { 1, 1, in_c * 8, 1 },
	{ 3, 3, in_c * 8, in_c * 4 }, { 1, 1, in_c * 4, 1 }, { 1, 1, in_c * 4, 1 },
	{ 3, 3, in_c * 4, in_c * 2 }, { 1, 1, in_c * 2, 1 }, { 1, 1, in_c * 2, 1 },
	{ 3, 3, in_c * 2, in_c * 1 }, { 1, 1, in_c * 1, 1 }, { 1, 1, in_c * 1, 1 },
	{ 3, 3, in_c * 1, label_c }, { 1, 1, label_c, 1 }, { 1, 1, label_c, 1 }
};
int CheckArchitecture(int in_h, int in_w, int inputC)
{
	int filter_index = 0;
	printf("%s Feature Map shapes\n", CHAR_INFO);

	printf("input (%d, %d, %d)\n", in_h, in_w, inputC);

	for (int i = 0; i < sizeof(NetLayer); i++)
	{
		char layer = NetLayer[i];
		if (layer == CONV)
		{
			int filterDepth = filterShape[filter_index][2];
			if (inputC != filterDepth)
			{
				printf("%s CONV data channel size (%d) is not Equal with (%d)th Filter channel (%d)\n",
					CHAR_ERROR, inputC, filter_index, filterDepth);
				return -1;
			}
			inputC = filterShape[filter_index][3];
			filter_index++;
		}
		else if (layer == BN || layer == BIAS)
		{
			int filterCount = filterShape[filter_index][2];
			if (inputC != filterCount)
			{
				printf("%s BN,BAIS data channel size (%d) is not Equal with Filter channel (%d)\n",
					CHAR_ERROR, inputC, filterCount);
				return -1;
			}
			if (layer == BIAS)filter_index++;
			else filter_index += 2;
		}
		else if (layer == POOL){
			in_w /= 2;
			in_h /= 2;
		}
		else if (layer == UN_POOL){
			in_w *= 2;
			in_h *= 2;
		}

		printf("%d %c (%d, %d, %d)\n", i, layer, in_h, in_w, inputC);
	}

	int filter_count = sizeof(filterShape) / sizeof(int) / FILTER_DIM;
	if (filter_count != filter_index){
		printf("%s filterCount (%d) is not Equal with convolution count in Network (%d)\n", CHAR_ERROR, filter_count, filter_index);
		return -1;
	}

	return 0;
}

int CheckFilterCount(int in_h, int in_w, int inputC)
{
	int filter_index = 0;
	printf("%s Check Filter Count\n", CHAR_INFO);

	printf("input (%d, %d, %d)\n", in_h, in_w, inputC);

	for (int i = 0; i < sizeof(NetLayer); i++)
	{
		char layer = NetLayer[i];
		if (layer == CONV)
		{
			int filterDepth = filterShape[filter_index][2];
			if (inputC != filterDepth)
			{
				printf("%s Check CONV data channel size (%d) is not Equal with (%d)th Filter channel (%d)\n",
					CHAR_ERROR, inputC, filter_index, filterDepth);
				return -1;
			}
			inputC = filterShape[filter_index][3];
			filter_index++;
		}
		else if (layer == BN || layer == BIAS)
		{
			int filterCount = filterShape[filter_index][2];
			if (inputC != filterCount)
			{
				printf("%s Check BN,BAIS data channel size (%d) is not Equal with Filter channel (%d)\n",
					CHAR_ERROR, inputC, filterCount);
				return -1;
			}
			if (layer == BIAS)filter_index++;
			else filter_index += 2;
		}
	}

	int filter_count = sizeof(filterShape) / sizeof(int) / FILTER_DIM;
	if (filter_count != filter_index){
		printf("%s filterCount (%d) is not Equal with convolution count in Network (%d)\n", CHAR_ERROR, filter_count, filter_index);
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

	char * variablePath = "c:/Users/pc/Documents/Visual Studio 2013/Projects/cudnn_model_run_windows7/weights/weight.dat";
	char * dataPath = "c:/Users/pc/Documents/Visual Studio 2013/Projects/DopplerTrainPreProcess/IQApp_cuda/bin/x64/Debug/trainData/das9/das_301_00.dat";
	//char * dataPath = "c:/Users/pc/Documents/Visual Studio 2013/Projects/DopplerTrainPreProcess/IQApp_cuda/bin/x64/Debug/trainData/das9/das_301_10.dat";

	int mask_len = in_w * in_h;
	int input_len = in_c * mask_len;
	int data_len = input_len + mask_len;
	float* input = new float[data_len];
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

	size_t t = fread(input, sizeof(float), data_len, inf);
	fclose(inf);
	printf("Read %d\n", t);
	if (t != data_len)  printf("[WARN] read count (%d) != (%d) \n", t, data_len);

	if (in_w<10)
	for (int i = 0; i < data_len; i++) input[i] = 1;

	cudaMemcpy(input_d, input + mask_len, input_len * sizeof(float), cudaMemcpyHostToDevice);

	network.LoadWeight(variablePath, &filterShape[0][0], sizeof(filterShape) / sizeof(int));
	//network.InitFilterDesc();
	network.CreateTensorDescriptor(NetLayer, sizeof(NetLayer), in_h, in_w, in_c);
	network.Init(in_h, in_w, in_c);
	network.CopyInput(input_d);
	network.inference();
	network.GetInference(mask_d);
	cudaMemcpy(mask, mask_d, mask_len, cudaMemcpyDeviceToHost);
	SaveImageFile("mask.bmp", mask, in_w, in_h);

	return 0;
}