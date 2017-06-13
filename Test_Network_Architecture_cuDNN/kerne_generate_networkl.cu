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
	{ 3, 3, in_c, in_c * 2 }, { 1, 1, 1, in_c * 2 }, { 1, 1, 1, in_c * 2 },
	{ 3, 3, in_c * 2, in_c * 4 }, { 1, 1, 1, in_c * 4 }, { 1, 1, 1, in_c * 4 },
	{ 3, 3, in_c * 4, in_c * 8 }, { 1, 1, 1, in_c * 8 }, { 1, 1, 1, in_c * 8 },
	{ 3, 3, in_c * 8, in_c * 16 }, { 1, 1, 1, in_c * 16 }, { 1, 1, 1, in_c * 16 },
	{ 3, 3, in_c * 16, in_c * 32 }, { 1, 1, 1, in_c * 32 }, { 1, 1, 1, in_c * 32 },
	{ 3, 3, in_c * 32, in_c * 64 }, { 1, 1, 1, in_c * 64 }, { 1, 1, 1, in_c * 64 },
	{ 3, 3, in_c * 64, in_c * 32 }, { 1, 1, 1, in_c * 32 }, { 1, 1, 1, in_c * 32 },
	{ 3, 3, in_c * 32, in_c * 32 }, { 1, 1, 1, in_c * 32 }, { 1, 1, 1, in_c * 32 },
	{ 3, 3, in_c * 32, in_c * 16 }, { 1, 1, 1, in_c * 16 }, { 1, 1, 1, in_c * 16 },
	{ 3, 3, in_c * 16, in_c * 8 }, { 1, 1, 1, in_c * 8 }, { 1, 1, 1, in_c * 8 },
	{ 3, 3, in_c * 8, in_c * 4 }, { 1, 1, 1, in_c * 4 }, { 1, 1, 1, in_c * 4 },
	{ 3, 3, in_c * 4, in_c * 2 }, { 1, 1, 1, in_c * 2 }, { 1, 1, 1, in_c * 2 },
	{ 3, 3, in_c * 2, in_c * 1 }, { 1, 1, 1, in_c * 1 }, { 1, 1, 1, in_c * 1 },
	{ 3, 3, in_c * 1, label_c }, { 1, 1, 1, label_c }, { 1, 1, 1, label_c }
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

	char * variablePath = "../weights/weight.dat";		 
	
	network.LoadWeight(variablePath, &filterShape[0][0], sizeof(filterShape) / sizeof(int));
	network.InitFilterDesc();
	network.CreateTensorDescriptor(NetLayer, sizeof(NetLayer), in_h, in_w, in_c);   

	return 0;
}