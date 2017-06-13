#include <stdio.h>
#include "constant.h"

char NetLayer[] = { 
	CONV, BN, POOL, 
	CONV, BN, UN_POOL, 
	ADD, CONV, BN, 
	CONV, BN, 
	CONV, BN
};
int filterShape[][FILTER_DIM] = {
	{ 3, 3, 3, 6 }, { 1, 1, 1, 6 }, { 1, 1, 1, 6 },
	{ 3, 3, 6, 12 }, { 1, 1, 1, 12 }, { 1, 1, 1, 12 },
	{ 3, 3, 12, 6 }, { 1, 1, 1, 6 }, { 1, 1, 1, 6 },
	{ 3, 3, 6, 3 }, { 1, 1, 1, 3 }, { 1, 1, 1, 3 },
	{ 3, 3, 3, 2 }, { 1, 1, 1, 2 }, { 1, 1, 1, 2 } };

int CheckArchitecture(int inputH,int inputW,int inputC)
{	
	int outW = inputW, outH = inputH, outC = inputC;
	int convolution_index = 0;
	printf("%s Feature Map shapes\n", CHAR_INFO);

	for (int i = 0; i < sizeof(NetLayer); i++)
	{
		char layer = NetLayer[i];
		if (layer == CONV)
		{
			int filterDepth = filterShape[convolution_index][2];
			if (inputC != filterDepth)
			{
				printf("%s data channel size (%d) is not Equal with Filter channel (%d)\n", 
					CHAR_ERROR, inputC, filterDepth);
				return -1;
			}
			outC = filterShape[convolution_index][3];
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
			outW /= 2;
			outH /= 2;
		}
		else if (layer == UN_POOL){
			outW *= 2;
			outH *= 2;
		}

		printf("%d %c (%d, %d, %d) \t->\t (%d, %d, %d) \n", i,
			layer, inputH, inputW, inputC, outH, outW, outC);

		inputH = outH;
		inputW = outW;
		inputC = outC;
	}

	int filter_count = sizeof(filterShape) / sizeof(int) / FILTER_DIM;
	if (filter_count != convolution_index){
		printf("%s filterCount (%d) is not Equal with convolution count in Network (%d)\n", CHAR_ERROR, filter_count, convolution_index);
		return -1;
	}

	return 0;
}

int main(int argc, char* argv[])
{
	for (int i = 0; i < sizeof(filterShape) / sizeof(int) / FILTER_DIM; i++)
	{
		printf("conv filter %d (%d,%d,%d,%d)\n", i, filterShape[i][0], filterShape[i][1], filterShape[i][2], filterShape[i][3]);
	}
	for (int i = 0; i < sizeof(NetLayer); i++)
	{
		printf("%d %c\n", i, NetLayer[i]);
	}

	int inputW = 128;
	int inputH = 128;
	int inputC = 3;
	int resout = CheckArchitecture(inputH, inputW, inputC);
	
	return 0;
}