#include <stdio.h>

#include "constant.h"

const int FILTER_DIM = 4;

char NetLayer[] = { 'c', 'n', 'p', 'c', 'n', 'u', 'a', 'c', 'n', 'c', 'n', 'c' };
int filterShape[][FILTER_DIM] = { { 3, 3, 3, 6 }, { 3, 3, 6, 12 }, { 3, 3, 12, 6 }, { 3, 3, 6, 3 }, { 3, 3, 3, 2 } };

int CheckArchitecture(int inputH,int inputW,int inputC)
{	
	int outW = inputW, outH = inputH, outC = inputC;
	int convolution_index = 0;
	printf("%s Feature Map shapes\n", CHAR_INFO);

	for (int i = 0; i < sizeof(NetLayer); i++)
	{
		char layer = NetLayer[i];
		if (layer == 'c')
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
		else if (layer == 'p'){
			outW /= 2;
			outH /= 2;
		}
		else if (layer == 'u'){
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