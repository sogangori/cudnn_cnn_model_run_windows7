#include <stdio.h>
#include "variable_reader.h"

const int FILTER_DIM = 4;

int main(int argc, char* argv[])
{
	int filterShape2[][FILTER_DIM] = { 
		{ 3, 3, 3, 3 }, { 1, 1, 1, 3 },
		{ 3, 3, 3, 3 }, { 1, 1, 1, 3 },
		{ 3, 3, 3, 3 }, { 1, 1, 1, 3 },
		{ 3, 3, 3, 3 }, { 1, 1, 1, 3 },
		{ 3, 3, 3, 3 }, { 1, 1, 1, 3 },
		{ 3, 3, 3, 3 }, { 1, 1, 1, 3 },
		{ 3, 3, 3, 3 }, { 1, 1, 1, 3 },
		{ 3, 3, 3, 3 }, { 1, 1, 1, 3 },
		{ 3, 3, 3, 3 }, { 1, 1, 1, 3 },
		{ 3, 3, 3, 3 }, { 1, 1, 1, 3 },
		{ 3, 3, 3, 3 }, { 1, 1, 1, 3 },
		{ 3, 3, 3, 3 }, { 1, 1, 1, 3 },
		{ 3, 3, 3, 3 }, { 1, 1, 1, 3 },
		{ 3, 3, 3, 2 }, { 1, 1, 1, 2 }
	};
	char * variablePath2 = "../weights/variable.txt";
	
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

	int filter_count_in_network = 0;
	for (int i = 0; i < sizeof(filterShape) / sizeof(int) / FILTER_DIM; i++)
	{
		int count = filterShape[i][0] * filterShape[i][1] * filterShape[i][2] * filterShape[i][3];
		filter_count_in_network += count;
	}
	printf("filter_count_in_network:%d\n", filter_count_in_network);

	char * variablePath = "../weights/variable2.txt";
	variable_reader variable_read = variable_reader(variablePath);
	
	int filter_len = variable_read.ReadAll();
	printf("filter_len :%d \n", filter_len);
	
	for (int i = 0; i < 10; i++)
	{
		printf("%.2f ", variable_read.src[i]);
	}
	printf("\n");


	if (filter_len != filter_count_in_network){
		printf("[ERROR] filter_len(%d) != filter_count_in_network(%d) \n", filter_len, filter_count_in_network);
	}
	return 0;
}

