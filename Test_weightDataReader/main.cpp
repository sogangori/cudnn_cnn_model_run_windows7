#include <stdio.h>

void read(float* src, char* path, int length)
{
	FILE* inf = fopen(path, "rb");

	int size = fread(src, sizeof(float), length, inf);
	printf("read %d\n", size);
	fclose(inf);
}

int main(int argc, char* argv[])
{
	int filter_len = 8408788;
	char * variablePath = "../weights/weight.dat";
	float* src = new float[filter_len];
	read(src, variablePath, filter_len);

	for (int i = 0; i < 5; i++)
	{
		printf("%d,%.3f\n", i, src[i]);
	}
	for (int i = 0; i < 10; i++)
	{
		int offset = filter_len - 1 - i;
		printf("%d,%.3f\n", offset, src[offset]);
	}
	return 0;
}

