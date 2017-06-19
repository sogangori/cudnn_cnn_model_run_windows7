#ifndef VAREADER_H
#define VAREADER_H
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;


class variable_reader
{
public:

	char* path;
	int length;
	float* src;

	variable_reader(char* pat){
		path = pat;
	}

	void Print(){
		cout << path << " (" << length << ")" << endl;
	}

	int ReadAll()
	{
		FILE* inf = fopen(path, "r");
		if (inf == NULL)	{
			printf("ERROR Can't Read Text File(float) %s \n", path);
			return 2;
		}
		else{
			printf("EXIST Text File %s (%d)\n", path, length);
		}

		vector<float> src_vec;

		int result = 1;
		float value = 0;
		while (result>0) {
			result = fscanf(inf, "%f", &value);
			if (result>0) src_vec.push_back(value);
		}
		fclose(inf);
		length = src_vec.size();
		src = new float[length];
		std::copy(src_vec.begin(), src_vec.begin() + length, src);
		return length;
	}

};


void read(float* src, char* path, int length)
{
	FILE* inf = fopen(path, "rb");

	int size = fread(src, sizeof(float), length, inf);
	printf("read %d\n", size);
	fclose(inf);
}

void write(float* src, char* path, int length)
{
	FILE* inf = fopen(path, "wb");
	
	int size = fwrite(src, sizeof(float), length, inf);
	printf("write %d\n", size);
	fclose(inf);	
}

int main()
{
	char * variablePath = "c:/ultrasound/filter/network/variable_small_bn.txt";
	char * variablePath2 = "../weights/weight_small_bn.dat";
	variable_reader variable_read = variable_reader(variablePath);

	int filter_len = variable_read.ReadAll();
	printf("filter_len :%d \n", filter_len);

	write(variable_read.src, variablePath2, filter_len);
	float* src = new float[filter_len];
	read(src, variablePath2, filter_len);

	for (int i = 0; i < 5; i++)
	{
		printf("%d,%.3f,%.3f\n", i, variable_read.src[i], src[i]);
	}
	for (int i = 0; i < 10; i++)
	{
		int offset = filter_len - 1 - i;
		printf("%d,%.3f,%.3f\n", offset, variable_read.src[offset], src[offset]);
	}
	return 0;
}
#endif
