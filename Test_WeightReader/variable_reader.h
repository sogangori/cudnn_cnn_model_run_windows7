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
#endif