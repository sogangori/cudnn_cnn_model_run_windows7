#include <stdio.h>
#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef unsigned char uchar;

int ReadBinary(const char* filePath, float* dst, int length,int offset) {

	FILE *inf = fopen(filePath, "rb");
	if (inf == NULL) {
		printf("ERROR Can't Read float File %s \n", filePath);
		return 1;
	}
	fseek(inf, sizeof(float)*offset * 280, SEEK_SET);
	size_t t = fread(dst, sizeof(float), length, inf);
	fclose(inf);
	if (t == length) return 0;
	else return 2;
}

void get_BMP_header(uchar *tmp, int  nx_pixel, int ny_pixel) {
	int j, size;

	int nx_pixel_0, nx_pixel_1;
	int ny_pixel_0, ny_pixel_1;

	nx_pixel_0 = nx_pixel % 256;
	nx_pixel_1 = nx_pixel / 256;

	ny_pixel_0 = ny_pixel % 256;
	ny_pixel_1 = ny_pixel / 256;

	if (nx_pixel % 4 == 0)
		size = nx_pixel * ny_pixel;
	else
		size = (nx_pixel + 4 - (nx_pixel % 4))*ny_pixel;

	/* BMP File header */

	/* UNIT - bfType */
	tmp[0] = (uchar) 'B';
	tmp[1] = (uchar) 'M';

	/* DWORD - dfSize : in bytes */
	tmp[5] = 0x00;
	tmp[4] = 0x00;
	tmp[3] = (size + 1078) / 256;
	tmp[2] = (size + 1078) % 256;

	/* UNIT - bfReserved1 : must be set to zero */
	tmp[7] = 0x00;
	tmp[6] = 0x00;

	/* UNIT - bfReserved2 : must be set to zero */
	tmp[9] = 0x00;
	tmp[8] = 0x00;

	/* DWORD - bf0ffBits : Specifies the byte offset from the BITMAPFILEHEADER structure to the actual bitmap data in the file */
	tmp[13] = 0x00;
	tmp[12] = 0x00;
	tmp[11] = 0x04;
	tmp[10] = 0x36;

	/* DWORD - biSize : Specifies the number of bytes required by the structure */
	tmp[17] = 0x00;
	tmp[16] = 0x00;
	tmp[15] = 0x00;
	tmp[14] = 0x28;

	/* LONG - biWidth : Specifies the width of the bitmap, in pixels */
	tmp[21] = 0x00;
	tmp[20] = 0x00;
	tmp[19] = nx_pixel_1;
	tmp[18] = nx_pixel_0;

	/* LONG - biHeight */
	tmp[25] = 0x00;
	tmp[24] = 0x00;
	tmp[23] = ny_pixel_1;
	tmp[22] = ny_pixel_0;

	/* WORD - biPlanes : Specifies the number of planes for the target device */
	tmp[27] = 0x00;
	tmp[26] = 0x01;

	/* WORD - biBitCount : Specifies the number of bits per pixel. 1,4,8 or 24 */
	tmp[29] = 0x00;
	tmp[28] = 0x08;

	/* DWORD - biCompression : Specifies the type of compression for a compressed bitmap. */
	tmp[33] = 0x00;
	tmp[32] = 0x00;
	tmp[31] = 0x00;
	tmp[30] = 0x00;

	/* DWORD - biSizeImage : Specifies the size, in bytes, of the image */
	tmp[37] = 0x00;
	tmp[36] = 0x00;
	tmp[35] = size / 256;
	tmp[34] = size % 256;

	/* LONG - biXPelsPerMeter */
	tmp[38] = 0x00;
	tmp[39] = 0x00;
	tmp[40] = 0x00;
	tmp[41] = 0x00;

	/* LONG - biYPelsPerMeter */
	tmp[42] = 0x00;
	tmp[43] = 0x00;
	tmp[44] = 0x00;
	tmp[45] = 0x00;

	/* DWORD - biClrUsed */
	tmp[46] = 0x00;
	tmp[47] = 0x00;
	tmp[48] = 0x00;
	tmp[49] = 0x00;

	/* DWORD - biClrImportant */
	tmp[50] = 0x00;
	tmp[51] = 0x00;
	tmp[52] = 0x00;
	tmp[53] = 0x00;

	/* Palette */
	for (j = 0; j<256; j++) {
		tmp[54 + j * 4] = (uchar)j;
		tmp[55 + j * 4] = (uchar)j;
		tmp[56 + j * 4] = (uchar)j;
		tmp[57 + j * 4] = (uchar)0;
	}
}

void SaveImageFile(char *path, uchar* image, int width, int height)
{
	unsigned char char_buff[1500];
	unsigned char temp_char;

	FILE *fp = fopen(path, "wb");

	get_BMP_header(char_buff, width, height);

	fwrite(char_buff, 1, 1078, fp);

	for (int i = 0; i<height; i++) {
		for (int j = 0; j <width; j++) {
			int idx = i*width + j;
			temp_char = image[idx];
			fwrite(&temp_char, 1, 1, fp);
		}
		if (width % 4 != 0) {
			for (int k = 0; k<4 - (width % 4); k++)
				fwrite(&temp_char, 1, 1, fp);
		}
	}

	fclose(fp);
}

int ReadInput(float* dst, int len, int offset, int index)
{
	char * folder = "c:/Users/pc/Documents/Visual Studio 2013/Projects/DopplerTrainPreProcess/IQApp_cuda/bin/x64/Debug/trainData/das_12_float";
	
	char * file = "das_301_03.dat";
	if (index == 1) file = "das_301_11.dat";
	char path[150];
	sprintf(path, "%s/%s", folder, file);
	printf("path : %s\n", path);

	return ReadBinary(path, dst, len, offset);
}

using funcWork = int(*)(int, void*);

int main()
{
	auto dll = LoadLibrary(TEXT("c:/Users/pc/Documents/Visual Studio 2013/Projects/cudnn_model_run_windows7/x64/Release/Network.dll"));
	if (dll == nullptr) {
		puts("로드 실패");
		exit(0);
	}
	else puts("로드 성공");
		
	auto externWork = (funcWork)GetProcAddress(dll, "externWork");	
	if (externWork == nullptr) printf("[ERROR] null. externWork \n");
	else printf("[OK] externWork\n");

	int w = 256;
	int h = 256;
	int c = 12;
	float *inData = new float[c * w * h];	
	uchar *outData = new uchar[w * h];
	float *inData_d;
	uchar *outData_d;

	cudaMalloc(&inData_d, c*w*h*sizeof(float));
	cudaMalloc(&outData_d, w*h*sizeof(uchar));
	int error = 0;
	error = externWork(0, NULL);
	if (0 != error){
		printf("[Error] externWork(0) %d\n");
		return;
	}
	error = externWork(1, NULL);
	error = externWork(2, NULL);

	char path[50];
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed_time_ms = 0.0f;
	float elapsed_time_mean_ms = 0.0f;		
	
	int iter = 2;
	for (int i = 0; i < iter; i++)
	{
		int result = ReadInput(inData, c * w * h, w * h, i);
		printf("File Read Result: %d\n", result);		
		cudaMemcpy(inData_d, inData, c*w*h*sizeof(float), cudaMemcpyHostToDevice);
		error = externWork(3, inData_d);
		cudaEventRecord(start);
		if (i > 0) elapsed_time_mean_ms += elapsed_time_ms;
		printf("[%d] inference() %.2f ms \n", i, elapsed_time_ms);
		externWork(4, outData_d);
		externWork(5, outData_d);
		cudaMemcpy(outData, outData_d, w*h, cudaMemcpyDeviceToHost);
		sprintf(path, "predict_%d.bmp", i);
		SaveImageFile(path, outData, w, h);
	
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed_time_ms, start, stop);
		printf("mean %.2f ms \n", elapsed_time_mean_ms / (iter - 1));
		externWork(5, inData_d);
	}
	return 0;
}