#include <stdio.h>

typedef unsigned char uchar;

void Read(char* path, float* src, int length)
{
	FILE* inf = fopen(path, "rb");

	int size = fread(src, sizeof(float), length, inf);
	printf("read %d\n", size);
	fclose(inf);
}

int ReadBinary(const char* filePath, float* dst, int length) {

	FILE *inf = fopen(filePath, "rb");
	if (inf == NULL) {
		printf("ERROR Can't Read float File %s \n", filePath);
		return 1;
	}

	size_t t = fread(dst, sizeof(float), length, inf);
	fclose(inf);
	if (t == length) return 0;
	else {
		printf("[WARN] read count (%d) != (%d) \n", t, length);
		return 2;
	}
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
	printf("changed original\n");
}

void ImageSave(float* src, int width, int height, char* path) {

	uchar * normaled = new uchar[width*height];

	for (int i = 0; i < width*height; i++)
	{
		normaled[i] = src[i] * 255;
	}

	SaveImageFile(path, normaled, width, height);
	delete[] normaled;
}

void RotateFilterHWCN2NCHW(float*src, float*dst, int* filterShapePtr, int filterCount)
{
	//필터 돌리자 HWCN -> NCHW
	int v_offset = 0;
	for (int i = 0; i < filterCount; i++)
	{
		int offset = i * 4;
		int height = filterShapePtr[offset + 0];
		int width = filterShapePtr[offset + 1];
		int channel = filterShapePtr[offset + 2];
		int kcount = filterShapePtr[offset + 3];
		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				for (int c = 0; c < channel; c++)
				{
					for (int k = 0; k < kcount; k++)
					{
						int index_in = v_offset
							+ (h * width * channel * kcount)
							+ (w * channel * kcount)
							+ (c * kcount)
							+ k;
						int index_out = v_offset
							+ k * channel * height * width
							+ c * height * width
							+ h * width
							+ w;
						dst[index_out] = src[index_in];
					}
				}
			}
		}
		v_offset += height*width*channel*kcount;
	}
}