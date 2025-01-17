#include "ImageUtils.cuh"

void Image::initSelf(int imageWidth, int imageHeight, bool grayOnly, bool whiteBackGround, bool useMask, bool accumulate,
		bool needFloat)
{
	width = imageWidth;
	height = imageHeight;
	grayImage = grayOnly;
	whiteBG = whiteBackGround;
	floatFormat = needFloat;

	bbox[1] = 0;
	bbox[0] = imageHeight - 1;
	bbox[3] = 0;
	bbox[2] = imageWidth - 1;
	if (!useMask)
	{
		masksIndex = 0;
	}

	maskHost = (uchar*)malloc(sizeof(uchar) * width * height);
	cudaMalloc((void**)&maskDevice, sizeof(uchar) * width * height);

	disparityHost = (float*)calloc(width * height, sizeof(float));
	cudaMalloc((void**)&disparityDevice, sizeof(float) * width * height);
	if(floatFormat)
	{
		cudaMemcpy(disparityDevice, disparityHost, sizeof(float) * width * height, cudaMemcpyHostToDevice);
	}
		

	needAccumulateInfo = accumulate;
	if (needAccumulateInfo)
	{
		cudaMalloc((void**)&accumulatedOpacityDevice, sizeof(float) * width * height);
	}
	
	if (grayOnly)
	{
		if (needAccumulateInfo)
		{
			cudaMalloc((void**)&accumulatedGrayDevice, sizeof(float) * width * height);
		}

		grayHost = (uchar*)calloc(width * height, sizeof(uchar));
		cudaMalloc((void**)&grayDevice, sizeof(uchar) * width * height);
		if(floatFormat)
		{
			cudaMalloc((void**)&grayFloatDevice, sizeof(float) * imageWidth * imageHeight);
		}
	}
	else
	{
		if (needAccumulateInfo)
		{
			cudaMalloc((void**)&accumulatedRedDevice, sizeof(float) * width * height);
			cudaMalloc((void**)&accumulatedGreenDevice, sizeof(float) * width * height);
			cudaMalloc((void**)&accumulatedBlueDevice, sizeof(float) * width * height);

		}

		if(floatFormat)
		{
			cudaMalloc((void**)&redFloatDevice, sizeof(float) * width * height);
			cudaMalloc((void**)&greenFloatDevice, sizeof(float) * width * height);
			cudaMalloc((void**)&blueFloatDevice, sizeof(float) * width * height);
		}

		redHost = (uchar*)calloc(width * height, sizeof(uchar));
		cudaMalloc((void**)&redDevice, sizeof(uchar) * width * height);
		

		greenHost = (uchar*)calloc(width * height, sizeof(uchar));
		cudaMalloc((void**)&greenDevice, sizeof(uchar) * width * height);
		

		blueHost = (uchar*)calloc(width * height, sizeof(uchar));
		cudaMalloc((void**)&blueDevice, sizeof(uchar) * width * height);
		
	}
	checkGPUStatus("Initial Image");
}

void Image::initSelfFromFile(const char* file, int imageWidth, int imageHeight, bool grayOnly, bool useMask)
{
	initSelf(imageWidth, imageHeight, grayOnly, false, useMask);

	char maskFile[1024];
	sprintf(maskFile, "%s_mask.txt", file);
	readMask(maskHost, imageHeight, imageWidth, maskFile, bbox);
	cudaMemcpy(maskDevice,maskHost,sizeof(uchar)*imageHeight*imageWidth,cudaMemcpyHostToDevice);

	if (grayOnly)
	{
		char grayFile[1024];
		sprintf(grayFile, "%s.txt", file);

		readUcharFromFile(grayHost, imageHeight, imageWidth, grayFile);
		cudaMemcpy(grayDevice, grayHost, sizeof(uchar) * imageWidth * imageHeight, cudaMemcpyHostToDevice);
	}
	else
	{
		char channelSubfix[3][16] = { "_red.txt","_green.txt","_blue.txt" };
		char channels[3][255];
		for (int channel = 0; channel < 3; channel++)
		{
			sprintf(channels[channel], "%s%s", file, channelSubfix[channel]);
		}

		readUcharFromFile(redHost, imageHeight, imageWidth, channels[0]);	
		cudaMemcpy(redDevice, redHost, sizeof(uchar) * imageWidth * imageHeight, cudaMemcpyHostToDevice);
		readUcharFromFile(greenHost, imageHeight, imageWidth, channels[1]);
		cudaMemcpy(greenDevice, greenHost, sizeof(uchar) * imageWidth * imageHeight, cudaMemcpyHostToDevice);
		readUcharFromFile(blueHost, imageHeight, imageWidth, channels[2]);
		cudaMemcpy(blueDevice, blueHost, sizeof(uchar) * imageWidth * imageHeight, cudaMemcpyHostToDevice);
	}
	checkGPUStatus("Initial Image from file");

	if (!useMask)
	{
		masksLen = imageWidth * imageHeight;
		return;
	}

	processMasksInfo();


}

void Image::processMasksInfo()
{
	assert(bbox[1] > bbox[0] && bbox[3] > bbox[2]);
	
	masksLen = (bbox[1] + 1 - bbox[0]) * (bbox[3] + 1 - bbox[2]);
	masksIndex = (int*)malloc(sizeof(int) * masksLen);
	
	int count = 0;
	for (int row = bbox[0]; row < bbox[1] + 1; row++)
	{
		for (int col = bbox[2]; col < bbox[3] + 1; col++)
		{
			masksIndex[count++] = row * width + col;
		}
	}

	freeAreaLen = width * height - masksLen;
	freeAreaIndex = (int*)malloc(sizeof(int) * freeAreaLen);

	count = 0;
	for (int row = 0; row < height; row++)
	{
		int col = 0;
		if (row < bbox[0] || row>bbox[1])
		{
			while (col < width)
			{
				freeAreaIndex[count] = row * width + col;
				++col;
				++count;
			}
		}
		else
		{
			while (col < bbox[2])
			{
				freeAreaIndex[count] = row * width + col;
				++col;
				++count;
			}
			col = bbox[3] + 1;
			while (col < width)
			{
				freeAreaIndex[count] = row * width + col;
				++col;
				++count;
			}
		}
	}
}

__global__ void downSampleImageGPU(Image src, Image dest, int downSampleTimes)
{
	int pixelPos = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	if (pixelPos >= dest.width * dest.height)return;
	int pixelRow = pixelPos / dest.width;
	int pixelCol = pixelPos % dest.width;

	int srcRow = pixelRow * 2;
	int srcCol = pixelCol * 2;

	int averageValueDensity = 0;
	int averageValueRed = 0;
	int averageValueGreen = 0;
	int averageValueBlue = 0;

	int count = 0;
	for (int row = -downSampleTimes; row < downSampleTimes + 1; row++)
	{
		srcRow = pixelRow * pow(2, downSampleTimes) + row;
		for (int col = -downSampleTimes; col < downSampleTimes + 1; col++)
		{
			srcCol = pixelCol * pow(2, downSampleTimes) + col;
			if (
				srcRow < 0 ||
				srcRow >= src.height ||
				srcCol < 0 ||
				srcCol >= src.width)
			{
				continue;
			}
			++count;
			if (src.grayImage)
			{
				averageValueDensity += src.grayDevice[srcRow * src.width + srcCol];
			}
			else
			{
				averageValueRed += src.redDevice[srcRow * src.width + srcCol];
				averageValueGreen += src.greenDevice[srcRow * src.width + srcCol];
				averageValueBlue += src.blueDevice[srcRow * src.width + srcCol];
			}
		}
	}

	if (src.grayImage)
	{
		dest.grayDevice[pixelRow * dest.width + pixelCol] = averageValueDensity / count;
	}
	else
	{
		dest.redDevice[pixelRow * dest.width + pixelCol] = averageValueRed / count;
		dest.greenDevice[pixelRow * dest.width + pixelCol] = averageValueGreen / count;
		dest.blueDevice[pixelRow * dest.width + pixelCol] = averageValueBlue / count;
	}
}

void Image::downSampleImage(
	Image src, Image& dest,
	const int downSampleTimes)
{
	int targetWidth = src.width / pow(2, downSampleTimes);
	int targetHeight = src.height / pow(2, downSampleTimes);
	dest.initSelf(targetWidth, targetHeight, src.grayImage);
	dim3 grid((targetWidth + 15) / 16, (targetHeight + 15) / 16);
	dim3 block(16, 16);
	downSampleImageGPU << <grid, block >> > (src, dest, downSampleTimes);
	checkGPUStatus("downsample image");
}

void Image::copy2HostMemory()
{
	cudaMemcpy(maskHost,maskDevice,sizeof(uchar)*width*height,cudaMemcpyDeviceToHost);
	cudaMemcpy(disparityHost, disparityDevice, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	if (grayImage)
	{
		cudaMemcpy(grayHost, grayDevice, sizeof(uchar) * width * height, cudaMemcpyDeviceToHost);
	}
	else
	{
		cudaMemcpy(redHost, redDevice, sizeof(uchar) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(greenHost, greenDevice, sizeof(uchar) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(blueHost, blueDevice, sizeof(uchar) * width * height, cudaMemcpyDeviceToHost);
	}
	checkGPUStatus("Transfer Image to Main memory");
}

/*
@param file:directory for saving files
@return None
*/
void Image::saveSelf(const char* saveFileName,const bool needWhite)
{
	char disparityFile[256];
	sprintf(disparityFile, "%s_disparity.txt", saveFileName);
	saveFloaters2File(disparityHost, height, width, disparityFile);

	char maskFile[256];
	sprintf(maskFile,"%s_mask.txt",saveFileName);
	saveUchar2File(maskHost,height,width,maskFile);

	if (grayImage)
	{
		char grayImageFile[256];
		sprintf(grayImageFile, "%s_graylized.txt", saveFileName);
		saveUchar2File(grayHost, height, width, grayImageFile);
	}
	else
	{
		char channelSubfix[3][16] = { "_red.txt","_green.txt","_blue.txt" };
		char channelSubfixFloaters[3][64] = { "_red_Floaters.txt","_green_Floaters.txt","_blue_Floaters.txt" };
		char channels[3][1024];
		char channelsFloaters[3][1024];
		for (int channel = 0; channel < 3; channel++)
		{
			sprintf(channels[channel], "%s%s", saveFileName, channelSubfix[channel]);
			sprintf(channelsFloaters[channel], "%s%s", saveFileName, channelSubfixFloaters[channel]);
		}
		if (!whiteBG&&needWhite)
		{
			saveUchar2File(redHost, height, width, channels[0]);
			saveUchar2File(greenHost, height, width, channels[1]);
			saveUchar2File(blueHost, height, width, channels[2]);
		}

		cudaMemcpy(disparityHost, redFloatDevice, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		saveFloaters2File(disparityHost, height, width, channelsFloaters[0]);
		cudaMemcpy(disparityHost, greenFloatDevice, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		saveFloaters2File(disparityHost, height, width, channelsFloaters[1]);
		cudaMemcpy(disparityHost, blueFloatDevice, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		saveFloaters2File(disparityHost, height, width, channelsFloaters[2]);
		

	}
}

void Image::freeSelf()
{
	if(floatFormat)
	{
		cudaFree(disparityDevice);
	}
	
	free(disparityHost);
	cudaFree(maskDevice);
	free(maskHost);

	if (grayImage)
	{
		free(grayHost);
		cudaFree(grayDevice);
		if(floatFormat)
		{
			cudaFree(grayFloatDevice);	
		}
	}

	if (!grayImage)
	{
		free(redHost);
		free(greenHost);
		free(blueHost);

		cudaFree(redDevice);
		cudaFree(greenDevice);
		cudaFree(blueDevice);

		if(floatFormat)
		{
			cudaFree(redFloatDevice);
			cudaFree(greenFloatDevice);
			cudaFree(blueFloatDevice);
		}
	}
	if (masksIndex)
	{
		free(masksIndex);
	}

	if (needAccumulateInfo)
	{
		cudaFree(accumulatedOpacityDevice);
		if (grayImage)
		{
			cudaFree(accumulatedGrayDevice);
		}
		else
		{
			cudaFree(accumulatedRedDevice);
			cudaFree(accumulatedGreenDevice);
			cudaFree(accumulatedBlueDevice);
		}
	}
	checkGPUStatus("Free Image");
}
