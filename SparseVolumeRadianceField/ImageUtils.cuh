#pragma once

#include "Utils.cuh"

typedef struct Image
{
	int width;
	int height;

	int bbox[4];
	int* masksIndex = 0, masksLen;
	int* freeAreaIndex = 0, freeAreaLen;

	bool grayImage = true;
	bool whiteBG = false;
	bool floatFormat = false;
	
	uchar* grayHost = 0;
	uchar* grayDevice = 0;
	float* grayFloatDevice = 0;

	uchar* maskDevice = 0, * maskHost = 0;
	uchar* redDevice = 0, * redHost = 0;
	uchar* greenDevice = 0, * greenHost = 0;
	uchar* blueDevice = 0, * blueHost = 0;

	float* redFloatDevice = 0;
	float* greenFloatDevice = 0;
	float* blueFloatDevice = 0;

	float* disparityDevice = 0, * disparityHost = 0;

	bool needAccumulateInfo = false;
	float* accumulatedOpacityDevice = 0;
	float* accumulatedRedDevice = 0, * accumulatedBlueDevice = 0, * accumulatedGreenDevice = 0, * accumulatedGrayDevice = 0;

	void initSelf
	(
		int imageWidth = 640, int imageHeight = 480,
		bool grayOnly = true,
		bool whiteBackGround = false,
		bool useMask = false,
		bool accumulate = false,
		bool needFloat = false
	);
	void initSelfFromFile(const char* file, int imageWidth = 640, int imageHeight = 480, bool grayOnly = true, bool useMask = false);
	
	void processMasksInfo();
	
	/*
	resize image to target size
	@param raw:raw image
	@param downSampleTimes:
	*/
	static void downSampleImage(
		Image src, Image& dest,
		const int downSampleTimes = 1);
	void copy2HostMemory();

	void saveSelf(const char* fileName,const bool needWhite = true);
	void freeSelf();
	void freeGPU();
}Image;