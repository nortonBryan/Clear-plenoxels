#include "SceneUtils.cuh"

__global__ void initialSelfGPU(ObjectScene src, float density, float colors)
{
	int pixelPos = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	if (pixelPos >= src.sceneWidth * src.sceneHeight)return;
	int pixelRow = pixelPos / src.sceneWidth;
	int pixelCol = pixelPos % src.sceneWidth;
	unsigned int srcSceneIndex, destSceneIndex;
	double distance;
	for (int depth = 0; depth < src.sceneDepth; depth++)
	{
		srcSceneIndex = depth * src.sceneWidth * src.sceneHeight + pixelRow * src.sceneWidth + pixelCol;
		src.sceneOpacityDevice[srcSceneIndex] = density;
		src.adaptiveLRDevice[srcSceneIndex] = 1.f;
		if (src.grayImage)
		{
			src.AmbientGrayDevice[srcSceneIndex] = colors;
		}
		else
		{
			// src.AmbientRedDevice[0][srcSceneIndex] = colors;
			// src.AmbientGreenDevice[0][srcSceneIndex] = colors;
			// src.AmbientBlueDevice[0][srcSceneIndex] = colors;

			for (int i = 0; i < src.sh_dim; i++)
			{
				src.AmbientRedDevice[i][srcSceneIndex] = colors;
				src.AmbientGreenDevice[i][srcSceneIndex] = colors;
				src.AmbientBlueDevice[i][srcSceneIndex] = colors;
			}
		}
	}
}

void initialChannel(float**& channelHostPtr, float**& channelDevicePtr, const int len, const int sh_dim)
{
	float** gpuPtr = (float**)malloc(sizeof(float*) * sh_dim);
	channelHostPtr = (float**)malloc(sizeof(float*) * sh_dim);

	cudaMalloc((void**)&channelDevicePtr, sizeof(float*) * sh_dim);

	for (int dimIdx = 0; dimIdx < sh_dim; dimIdx++)
	{
		float* dimArray;
		cudaMalloc((void**)&dimArray, sizeof(float) * len);
		gpuPtr[dimIdx] = dimArray;

		float* dimArrayHost = (float*)malloc(sizeof(float) * len);
		channelHostPtr[dimIdx] = dimArrayHost;
	}

	cudaMemcpy(channelDevicePtr, gpuPtr, sizeof(float*) * sh_dim, cudaMemcpyHostToDevice);

	free(gpuPtr);
	checkGPUStatus("initialChannel");
}

void objectScene::initSelf
(
	int width, int height, int depth,
	bool graylized, bool solid,
	float initialDensity, float initialColor,
	float gridStepSize, const int sh_dim,
	const int cagedNums
)
{
	sceneWidth = width;
	sceneHeight = height;
	sceneDepth = depth;

	printf("sh_dim = %d\n", sh_dim);
	assert(isSquareNumber(sh_dim));
	this->sh_dim = sh_dim;
	this->solid = solid;
	if (solid)
	{
		actualGridNums = width * height * depth;
		sparse = false;
	}
	else
	{
		actualGridNums = cagedNums;
		inCageNums = cagedNums;
		sparse = true;
	}

	printf("Actual Grid nums = %d\n", actualGridNums);

	grayImage = graylized;
	this->stepSize = gridStepSize;

	// gaussian puturbation sample========================================================================
	{
		maxSamplePoint = (int)(sqrt(width * width + height * height + depth * depth) / gridStepSize + 1.5);
		double* tempHost = (double*)calloc(maxSamplePoint, sizeof(double));
		cudaMalloc((void**)&offsetXDevice, sizeof(double) * maxSamplePoint);
		cudaMemcpy(offsetXDevice, tempHost, sizeof(double) * maxSamplePoint, cudaMemcpyHostToDevice);

		cudaMalloc((void**)&offsetYDevice, sizeof(double) * maxSamplePoint);
		cudaMemcpy(offsetYDevice, tempHost, sizeof(double) * maxSamplePoint, cudaMemcpyHostToDevice);

		cudaMalloc((void**)&offsetZDevice, sizeof(double) * maxSamplePoint);
		cudaMemcpy(offsetZDevice, tempHost, sizeof(double) * maxSamplePoint, cudaMemcpyHostToDevice);
		free(tempHost);
	}
	// gaussian puturbation sample========================================================================
	optimizingBand = 0;

	bbox_width_min = 0;
	bbox_width_max = width - 1;
	bbox_height_min = 0;
	bbox_height_max = height - 1;
	bbox_depth_min = 0;
	bbox_depth_max = depth - 1;

	occupancyHost = (uchar*)malloc(sizeof(uchar) * (width - 1) * (height - 1) * (depth - 1));
	cudaMalloc((void**)&occupancyDevice, sizeof(uchar) * (width - 1) * (height - 1) * (depth - 1));

	cudaMalloc((void**)&adaptiveLRDevice, sizeof(float) * width * height * depth);
	
	cudaMalloc((void**)&sceneOpacityDevice, sizeof(float) * actualGridNums);

	sceneOpacityHost = (float*)malloc(actualGridNums * sizeof(float));

	if (graylized)
	{
		AmbientGrayHost = (float*)malloc(actualGridNums * sizeof(float));
		cudaMalloc((void**)&AmbientGrayDevice, sizeof(float) * actualGridNums);
	}
	else
	{
		initialChannel(AmbientRedHost, AmbientRedDevice, actualGridNums, sh_dim);
		initialChannel(AmbientGreenHost, AmbientGreenDevice, actualGridNums, sh_dim);
		initialChannel(AmbientBlueHost, AmbientBlueDevice, actualGridNums,sh_dim);
	}

	if (solid)
	{
		dim3 grid((sceneWidth + 31) / 32, (sceneHeight + 31) / 32);
		dim3 block(32, 32);
		initialSelfGPU << <grid, block >> > (*this, initialDensity, initialColor);
	}

	if (!graylized)
	{
		totalVariableGPUPtrs = (float**)malloc(sizeof(float*) * (1 + 3 * sh_dim));
		totalVariableGPUPtrs[0] = sceneOpacityDevice;

		float** red = (float**)malloc(sizeof(float*) * sh_dim);
		cudaMemcpy(red, AmbientRedDevice, sizeof(float*) * sh_dim, cudaMemcpyDeviceToHost);

		float** green = (float**)malloc(sizeof(float*) * sh_dim);
		cudaMemcpy(green, AmbientGreenDevice, sizeof(float*) * sh_dim, cudaMemcpyDeviceToHost);

		float** blue = (float**)malloc(sizeof(float*) * sh_dim);
		cudaMemcpy(blue, AmbientBlueDevice, sizeof(float*) * sh_dim, cudaMemcpyDeviceToHost);

		for (int dimIdx = 0; dimIdx < sh_dim; dimIdx++)
		{
			totalVariableGPUPtrs[1 + dimIdx] = red[dimIdx];
			totalVariableGPUPtrs[1 + sh_dim + dimIdx] = green[dimIdx];
			totalVariableGPUPtrs[1 + 2 * sh_dim + dimIdx] = blue[dimIdx];
			// printf("Done\n");
		}
		free(red);
		free(green);
		free(blue);
		cudaMalloc((void**)&totalVariableGPUPtrsDevice, sizeof(float*) * (1 + 3 *sh_dim));
		cudaMemcpy(totalVariableGPUPtrsDevice, totalVariableGPUPtrs, sizeof(float*) * (1 + 3 *sh_dim), cudaMemcpyHostToDevice);
	}

	checkGPUStatus("Initial scene", true);
}

void objectScene::initSelfFromStackImage(
	const char* imageName,
	int width, int height, int depth,
	bool graylized)
{
	/*
	sceneWidth = width;
	sceneHeight = height;
	sceneDepth = depth;
	grayImage = graylized;

	bbox_width_min = 0;
	bbox_width_max = width;
	bbox_height_min = 0;
	bbox_height_max = height;
	bbox_depth_min = 0;
	bbox_depth_max = depth;

	if (graylized)
	{
		uchar* density = (uchar*)calloc(width * height, sizeof(uchar));
		readUcharFromFile(density, height, width, imageName);

		AmbientGrayHost = (float*)calloc(width * height * depth, sizeof(float));
		stackArray<uchar, float>(density, AmbientGrayHost, height, width, depth);
		cudaMalloc((void**)&AmbientGrayDevice, sizeof(float) * width * height * depth);
		cudaMemcpy(AmbientGrayDevice, AmbientGrayHost, sizeof(float) * width * height * depth, cudaMemcpyHostToDevice);
		free(density);
	}
	else
	{
		char channelSubfix[3][16] = { "_red.txt","_green.txt","_blue.txt" };
		char channels[3][255];
		for (int channel = 0; channel < 3; channel++)
		{
			sprintf(channels[channel], "%s%s", imageName, channelSubfix[channel]);
		}

		uchar* density = (uchar*)calloc(width * height, sizeof(uchar));

		readUcharFromFile(density, height, width, channels[0]);
		AmbientRedHost = (float*)calloc(width * height * depth, sizeof(float));
		stackArray<uchar, float>(density, AmbientRedHost, height, width, depth);
		cudaMalloc((void**)&AmbientRedDevice, sizeof(uchar) * width * height * depth);
		cudaMemcpy(AmbientRedDevice, AmbientRedHost, sizeof(uchar) * width * height * depth, cudaMemcpyHostToDevice);

		readUcharFromFile(density, height, width, channels[1]);
		AmbientGreenHost = (float*)calloc(width * height * depth, sizeof(float));
		stackArray<uchar>(density, AmbientGreenHost, height, width, depth);
		cudaMalloc((void**)&AmbientGreenDevice, sizeof(uchar) * width * height * depth);
		cudaMemcpy(AmbientGreenDevice, AmbientGreenHost, sizeof(uchar) * width * height * depth, cudaMemcpyHostToDevice);

		readUcharFromFile(density, height, width, channels[2]);
		AmbientBlueHost = (float*)calloc(width * height * depth, sizeof(float));
		stackArray<uchar>(density, AmbientBlueHost, height, width, depth);
		cudaMalloc((void**)&AmbientBlueDevice, sizeof(uchar) * width * height * depth);
		cudaMemcpy(AmbientBlueDevice, AmbientBlueHost, sizeof(uchar) * width * height * depth, cudaMemcpyHostToDevice);

		free(density);

	}

	uchar* opacity = (uchar*)calloc(width * height, sizeof(uchar));
	setArrayValue<uchar>(opacity, height, width, 1.);
	sceneOpacityHost = (float*)malloc(width * height * depth * sizeof(float));
	stackArray<uchar, float>(opacity, sceneOpacityHost, height, width, depth);
	cudaMalloc((void**)&sceneOpacityDevice, sizeof(float) * width * height * depth);
	cudaMemcpy(sceneOpacityDevice, sceneOpacityHost, sizeof(float) * width * height * depth, cudaMemcpyHostToDevice);
	free(opacity);
	checkGPUStatus("Initial ObjectScene");
	*/
}

inline __device__ int getIndex
(
	int col, int row, int depth,
	const int sceneWidth, const int sceneHeight, const int sceneDepth
)
{
	col = col < sceneWidth ? col : sceneWidth - 1;
	row = row < sceneHeight ? row : sceneHeight - 1;
	depth = depth < sceneDepth ? depth : sceneDepth - 1;
	return depth * sceneWidth * sceneHeight + row * sceneWidth + col;
}

__device__ bool upsample_trilinear(ObjectScene scene, float* baseCoords, float* tri_res,bool lowBandOnly = true)
{
	unsigned int baseIndex[3] =
	{
		(unsigned int)baseCoords[0],
		(unsigned int)baseCoords[1],
		(unsigned int)baseCoords[2]
	};

	unsigned int sceneIndex;
	if (scene.sparse)
	{
		sceneIndex = getIndex(baseIndex[0], baseIndex[1], baseIndex[2], scene.sceneWidth - 1, scene.sceneHeight - 1, scene.sceneDepth - 1);
		if (!scene.occupancyDevice[sceneIndex])
		{
			return false;
		}
	}

	double x = baseCoords[0] - baseIndex[0];
	double y = baseCoords[1] - baseIndex[1];
	double z = baseCoords[2] - baseIndex[2];
	double contributes;

	for (int i = 0; i < scene.sh_dim * 3 + 1; i++)
	{
		tri_res[i] = 0.f;
	}
	int endSH_dim = lowBandOnly?1:scene.sh_dim;

	{
		//010
		sceneIndex = getIndex(baseIndex[0], baseIndex[1], baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = (1. - x) * (1. - y) * (1. - z);
		//contributes = 1.;
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = 0; i < endSH_dim; i++)
		{
			tri_res[1 + i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			tri_res[1 + scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			tri_res[1 + 2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}
		//return;

		//110
		sceneIndex = getIndex(baseIndex[0], baseIndex[1] + 1, baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = (1. - x) * y * (1. - z);
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = 0; i < endSH_dim; i++)
		{
			tri_res[1 + i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			tri_res[1 + scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			tri_res[1 + 2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//000
		sceneIndex = getIndex(baseIndex[0] + 1, baseIndex[1], baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = x * (1. - y) * (1. - z);
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = 0; i < endSH_dim; i++)
		{
			tri_res[1 + i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			tri_res[1 + scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			tri_res[1 + 2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//100
		sceneIndex = getIndex(baseIndex[0] + 1, baseIndex[1] + 1, baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = x * y * (1. - z);
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = 0; i < endSH_dim; i++)
		{
			tri_res[1 + i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			tri_res[1 + scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			tri_res[1 + 2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//011
		sceneIndex = getIndex(baseIndex[0], baseIndex[1], baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = (1. - x) * (1. - y) * z;
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = 0; i < endSH_dim; i++)
		{
			tri_res[1 + i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			tri_res[1 + scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			tri_res[1 + 2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//111
		sceneIndex = getIndex(baseIndex[0], baseIndex[1] + 1, baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = (1. - x) * y * z;
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = 0; i < endSH_dim; i++)
		{
			tri_res[1 + i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			tri_res[1 + scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			tri_res[1 + 2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//001
		sceneIndex = getIndex(baseIndex[0] + 1, baseIndex[1], baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = x * (1. - y) * z;
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = 0; i < endSH_dim; i++)
		{
			tri_res[1 + i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			tri_res[1 + scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			tri_res[1 + 2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//101
		sceneIndex = getIndex(baseIndex[0] + 1, baseIndex[1] + 1, baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = x * y * z;
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = 0; i < endSH_dim; i++)
		{
			tri_res[1 + i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			tri_res[1 + scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			tri_res[1 + 2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}
	}

	return true;
}

__global__ void upsampleScene(ObjectScene src, ObjectScene dest, float threshold, const bool lowBoundOnly = false)
{
	int pixelPos = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	if (pixelPos >= dest.sceneWidth * dest.sceneHeight)return;
	int pixelRow = pixelPos / dest.sceneWidth;
	int pixelCol = pixelPos % dest.sceneWidth;
	unsigned int destSceneIndex;

	float triRes[76];
	float baseCoords[3] =
	{
		pixelCol * 1.f * src.sceneWidth / dest.sceneWidth,
		pixelRow * 1.f * src.sceneHeight / dest.sceneHeight,
		0.f
	};

	bool val;
	for (int depth = 0; depth < dest.sceneDepth; depth++)
	{
		destSceneIndex = getIndex(pixelCol, pixelRow, depth, dest.sceneWidth, dest.sceneHeight, dest.sceneDepth);

		if (dest.sparse && !dest.inCageInfoDevice[destSceneIndex])
		{
			continue;
		}
		
		if(dest.sparse)
		{
			destSceneIndex = dest.indexOffsetsDevice[destSceneIndex];
		}

		baseCoords[2] = depth * 1.f * src.sceneDepth / dest.sceneDepth;

		val = upsample_trilinear(src, baseCoords, triRes,lowBoundOnly);

		if (val)
		{
			if (triRes[0] < threshold)
			{
				dest.sceneOpacityDevice[destSceneIndex] = 0.f;
				// dest.adaptiveLRDevice[destSceneIndex] = 0.;
			}
			else
			{
				dest.sceneOpacityDevice[destSceneIndex] = triRes[0];
			}
			for (int i = 0; i < (lowBoundOnly ? 1 : src.sh_dim); i++)
			{
				dest.AmbientRedDevice[i][destSceneIndex] = triRes[1 + i];
				dest.AmbientGreenDevice[i][destSceneIndex] = triRes[1 + src.sh_dim + i];
				dest.AmbientBlueDevice[i][destSceneIndex] = triRes[1 + src.sh_dim * 2 + i];
			}
		}
		else
		{
			dest.sceneOpacityDevice[destSceneIndex] = 0.f;
			
			for (int i = 0; i < (lowBoundOnly ? 1 : src.sh_dim); i++)
			{
				dest.AmbientRedDevice[i][destSceneIndex] = 0.f;
				dest.AmbientGreenDevice[i][destSceneIndex] = 0.f;
				dest.AmbientBlueDevice[i][destSceneIndex] = 0.f;
			}
		}
	}
}

void objectScene::initSelfByUpsample
(
	objectScene src,
	float threshold,
	const bool lowBandOnly,
	const bool toSparse,
	const int targetSH_dim
)
{
	printf("lowBandOnly = %s\n",lowBandOnly?"True":"False");

	if (!toSparse)
	{
		initSelf
		(
			src.sceneWidth * 2, src.sceneHeight * 2, src.sceneDepth * 2,
			src.grayImage, true,
			0.f, 0.f,
			src.stepSize,
			lowBandOnly?1:targetSH_dim
		);

		dim3 grid((sceneWidth + 15) / 16, (sceneHeight + 15) / 16);
		dim3 block(16, 16);
		
		upsampleScene << <grid, block >> > (src, *this, threshold, src.sh_dim!=targetSH_dim);
		checkGPUStatus("upSampling scene");
		sparse = false;
	}
	else
	{
		if (src.sceneDepth > 128)
		{
			ObjectScene tempScene;
			tempScene.initSelfByUpsample(src, threshold, true, false, 1);
			
			tempScene.pruneSelf(threshold);
			tempScene.getCagedNums();
			tempScene.sparse = true;
			tempScene.copy2HostMemory(1);
			
			initSelf
			(
				tempScene.sceneWidth, tempScene.sceneHeight, tempScene.sceneDepth,
				tempScene.grayImage, false,
				0.f, 0.f,
				tempScene.stepSize,
				lowBandOnly?1:targetSH_dim,
				tempScene.inCageNums
			);

			sparse = true;

			cudaMemcpy(occupancyDevice, tempScene.occupancyDevice, sizeof(uchar) * (tempScene.sceneWidth - 1) * (tempScene.sceneDepth - 1) * (tempScene.sceneHeight - 1), cudaMemcpyDeviceToDevice);

			inCageInfoHost = (uchar*)calloc(sceneWidth * sceneHeight * sceneDepth, sizeof(uchar));
			cudaMalloc((void**)&inCageInfoDevice, sizeof(unsigned char) * sceneWidth * sceneHeight * sceneDepth);
			cudaMemcpy(inCageInfoDevice, tempScene.inCageInfoDevice, sizeof(unsigned char) * sceneWidth * sceneHeight * sceneDepth, cudaMemcpyDeviceToDevice);
			
			cudaMalloc((void**)&indexOffsetsDevice, sizeof(unsigned int) * sceneWidth * sceneHeight * sceneDepth);
			cudaMemcpy(indexOffsetsDevice, tempScene.indexOffsetsDevice, sizeof(unsigned int) * sceneWidth * sceneHeight * sceneDepth, cudaMemcpyDeviceToDevice);
			indexOffsetsHost = (unsigned int*)malloc(sizeof(unsigned int) * sceneWidth * sceneHeight * sceneDepth);
			cudaMemcpy(indexOffsetsHost, indexOffsetsDevice, sizeof(unsigned int) * sceneWidth * sceneHeight * sceneDepth, cudaMemcpyDeviceToHost);
			
			dim3 grid((sceneWidth + 15) / 16, (sceneHeight + 15) / 16);
			dim3 block(16, 16);
			upsampleScene << <grid, block >> > (src, *this, threshold, src.sh_dim==targetSH_dim);
			checkGPUStatus("upSampling from sparse scene to sparse scene");

			src.freeSelf();
			tempScene.freeSelf();
		}
		else
		{
			ObjectScene tempScene;
			tempScene.initSelfByUpsample(src, threshold, lowBandOnly, false,targetSH_dim);

			src.freeSelf();

			tempScene.pruneSelf(threshold);
			tempScene.getCagedNums();
			tempScene.sparse = true;
			tempScene.copy2HostMemory(lowBandOnly ? 1 : targetSH_dim);
			
			initSelf
			(
				tempScene.sceneWidth, tempScene.sceneHeight, tempScene.sceneDepth,
				tempScene.grayImage, false,
				0.f, 0.f,
				tempScene.stepSize,
				lowBandOnly ? 1 : targetSH_dim,
				tempScene.inCageNums
			);

			sparse = true;

			int cageIdx = 0;
			
			for (int idx = 0; idx < sceneWidth * sceneHeight * sceneDepth; idx++)
			{
				if (tempScene.inCageInfoHost[idx])
				{
					printf("transfering %9d/%9d...\r", cageIdx + 1, inCageNums);
					fflush(stdout);
					sceneOpacityHost[cageIdx] = tempScene.sceneOpacityHost[idx];

					for (int i = 0; i < (lowBandOnly ? 1 : targetSH_dim); i++)
					{
						AmbientRedHost[i][cageIdx] = tempScene.AmbientRedHost[i][idx];
						AmbientGreenHost[i][cageIdx] = tempScene.AmbientGreenHost[i][idx];
						AmbientBlueHost[i][cageIdx] = tempScene.AmbientBlueHost[i][idx];
					}
					++cageIdx;

				}
			}
			printf("\n");

			cudaMalloc((void**)&indexOffsetsDevice, sizeof(unsigned int) * sceneWidth * sceneHeight * sceneDepth);
			indexOffsetsHost = (unsigned int*)malloc(sizeof(unsigned int) * sceneWidth * sceneHeight * sceneDepth);

			cudaMemcpy(indexOffsetsDevice, tempScene.indexOffsetsDevice, sizeof(unsigned int) * sceneWidth * sceneHeight * sceneDepth, cudaMemcpyDeviceToDevice);

			inCageInfoHost = (uchar*)calloc(sceneWidth * sceneHeight * sceneDepth, sizeof(uchar));
			cudaMalloc((void**)&inCageInfoDevice, sizeof(unsigned char) * sceneWidth * sceneHeight * sceneDepth);

			cudaMemcpy(inCageInfoDevice, tempScene.inCageInfoDevice, sizeof(unsigned char) * sceneWidth * sceneHeight * sceneDepth, cudaMemcpyDeviceToDevice);

			cudaMemcpy(occupancyDevice, tempScene.occupancyDevice, sizeof(uchar) * (sceneWidth - 1) * (sceneHeight - 1) * (sceneDepth - 1), cudaMemcpyDeviceToDevice);

			tempScene.freeSelf();
			checkGPUStatus("Copy to sparse");

			float** redDevicePtr = (float**)malloc(sizeof(float*) * (lowBandOnly ? 1 : targetSH_dim));
			cudaMemcpy(redDevicePtr, AmbientRedDevice, sizeof(float*) * (lowBandOnly ? 1 : targetSH_dim), cudaMemcpyDeviceToHost);

			float** greenDevicePtr = (float**)malloc(sizeof(float*) * (lowBandOnly ? 1 : targetSH_dim));
			cudaMemcpy(greenDevicePtr, AmbientGreenDevice, sizeof(float*) * (lowBandOnly ? 1 : targetSH_dim), cudaMemcpyDeviceToHost);

			float** blueDevicePtr = (float**)malloc(sizeof(float*) * (lowBandOnly ? 1 : targetSH_dim));
			cudaMemcpy(blueDevicePtr, AmbientBlueDevice, sizeof(float*) * (lowBandOnly ? 1 : targetSH_dim), cudaMemcpyDeviceToHost);

			for (int i = 0; i < (lowBandOnly ? 1 : targetSH_dim); i++)
			{
				cudaMemcpy(redDevicePtr[i], AmbientRedHost[i], sizeof(float) * inCageNums, cudaMemcpyHostToDevice);
				cudaMemcpy(greenDevicePtr[i], AmbientGreenHost[i], sizeof(float) * inCageNums, cudaMemcpyHostToDevice);
				cudaMemcpy(blueDevicePtr[i], AmbientBlueHost[i], sizeof(float) * inCageNums, cudaMemcpyHostToDevice);
			}
			cudaMemcpy(sceneOpacityDevice, sceneOpacityHost, sizeof(float) * inCageNums, cudaMemcpyHostToDevice);
			free(redDevicePtr);
			free(greenDevicePtr);
			free(blueDevicePtr);
		}

		checkGPUStatus("Transfer to sparse", true);
	}

	// bbox_depth_max = (src.bbox_depth_max) * 2;
	// bbox_depth_min = (src.bbox_depth_min) * 2;
	// bbox_height_max = (src.bbox_height_max) * 2;
	// bbox_height_min = (src.bbox_height_min) * 2;
	// bbox_width_max = (src.bbox_width_max) * 2;
	// bbox_width_min = (src.bbox_width_min) * 2;

	// printf("\nBoundingBox is [%4d,%4d] [%4d,%4d] [%4d,%4d]\n",
	// 	bbox_width_min, bbox_width_max,
	// 	bbox_height_min, bbox_height_max,
	// 	bbox_depth_min, bbox_depth_max
	// );

}

void loadChannelSHs
(
	const char* sceneDirectory, const char* channelName,
	float**& channleHostPtr, float**& channelDevicePtr,
	int width, int height, int depth, const int sh_dims,
	const bool sparse = false, const int incageNums = 0
)
{
	char channelPrefix[512], fileName[512];
	sprintf(channelPrefix, "%s%s%s_", sceneDirectory, PATH_CONCATENATOR, channelName);
	//"XXX/red_"

	float** gpuPtr = (float**)malloc(sizeof(float*) * sh_dims);
	cudaMemcpy(gpuPtr, channelDevicePtr, sizeof(float*) * sh_dims, cudaMemcpyDeviceToHost);

	if (!sparse)
	{
		for (int i = 0; i < sh_dims; i++)
		{
			sprintf(fileName, "%s%d.txt", channelPrefix, i);
			readFloatersFromFile(channleHostPtr[i], depth, width * height, fileName);

			cudaMemcpy(gpuPtr[i], channleHostPtr[i], sizeof(float) * width * height * depth, cudaMemcpyHostToDevice);
			printf("%s %d loaded.                \t\t\n", channelName, i);
		}

	}
	else
	{
		for (int i = 0; i < sh_dims; i++)
		{
			sprintf(fileName, "%s%d_sparse.txt", channelPrefix, i);
			readFloatersFromFile(channleHostPtr[i], incageNums, 1, fileName);

			cudaMemcpy(gpuPtr[i], channleHostPtr[i], sizeof(float) * incageNums, cudaMemcpyHostToDevice);
			printf("%s %d loaded.                \t\t\n", channelName, i);
		}

	}

	free(gpuPtr);

}

void objectScene::initFromFile
(
	const char* sceneDirectory,
	int width, int height, int depth,
	bool graylized, float gridStepSize, const int sh_dim, const int targetSH_Dim,
	const bool sparse, const bool haveOccupancyInfo
)
{
	char opacityFileName[256], grayFileName[256];
	char channelSubfix[3][16] = { "red","green","blue" };

	if (!sparse)
	{
		initSelf
		(
			width, height, depth,
			graylized, true,
			0., 0., gridStepSize,
			targetSH_Dim
		);

		sprintf(opacityFileName, "%s%sopacity_band%d.txt", sceneDirectory, PATH_CONCATENATOR, (int)(sqrt(sh_dim) - 1));
		readFloatersFromFile(sceneOpacityHost, depth, width * height, opacityFileName);
		printf("Density Loaded successfully\n");
		cudaMemcpy(sceneOpacityDevice, sceneOpacityHost, sizeof(float) * width * height * depth, cudaMemcpyHostToDevice);
	}
	else
	{
		//InCageNumsInfo.txt
		this->sparse = true;
		char InCageNumsInfoFileName[256];
		// sparse_inCageInfo_band0
		sprintf(InCageNumsInfoFileName, "%s%ssparse_inCageInfo_band%d.txt", sceneDirectory, PATH_CONCATENATOR, (int)(sqrt(sh_dim) - 1));
		FILE* InCageNumsInfoFile = fopen(InCageNumsInfoFileName, "r");
		if (!InCageNumsInfoFile)
		{
			printf("Unable to determine in cage nums: can not find %s\n", InCageNumsInfoFileName);
			exit(-1);
		}
		int incageNums;
		fscanf(InCageNumsInfoFile, "%d", &incageNums);
		initSelf
		(
			width, height, depth,
			graylized, false,
			0., 0., gridStepSize,
			targetSH_Dim, incageNums
		);

		sprintf(opacityFileName, "%s%ssparse_opacity_band%d.txt", sceneDirectory, PATH_CONCATENATOR, (int)(sqrt(sh_dim) - 1));
		readFloatersFromFile(sceneOpacityHost, incageNums, 1, opacityFileName);
		printf("Density Loaded successfully\n");
		cudaMemcpy(sceneOpacityDevice, sceneOpacityHost, sizeof(float) * actualGridNums, cudaMemcpyHostToDevice);

		inCageInfoHost = (uchar*)calloc(sceneWidth * sceneHeight * sceneDepth, sizeof(uchar));
		cudaMalloc((void**)&inCageInfoDevice, sizeof(unsigned char) * sceneWidth * sceneHeight * sceneDepth);
		sprintf(opacityFileName, "%s%ssparse_inCageInfo_band%d.txt", sceneDirectory, PATH_CONCATENATOR, (int)(sqrt(sh_dim) - 1));
		readUcharFromFile(inCageInfoHost, depth, sceneWidth * sceneHeight, opacityFileName);
		cudaMemcpy(inCageInfoDevice, inCageInfoHost, sizeof(uchar) * sceneWidth * sceneHeight * sceneDepth, cudaMemcpyHostToDevice);

		sprintf(opacityFileName, "%s%ssparse_indexOffset.txt", sceneDirectory, PATH_CONCATENATOR);
		indexOffsetsHost = (unsigned int*)malloc(sizeof(unsigned int) * sceneWidth * sceneHeight * sceneDepth);
		cudaMalloc((void**)&indexOffsetsDevice, sizeof(unsigned int) * sceneWidth * sceneHeight * sceneDepth);
		readIntFromFile(indexOffsetsHost, sceneDepth, sceneHeight * sceneWidth, opacityFileName);
		cudaMemcpy(indexOffsetsDevice, indexOffsetsHost, sizeof(unsigned int) * sceneWidth * sceneHeight * sceneDepth, cudaMemcpyHostToDevice);
	}

	if (haveOccupancyInfo)
	{
		sprintf(opacityFileName, "%s%soccupancy.txt", sceneDirectory, PATH_CONCATENATOR);
		readUcharFromFile(occupancyHost, sceneDepth - 1, (sceneHeight - 1) * (sceneWidth - 1), opacityFileName);
		cudaMemcpy(occupancyDevice, occupancyHost, sizeof(uchar) * (sceneDepth - 1) * (sceneHeight - 1) * (sceneWidth - 1), cudaMemcpyHostToDevice);
	}

	if (graylized)
	{
		sprintf(grayFileName, "%s%scolor_gray.txt", sceneDirectory, PATH_CONCATENATOR);
		readFloatersFromFile(AmbientGrayHost, depth, width * height, grayFileName);
		cudaMemcpy(AmbientGrayDevice, AmbientGrayHost, sizeof(float) * width * height * depth, cudaMemcpyHostToDevice);
	}
	else
	{
		loadChannelSHs(sceneDirectory, channelSubfix[0], AmbientRedHost, AmbientRedDevice, width, height, depth, sh_dim, sparse, actualGridNums);
		loadChannelSHs(sceneDirectory, channelSubfix[1], AmbientGreenHost, AmbientGreenDevice, width, height, depth, sh_dim, sparse, actualGridNums);
		loadChannelSHs(sceneDirectory, channelSubfix[2], AmbientBlueHost, AmbientBlueDevice, width, height, depth, sh_dim, sparse, actualGridNums);

	}
}

void objectScene::setSelfSampleOffset(float radius)
{
	randGPU(offsetXDevice, maxSamplePoint, radius, 1);
	randGPU(offsetYDevice, maxSamplePoint, radius, 1);
	randGPU(offsetZDevice, maxSamplePoint, radius, 1);
	checkGPUStatus("Rand sample offset");
}

__device__ bool occupiedVoxel(ObjectScene scene, int* baseCoords, float thresh)
{
	int neighboorIndex;
	int count = 0;
	for (int depth = 0; depth <= 1; depth++)
	{
		for (int height = 0; height <= 1; height++)
		{
			for (int width = 0; width <= 1; width++)
			{
				neighboorIndex = getIndex
				(
					baseCoords[0] + width,
					baseCoords[1] + height,
					baseCoords[2] + depth,
					scene.sceneWidth,
					scene.sceneHeight,
					scene.sceneDepth
				);
				if (scene.sceneOpacityDevice[neighboorIndex] > thresh)
				{
					return true;
					++count;
				}
			}
		}
	}
	return false;
	return count >= 4;
}

__global__ void pruneScene(ObjectScene scene, float opacityThreshold, int* count)
{
	int pixelPos = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	if (pixelPos >= (scene.sceneWidth - 1) * (scene.sceneHeight - 1))return;

	int pixelRow = pixelPos / (scene.sceneWidth - 1);
	int pixelCol = pixelPos % (scene.sceneWidth - 1);

	int sceneIndex;
	count[pixelPos] = 0;
	int baseCoords[3] = { pixelCol,pixelRow,0 };
	for (int depth = 0; depth < scene.sceneDepth - 1; depth++)
	{
		baseCoords[2] = depth;
		sceneIndex = depth * (scene.sceneWidth - 1) * (scene.sceneHeight - 1) + pixelPos;
		if (!occupiedVoxel(scene, baseCoords, opacityThreshold))
		{
			scene.occupancyDevice[sceneIndex] = 0;
			count[pixelPos] += 1;
		}
		else
		{
			scene.occupancyDevice[sceneIndex] = 1;
			sceneIndex = depth * scene.sceneWidth * scene.sceneHeight + pixelRow * scene.sceneWidth + pixelCol;
			scene.inCageInfoDevice[sceneIndex] = 1;

			if (depth == scene.sceneDepth - 2)
			{
				sceneIndex = (scene.sceneDepth - 1) * scene.sceneWidth * scene.sceneHeight + pixelRow * scene.sceneWidth + pixelCol;
				scene.inCageInfoDevice[sceneIndex] = 1;
			}

			if (pixelRow == scene.sceneHeight - 2)
			{
				sceneIndex = depth * scene.sceneWidth * scene.sceneHeight + (scene.sceneHeight - 1) * scene.sceneWidth + pixelCol;
				scene.inCageInfoDevice[sceneIndex] = 1;
			}

			if (pixelCol == scene.sceneWidth - 2)
			{
				sceneIndex = depth * scene.sceneWidth * scene.sceneHeight + scene.sceneHeight * scene.sceneWidth + pixelCol + 1;
				scene.inCageInfoDevice[sceneIndex] = 1;
			}
		}

	}



}

void objectScene::pruneSelf(float opacityThreshold)
{
	printf("Pruning Scene...\n");
	dim3 grid((sceneWidth - 1 + 31) / 32, (sceneHeight - 1 + 31) / 32);
	dim3 block(32, 32);

	int* count = (int*)malloc(sizeof(int) * (sceneWidth - 1) * (sceneHeight - 1)), * countGPU;
	cudaMalloc((void**)&countGPU, sizeof(int) * (sceneWidth - 1) * (sceneHeight - 1));

	inCageInfoHost = (uchar*)calloc(sceneWidth * sceneHeight * sceneDepth, sizeof(uchar));
	cudaMalloc((void**)&inCageInfoDevice, sizeof(unsigned char) * sceneWidth * sceneHeight * sceneDepth);
	cudaMemcpy(inCageInfoDevice, inCageInfoHost, sizeof(unsigned char) * sceneWidth * sceneHeight * sceneDepth, cudaMemcpyHostToDevice);

	pruneScene << <grid, block >> > (*this, opacityThreshold, countGPU);

	cudaMemcpy(count, countGPU, sizeof(int) * (sceneWidth - 1) * (sceneHeight - 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(inCageInfoHost, inCageInfoDevice, sizeof(unsigned char) * sceneWidth * sceneHeight * sceneDepth, cudaMemcpyDeviceToHost);

	cudaFree(countGPU);

	int emptyVoxel = 0;
	for (int i = 0; i < (sceneWidth - 1) * (sceneHeight - 1); i++)
	{
		emptyVoxel += count[i];
	}
	printf("Empty voxel:%d,ratio = %.10f\n", emptyVoxel, emptyVoxel * 1.0 / ((sceneWidth - 1) * (sceneHeight - 1) * (sceneDepth - 1)));

	checkGPUStatus("Prune scene");
	free(count);
	sparse = true;
}

void objectScene::transferSelf2Sparse(float densityThreshold)
{
	pruneSelf(densityThreshold);
	getCagedNums();

	copy2HostMemory(sh_dim);
	sparse = true;
	float* sparseDensityHost = (float*)malloc(sizeof(float) * actualGridNums), * sparseDensityDevice;
	cudaMalloc((void**)&sparseDensityDevice, sizeof(float) * actualGridNums);

	float** sparseRedChannelHost, ** sparseGreenChannelHost, ** sparseBlueChannelHost;
	float** sparseRedChannelDevice, ** sparseGreenChannelDevice, ** sparseBlueChannelDevice;

	initialChannel(sparseRedChannelHost, sparseRedChannelDevice, actualGridNums, sh_dim);
	initialChannel(sparseGreenChannelHost, sparseGreenChannelDevice, actualGridNums, sh_dim);
	initialChannel(sparseBlueChannelHost, sparseBlueChannelDevice, actualGridNums, sh_dim);

	int cageIdx = 0;
	for (int idx = 0; idx < sceneWidth * sceneHeight * sceneDepth; idx++)
	{
		if (inCageInfoHost[idx])
		{
			printf("transfering %9d/%9d...\r", cageIdx + 1, inCageNums);
			fflush(stdout);
			sparseDensityHost[cageIdx] = sceneOpacityHost[idx];

			for (int i = 0; i < (optimizingBand + 1) * (optimizingBand + 1); i++)
			{
				sparseRedChannelHost[i][cageIdx] = AmbientRedHost[i][idx];
				sparseGreenChannelHost[i][cageIdx] = AmbientGreenHost[i][idx];
				sparseBlueChannelHost[i][cageIdx] = AmbientBlueHost[i][idx];
			}
			++cageIdx;
		}
	}
	printf("\n");

	cudaMemcpy(sparseDensityDevice, sparseDensityHost, sizeof(float) * actualGridNums, cudaMemcpyHostToDevice);
	free(sceneOpacityHost);
	sceneOpacityHost = sparseDensityHost;
	cudaFree(sceneOpacityDevice);
	sceneOpacityDevice = sparseDensityDevice;

	float** redDevicePtr = (float**)malloc(sizeof(float*) * sh_dim);
	cudaMemcpy(redDevicePtr, AmbientRedDevice, sizeof(float*) * sh_dim, cudaMemcpyDeviceToHost);
	float** sparseRedDevicePtr = (float**)malloc(sizeof(float*) * sh_dim);
	cudaMemcpy(sparseRedDevicePtr, sparseRedChannelDevice, sizeof(float*) * sh_dim, cudaMemcpyDeviceToHost);

	float** greenDevicePtr = (float**)malloc(sizeof(float*) * sh_dim);
	cudaMemcpy(greenDevicePtr, AmbientGreenDevice, sizeof(float*) * sh_dim, cudaMemcpyDeviceToHost);
	float** sparseGreenDevicePtr = (float**)malloc(sizeof(float*) * sh_dim);
	cudaMemcpy(sparseGreenDevicePtr, sparseGreenChannelDevice, sizeof(float*) * sh_dim, cudaMemcpyDeviceToHost);

	float** blueDevicePtr = (float**)malloc(sizeof(float*) * sh_dim);
	cudaMemcpy(blueDevicePtr, AmbientBlueDevice, sizeof(float*) * sh_dim, cudaMemcpyDeviceToHost);
	float** sparseBlueDevicePtr = (float**)malloc(sizeof(float*) * sh_dim);
	cudaMemcpy(sparseBlueDevicePtr, sparseBlueChannelDevice, sizeof(float*) * sh_dim, cudaMemcpyDeviceToHost);

	totalVariableGPUPtrs[0] = sparseDensityDevice;
	for (int i = 0; i < (optimizingBand + 1) * (optimizingBand + 1); i++)
	{
		free(AmbientRedHost[i]);
		AmbientRedHost[i] = sparseRedChannelHost[i];
		cudaFree(redDevicePtr[i]);
		cudaMemcpy(sparseRedDevicePtr[i], sparseRedChannelHost[i], sizeof(float) * actualGridNums, cudaMemcpyHostToDevice);
		redDevicePtr[i] = sparseRedDevicePtr[i];

		free(AmbientGreenHost[i]);
		AmbientGreenHost[i] = sparseGreenChannelHost[i];
		cudaFree(greenDevicePtr[i]);
		cudaMemcpy(sparseGreenDevicePtr[i], sparseGreenChannelHost[i], sizeof(float) * actualGridNums, cudaMemcpyHostToDevice);
		greenDevicePtr[i] = sparseGreenDevicePtr[i];

		free(AmbientBlueHost[i]);
		AmbientBlueHost[i] = sparseBlueChannelHost[i];
		cudaFree(blueDevicePtr[i]);
		cudaMemcpy(sparseBlueDevicePtr[i], sparseBlueChannelHost[i], sizeof(float) * actualGridNums, cudaMemcpyHostToDevice);
		blueDevicePtr[i] = sparseBlueDevicePtr[i];

		totalVariableGPUPtrs[1 + i] = redDevicePtr[i];
		totalVariableGPUPtrs[1 + sh_dim + i] = greenDevicePtr[i];
		totalVariableGPUPtrs[1 + 2 * sh_dim + i] = blueDevicePtr[i];
	}
	cudaMemcpy(AmbientRedDevice, redDevicePtr, sizeof(float*) * sh_dim, cudaMemcpyHostToDevice);
	free(redDevicePtr);
	free(sparseRedDevicePtr);
	cudaFree(sparseRedChannelDevice);
	cudaMemcpy(AmbientGreenDevice, greenDevicePtr, sizeof(float*) * sh_dim, cudaMemcpyHostToDevice);
	free(greenDevicePtr);
	free(sparseGreenDevicePtr);
	cudaFree(sparseGreenChannelDevice);
	cudaMemcpy(AmbientBlueDevice, blueDevicePtr, sizeof(float*) * sh_dim, cudaMemcpyHostToDevice);
	free(blueDevicePtr);
	free(sparseBlueDevicePtr);
	cudaFree(sparseBlueChannelDevice);

	cudaMemcpy(totalVariableGPUPtrsDevice, totalVariableGPUPtrs, sizeof(float*) * (1 + 3 * sh_dim), cudaMemcpyHostToDevice);
	checkGPUStatus("Transfer to sparse", true);
}

void copyChannel2Host(float** channelGPUPtr, float** channelHostPtr, const int sh_dim, const int paramLen)
{
	float** gpuPtr = (float**)malloc(sizeof(float*) * sh_dim);
	cudaMemcpy(gpuPtr, channelGPUPtr, sizeof(float*) * sh_dim, cudaMemcpyDeviceToHost);
	for (int dimIdx = 0; dimIdx < sh_dim; dimIdx++)
	{
		cudaMemcpy(channelHostPtr[dimIdx], gpuPtr[dimIdx], sizeof(float) * paramLen, cudaMemcpyDeviceToHost);
	}

	free(gpuPtr);
}

void objectScene::copy2HostMemory(const int shDims)
{
	if (sparse)
	{
		cudaMemcpy(occupancyHost, occupancyDevice, sizeof(uchar) * (sceneWidth - 1) * (sceneHeight - 1) * (sceneDepth - 1), cudaMemcpyDeviceToHost);
		cudaMemcpy(indexOffsetsHost, indexOffsetsDevice, sizeof(unsigned int) * sceneWidth * sceneHeight * sceneDepth, cudaMemcpyDeviceToHost);
		cudaMemcpy(inCageInfoHost, inCageInfoDevice, sizeof(uchar) * sceneWidth * sceneHeight * sceneDepth, cudaMemcpyDeviceToHost);
	}


	if (grayImage)
	{
		cudaMemcpy(AmbientGrayHost, AmbientGrayDevice, sizeof(float) * actualGridNums, cudaMemcpyDeviceToHost);
	}
	else
	{
		copyChannel2Host(AmbientRedDevice, AmbientRedHost, shDims, actualGridNums);
		copyChannel2Host(AmbientGreenDevice, AmbientGreenHost, shDims, actualGridNums);
		copyChannel2Host(AmbientBlueDevice, AmbientBlueHost, shDims, actualGridNums);
	}
	cudaMemcpy(sceneOpacityHost, sceneOpacityDevice, sizeof(float) * actualGridNums, cudaMemcpyDeviceToHost);
	checkGPUStatus("Transfer ObjectScene to Main memory");
}

__device__ __host__ float colorsAct(float x)
{
	//return 1. / (1 + exp(-x));
	return 1. / (1 + exp(-0.1 * x));
}

__global__ void getEffectiveCountGPU(ObjectScene scene, int* count, float threshold = 1.5)
{
	int pixelPos = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	if (pixelPos >= scene.sceneWidth * scene.sceneHeight)return;
	int pixelRow = pixelPos / scene.sceneWidth;
	int pixelCol = pixelPos % scene.sceneWidth;
	count[pixelPos] = 0;
	int sceneIndex;

	for (int depth = 0; depth < scene.sceneDepth; depth++)
	{
		sceneIndex = depth * scene.sceneHeight * scene.sceneWidth + pixelRow * scene.sceneWidth + pixelCol;
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];			
		}
		if(sceneIndex&&scene.sceneOpacityDevice[sceneIndex] >= threshold)
		{
			count[pixelPos] += 1;
		}
	}
}

void objectScene::getCagedNums()
{
	printf("Getting Caged grid Nums...\n");
	dim3 grid((sceneWidth / 2 + 31) / 32, (sceneHeight / 2 + 31) / 32);
	dim3 block(32, 32);

	inCageNums = 0;
	for (int i = 0; i < sceneWidth * sceneHeight * sceneDepth; i++)
	{
		inCageNums += inCageInfoHost[i];
	}
	printf("in cage nums = %d\n", inCageNums);
	//actualGridNums = inCageNums;

	cudaMalloc((void**)&indexOffsetsDevice, sizeof(unsigned int) * sceneWidth * sceneHeight * sceneDepth);
	indexOffsetsHost = (unsigned int*)malloc(sizeof(unsigned int) * sceneWidth * sceneHeight * sceneDepth);

	int cagedVoxelIdx = 0, sceneIdx;
	for (int depth = 0; depth < sceneDepth; depth++)
	{
		for (int row = 0; row < sceneHeight; row++)
		{
			for (int col = 0; col < sceneWidth; col++)
			{
				sceneIdx = depth * sceneWidth * sceneHeight + row * sceneWidth + col;
				if (inCageInfoHost[sceneIdx])
				{
					indexOffsetsHost[sceneIdx] = cagedVoxelIdx++;
				}
				else
				{
					indexOffsetsHost[sceneIdx] = 0;
				}
			}
		}
	}
	cudaMemcpy(indexOffsetsDevice, indexOffsetsHost, sizeof(unsigned int) * sceneWidth * sceneHeight * sceneDepth, cudaMemcpyHostToDevice);
	printf("Checking... in cage nums: %d,cagedVoxelIdx:%d\n", inCageNums, cagedVoxelIdx);
	checkGPUStatus("Getting Caged grid Nums", true);
}

void objectScene::getEffectiveCount(objectScene scene, bool densityOnly, float threshold)
{
	cudaDeviceSynchronize();
	dim3 grid((sceneWidth + 31) / 32, (sceneHeight + 31) / 32);
	dim3 block(32, 32);
	int* effectiveCountDevice, * effectiveCountHost = (int*)malloc(sizeof(int) * sceneWidth * sceneHeight);
	cudaMalloc((void**)&effectiveCountDevice, sizeof(int) * sceneWidth * sceneHeight);
	getEffectiveCountGPU << <grid, block >> > (scene, effectiveCountDevice, threshold);
	cudaDeviceSynchronize();
	cudaMemcpy(effectiveCountHost, effectiveCountDevice, sizeof(int) * sceneWidth * sceneHeight, cudaMemcpyDeviceToHost);
	cudaFree(effectiveCountDevice);
	effectiveVoxel = 0;
	for (int row = 0; row < sceneHeight; row++)
	{
		for (int col = 0; col < sceneWidth; col++)
		{
			effectiveVoxel += effectiveCountHost[row * sceneWidth + col];
		}
	}
	free(effectiveCountHost);
	printf("%d\n", effectiveVoxel);
}

void objectScene::saveSelf2ply(const char* fileName, bool densityOnly, float threshold)
{

	FILE* model = fopen(fileName, "w");
	fprintf(model,
		"ply\n\
format ascii 1.0\n\
element vertex %d\n\
property float x\n\
property float y\n\
property float z\n\
property uchar red\n\
property uchar green\n\
property uchar blue\n\
end_header\n", effectiveVoxel);
	copy2HostMemory(1);
	int sceneIndex;
	for (int depth = 0; depth < sceneDepth; depth++)
	{
		for (int row = 0; row < sceneHeight; row++)
		{
			for (int col = 0; col < sceneWidth; col++)
			{
				sceneIndex = depth * sceneWidth * sceneHeight + row * sceneWidth + col;
				if (sparse)sceneIndex = indexOffsetsHost[sceneIndex];
				if (!sceneIndex)continue;
				double density = sceneOpacityHost[sceneIndex];
				int opacityInt = density * 255;
				if (density >= threshold)
				{
					if (densityOnly)
					{
						fprintf(model, "%.2f %.2f %.2f %3d %3d %3d\n",
							col - sceneWidth / 2., row - sceneHeight / 2., depth - sceneDepth / 2.,
							opacityInt, opacityInt, opacityInt
						);
					}
					else if (!grayImage)
					{
						float red = colorsAct(AmbientRedHost[0][sceneIndex]);
						float green = colorsAct(AmbientGreenHost[0][sceneIndex]);
						float blue = colorsAct(AmbientBlueHost[0][sceneIndex]);
						fprintf(model, "%.2f %.2f %.2f %3d %3d %3d\n",
							col - sceneWidth / 2., row - sceneHeight / 2., depth - sceneDepth / 2.,
							(int)(red * 255.f), (int)(green * 255.f), (int)(blue * 255.f)
						);
					}
					else
					{
						//float gray = colorsAct(AmbientGrayHost[sceneIndex]);
						//fprintf(model, "%.2f %.2f %.2f %3d %3d %3d\n",
						//	col - sceneWidth / 2., row - sceneHeight / 2., depth - sceneDepth / 2.,
						//	(int)gray * 255.f, (int)gray * 255.f, (int)gray * 255.f
						//);
					}

				}
			}
		}
		printf("depth %3d passed\r", depth + 1);
		fflush(stdout);
	}
	printf("\n");
	fclose(model);
}

void saveChannelSHs
(
	float** channelPtr,
	const char* destDir, const char* channelName,
	const int sh_dim,const bool hierarchyMode,
	const int width, const int height, const int depth,
	const bool sparse = false, const int actualInCageNums = 0
)
{
	char channelPrefix[512], fileName[512];
	sprintf(channelPrefix, "%s%s%s_", destDir, PATH_CONCATENATOR, channelName);

	int savingBand = (int)sqrt(sh_dim) - 1;
	int savingSHStartIdx = hierarchyMode?savingBand * savingBand:0;
	if (!sparse)
	{
		for (int dimIdx = savingSHStartIdx; dimIdx < sh_dim; dimIdx++)
		{
			sprintf(fileName, "%s%d.txt", channelPrefix, dimIdx);
			saveFloaters2File(channelPtr[dimIdx], depth, width * height, fileName, false, true);
			printf("%s %d saved successfully.\n", channelName, dimIdx);
		}
	}
	else
	{
		for (int dimIdx = savingSHStartIdx; dimIdx < sh_dim; dimIdx++)
		{
			sprintf(fileName, "%s%d_sparse.txt", channelPrefix, dimIdx);
			saveFloaters2File(channelPtr[dimIdx], actualInCageNums, 1, fileName, false, true);
			printf("%s %d saved successfully.\n", channelName, dimIdx);
		}
	}

}

void objectScene::saveSelf(const char* sceneDirectory, bool densityOnly, const int band,const bool hierarchyMode)
{
	// char lrFile[512];
	// sprintf(lrFile,"%s%slr.txt",sceneDirectory,PATH_CONCATENATOR);
	// cudaMemcpy(sceneOpacityHost,adaptiveLRDevice,sizeof(float)*sceneDepth*sceneWidth * sceneHeight,cudaMemcpyDeviceToHost);
	// saveFloaters2File(sceneOpacityHost,sceneDepth, sceneWidth * sceneHeight,lrFile);
	// return;
	char opacityFileName[256], occupancyFileName[256], inCageInfoFileName[256], indexOffsetFileName[256];
	char grayFileName[256];
	char channelSubfix[3][16] = { "red","green","blue" };

	copy2HostMemory((band + 1) * (band + 1));

	sprintf(occupancyFileName, "%s%soccupancy.txt", sceneDirectory, PATH_CONCATENATOR);
	saveUchar2File(occupancyHost, sceneDepth - 1, (sceneWidth - 1) * (sceneHeight - 1), occupancyFileName);

	if (!sparse)
	{
		sprintf(opacityFileName, "%s%sopacity_band%d.txt", sceneDirectory, PATH_CONCATENATOR, band);
		saveFloaters2File(sceneOpacityHost, sceneDepth, sceneWidth * sceneHeight, opacityFileName, false, true);
	}
	else
	{
		sprintf(opacityFileName, "%s%ssparse_opacity_band%d.txt", sceneDirectory, PATH_CONCATENATOR, band);
		saveFloaters2File(sceneOpacityHost, actualGridNums, 1, opacityFileName, false, true);

		sprintf(inCageInfoFileName, "%s%ssparse_inCageInfo_band%d.txt", sceneDirectory, PATH_CONCATENATOR, band);
		saveUchar2File(inCageInfoHost, sceneDepth, sceneWidth * sceneHeight, inCageInfoFileName);

		sprintf(indexOffsetFileName, "%s%ssparse_indexOffset.txt", sceneDirectory, PATH_CONCATENATOR);
		saveInt2File(indexOffsetsHost, sceneDepth, sceneWidth * sceneHeight, indexOffsetFileName);

		char inCageNumsFileName[256];
		sprintf(inCageNumsFileName, "%s%sInCageNumsInfo.txt", sceneDirectory, PATH_CONCATENATOR);
		FILE* incagenumsInfoFILE = fopen(inCageInfoFileName, "w");
		fprintf(incagenumsInfoFILE, "%d", inCageNums);
		fclose(incagenumsInfoFILE);

	}

	printf("Density saved successfully\n");
	if (densityOnly)return;
	if (grayImage)
	{
		sprintf(grayFileName, "%s%scolor_gray.txt", sceneDirectory, PATH_CONCATENATOR);
		saveFloaters2File(AmbientGrayHost, sceneDepth, sceneWidth * sceneHeight, grayFileName, false, true);
	}
	else
	{
		saveChannelSHs(AmbientRedHost, sceneDirectory, channelSubfix[0], (band + 1) * (band + 1), hierarchyMode,sceneWidth, sceneHeight, sceneDepth, sparse, actualGridNums);
		saveChannelSHs(AmbientGreenHost, sceneDirectory, channelSubfix[1], (band + 1) * (band + 1), hierarchyMode,sceneWidth, sceneHeight, sceneDepth, sparse, actualGridNums);
		saveChannelSHs(AmbientBlueHost, sceneDirectory, channelSubfix[2], (band + 1) * (band + 1), hierarchyMode,sceneWidth, sceneHeight, sceneDepth, sparse, actualGridNums);
	}
}

void objectScene::freeSelf()
{
	if (grayImage)
	{
		cudaFree(AmbientGrayDevice);
		cudaFree(DiffuseGrayDevice);
		free(AmbientGrayHost);
		free(DiffuseGrayHost);
	}
	else
	{
		float** redPtr = (float**)malloc(sizeof(float*) * sh_dim);
		float** greenPtr = (float**)malloc(sizeof(float*) * sh_dim);
		float** bluePtr = (float**)malloc(sizeof(float*) * sh_dim);
		cudaMemcpy(redPtr, AmbientRedDevice, sizeof(float*) * sh_dim, cudaMemcpyDeviceToHost);
		cudaMemcpy(greenPtr, AmbientGreenDevice, sizeof(float*) * sh_dim, cudaMemcpyDeviceToHost);
		cudaMemcpy(bluePtr, AmbientBlueDevice, sizeof(float*) * sh_dim, cudaMemcpyDeviceToHost);


		for (int dimIdx = 0; dimIdx < sh_dim; dimIdx++)
		{
			cudaFree(redPtr[dimIdx]);
			cudaFree(greenPtr[dimIdx]);
			cudaFree(bluePtr[dimIdx]);

			free(AmbientRedHost[dimIdx]);
			free(AmbientGreenHost[dimIdx]);
			free(AmbientBlueHost[dimIdx]);
		}
		free(redPtr);
		free(greenPtr);
		free(bluePtr);

		cudaFree(AmbientRedDevice);
		cudaFree(AmbientGreenDevice);
		cudaFree(AmbientBlueDevice);

		free(AmbientRedHost);
		free(AmbientGreenHost);
		free(AmbientBlueHost);
	}
	cudaFree(sceneOpacityDevice);
	free(sceneOpacityHost);
	if (!grayImage)
	{
		free(totalVariableGPUPtrs);
		cudaFree(totalVariableGPUPtrsDevice);
	}

	cudaFree(adaptiveLRDevice);
	cudaFree(occupancyDevice);
	cudaFree(offsetXDevice);
	cudaFree(offsetYDevice);
	cudaFree(offsetZDevice);

	free(occupancyHost);
	if (sparse)
	{
		cudaFree(indexOffsetsDevice);
		free(indexOffsetsHost);

		cudaFree(inCageInfoDevice);
		free(inCageInfoHost);
	}
	checkGPUStatus("Free ObjectScene", true);

}