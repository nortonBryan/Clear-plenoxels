#include "Rays.cuh"

__device__ __host__ inline double minf(double x1, double x2)
{
	return x1 > x2 ? x2 : x1;
}

__device__ __host__ inline double maxf(double x1, double x2)
{
	return x1 > x2 ? x1 : x2;
}

__device__ void normalizePoint(float* point)
{
	double sum = point[0] * point[0] + point[1] * point[1] + point[2] * point[2];
	sum = sqrt(sum);
	point[0] /= sum;
	point[1] /= sum;
	point[2] /= sum;
}

__device__ void getBound
(
	float* rayInfo,
	Camera camera, ObjectScene scene,
	const int pixelRow, const int pixelCol
)
{
	float xInCamera[4] =
	{
		(pixelCol + 0.5 - camera.principleDevice[0]) / camera.alphaDevice[0],
		(pixelRow + 0.5 - camera.principleDevice[1]) / camera.alphaDevice[1],
		1.0f,
		1.0f
	};
	float secondPoint[4];

	for (int row = 0; row < 4; row++)
	{
		secondPoint[row] = 0.;
		for (int col = 0; col < 4; col++)
		{
			secondPoint[row] += camera.c2wExternalMatDevice[row * 4 + col] * xInCamera[col];
		}
		secondPoint[row] -= camera.cameraCenterDevice[row];
	}
	normalizePoint(secondPoint);

	double t[6] =
	{
		(scene.bbox_width_max - camera.cameraCenterDevice[0]) / secondPoint[0],
		(scene.bbox_width_min - camera.cameraCenterDevice[0]) / secondPoint[0],
		(scene.bbox_height_max - camera.cameraCenterDevice[1]) / secondPoint[1],
		(scene.bbox_height_min - camera.cameraCenterDevice[1]) / secondPoint[1],
		(scene.bbox_depth_max - camera.cameraCenterDevice[2]) / secondPoint[2],
		(scene.bbox_depth_min - camera.cameraCenterDevice[2]) / secondPoint[2]
	};

	double t_min = minf(t[0], t[1]);
	double t_max = maxf(t[0], t[1]);

	t_min = maxf(t_min, minf(t[2], t[3]));
	t_max = minf(t_max, maxf(t[2], t[3]));

	t_min = maxf(t_min, minf(t[4], t[5]));
	t_max = minf(t_max, maxf(t[4], t[5]));

	rayInfo[0] = t_min;
	rayInfo[1] = t_max;
	rayInfo[2] = secondPoint[0];
	rayInfo[3] = secondPoint[1];
	rayInfo[4] = secondPoint[2];
	//theta:arccos(z/r),here:arccos(rayInfo[4])
	//phi:arctan(y/x),here:arctan(rayInfo[3]/rayInfo[4])
}

__global__ void initialRays(
	_rays rays, 
	Camera camera, 
	Image groundTruth, 
	ObjectScene scene,
	const int startIndex = 0,
	int* masksIndex = 0,bool depthPriorLearned=false)
{
	int pixelIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (rays.maskRaysOnly && pixelIndex >= groundTruth.masksLen || !rays.maskRaysOnly && pixelIndex >= groundTruth.width * groundTruth.height)return;
	const int rayIndex = startIndex + pixelIndex;

	if (rays.maskRaysOnly)
	{
		pixelIndex = masksIndex[pixelIndex];
	}
	int pixelRow = pixelIndex / groundTruth.width;
	int pixelCol = pixelIndex % groundTruth.width;
	
	float rayInfo[5];//near, far, view_dir_x,view_dir_y,view_dir_z
	getBound(rayInfo, camera, scene, pixelRow, pixelCol);

	{
		rays.cameraCenterX_Device[rayIndex] = camera.cameraCenterDevice[0];
		rays.cameraCenterY_Device[rayIndex] = camera.cameraCenterDevice[1];
		rays.cameraCenterZ_Device[rayIndex] = camera.cameraCenterDevice[2];

		if(depthPriorLearned)
		{
			rays.near_t_Device[rayIndex] = rayInfo[0] + groundTruth.disparityDevice[pixelIndex] * 0.5;
			rays.far_t_Device[rayIndex] = rayInfo[1]-10.;
		}
		else
		{
			rays.near_t_Device[rayIndex] = rayInfo[0];
			rays.far_t_Device[rayIndex] = rayInfo[1];
		}
		

		rays.directionX_Device[rayIndex] = rayInfo[2];
		rays.directionY_Device[rayIndex] = rayInfo[3];
		rays.directionZ_Device[rayIndex] = rayInfo[4];

		rays.gtMask_Device[rayIndex] = groundTruth.maskDevice[pixelIndex];

		if (rays.graylized)
		{
			rays.gtGray_Device[rayIndex] = groundTruth.grayDevice[pixelIndex];
		}
		else
		{
			if (rays.whiteBKGD)
			{
				rays.gtRed_Device[rayIndex] = groundTruth.redDevice[pixelIndex] * groundTruth.maskDevice[pixelIndex]/255.f + 255 - groundTruth.maskDevice[pixelIndex];
				rays.gtGreen_Device[rayIndex] = groundTruth.greenDevice[pixelIndex] * groundTruth.maskDevice[pixelIndex]/255.f + 255 - groundTruth.maskDevice[pixelIndex];
				rays.gtBlue_Device[rayIndex] = groundTruth.blueDevice[pixelIndex] * groundTruth.maskDevice[pixelIndex]/255.f + 255 - groundTruth.maskDevice[pixelIndex];
			}
			else
			{
				rays.gtRed_Device[rayIndex] = groundTruth.redDevice[pixelIndex];// * groundTruth.maskDevice[pixelIndex]/255.f + 255 - groundTruth.maskDevice[pixelIndex];
				rays.gtGreen_Device[rayIndex] = groundTruth.greenDevice[pixelIndex];//* groundTruth.maskDevice[pixelIndex]/255.f + 255 - groundTruth.maskDevice[pixelIndex];
				rays.gtBlue_Device[rayIndex] = groundTruth.blueDevice[pixelIndex];//* groundTruth.maskDevice[pixelIndex]/255.f + 255 - groundTruth.maskDevice[pixelIndex];

			}
		}
	}
	__syncthreads();
}

void _rays::allocateSelfMemory()
{
	assert(totalRays != 0);
	cudaMalloc((void**)&cameraCenterX_Device, sizeof(float) * totalRays);
	cudaMalloc((void**)&cameraCenterY_Device, sizeof(float) * totalRays);
	cudaMalloc((void**)&cameraCenterZ_Device, sizeof(float) * totalRays);

	cudaMalloc((void**)&directionX_Device, sizeof(float) * totalRays);
	cudaMalloc((void**)&directionY_Device, sizeof(float) * totalRays);
	cudaMalloc((void**)&directionZ_Device, sizeof(float) * totalRays);
	cudaMalloc((void**)&near_t_Device, sizeof(float) * totalRays);
	cudaMalloc((void**)&far_t_Device, sizeof(float) * totalRays);

	cudaMalloc((void**)&gtMask_Device, sizeof(uchar) * totalRays);

	cudaMalloc((void**)&accumulatedDensity_Device,sizeof(float)*totalRays);
	cudaMalloc((void**)&accumulatedWeights_Device, sizeof(float) * totalRays);
	
	if(!whiteBKGD)
	{
		printf("no whiteBKGD\n");
	}
	
	if (graylized)
	{
		cudaMalloc((void**)&accumulatedGray_Device, sizeof(float) * totalRays);
		cudaMalloc((void**)&gtGray_Device, sizeof(uchar) * totalRays);
	}
	else
	{
		cudaMalloc((void**)&accumulatedRed_Device, sizeof(float) * totalRays);
		cudaMalloc((void**)&accumulatedGreen_Device, sizeof(float) * totalRays);
		cudaMalloc((void**)&accumulatedBlue_Device, sizeof(float) * totalRays);

		cudaMalloc((void**)&gtRed_Device, sizeof(uchar) * totalRays);
		cudaMalloc((void**)&gtGreen_Device, sizeof(uchar) * totalRays);
		cudaMalloc((void**)&gtBlue_Device, sizeof(uchar) * totalRays);
	}
	checkGPUStatus("Allocate rays memory");
}

void _rays::initialSelf
(
	Camera* cameras, Image* groundTruths, 
	ObjectScene scene,
	const int datasetSize, bool masksRaysOnly,bool depthPriorLearned, bool whiteBKGD, int* order
)
{
	graylized = groundTruths[0].grayImage;
	this->maskRaysOnly = masksRaysOnly;
	this->whiteBKGD = whiteBKGD;

	if (masksRaysOnly)
	{
		totalRays = 0;
		for (int i = 0; i < datasetSize; i++)
		{
			totalRays += groundTruths[i].masksLen;
		}
	}
	else
	{
		totalRays = groundTruths[0].width * groundTruths[0].height * datasetSize;
	}
	
	printf("Initial rays...totalRays = %d,datasetSize = %d\n",totalRays,datasetSize);
	allocateSelfMemory();

	dim3 grid((groundTruths[0].width + 15) / 16, (groundTruths[0].height + 15) / 16);
	dim3 block(16, 16);

	int startIndex = 0;
	int* maskIndexes = 0;
	int reOrderedIndex;
	for (int i = 0; i < datasetSize; i++)
	{
		reOrderedIndex = order[i];
		if (masksRaysOnly)
		{
			cudaMalloc((void**)&maskIndexes, sizeof(int) * groundTruths[reOrderedIndex].masksLen);
			cudaMemcpy(maskIndexes,groundTruths[reOrderedIndex].masksIndex,sizeof(int)*groundTruths[reOrderedIndex].masksLen,cudaMemcpyHostToDevice);
		}
		initialRays << <grid, block >> > (*this, cameras[reOrderedIndex], groundTruths[reOrderedIndex], scene, startIndex, maskIndexes,depthPriorLearned);
		checkGPUStatus("Initial Rays");
		printf("Initial rays %3d/%-3d...\r", i + 1, datasetSize);
		fflush(stdout);
		startIndex += groundTruths[reOrderedIndex].masksLen;
		if (masksRaysOnly)
		{
			cudaFree(maskIndexes);
		}
	}
	printf("\n");
	checkGPUStatus("Initial Rays");
}

__global__ void shuffleToNewGroup(_rays src, _rays dest, int* shuffledIndexArray)
{
	int destRayIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (destRayIndex >= src.totalRays)return;

	dest.cameraCenterX_Device[destRayIndex] = src.cameraCenterX_Device[shuffledIndexArray[destRayIndex]];
	dest.cameraCenterY_Device[destRayIndex] = src.cameraCenterY_Device[shuffledIndexArray[destRayIndex]];
	dest.cameraCenterZ_Device[destRayIndex] = src.cameraCenterZ_Device[shuffledIndexArray[destRayIndex]];
	
	dest.directionX_Device[destRayIndex] = src.directionX_Device[shuffledIndexArray[destRayIndex]];
	dest.directionY_Device[destRayIndex] = src.directionY_Device[shuffledIndexArray[destRayIndex]];
	dest.directionZ_Device[destRayIndex] = src.directionZ_Device[shuffledIndexArray[destRayIndex]];

	dest.near_t_Device[destRayIndex] = src.near_t_Device[shuffledIndexArray[destRayIndex]];
	dest.far_t_Device[destRayIndex] = src.far_t_Device[shuffledIndexArray[destRayIndex]];

	dest.gtMask_Device[destRayIndex] = src.gtMask_Device[shuffledIndexArray[destRayIndex]];
	if (src.graylized)
	{
		dest.gtGray_Device[destRayIndex] = src.gtGray_Device[shuffledIndexArray[destRayIndex]];
	}
	else
	{
		dest.gtRed_Device[destRayIndex] = src.gtRed_Device[shuffledIndexArray[destRayIndex]];
		dest.gtGreen_Device[destRayIndex] = src.gtGreen_Device[shuffledIndexArray[destRayIndex]];
		dest.gtBlue_Device[destRayIndex] = src.gtBlue_Device[shuffledIndexArray[destRayIndex]];
	}
}

void _rays::shuffleTo(_rays* dest, int* shuffledIndexArray)
{
	dest->totalRays = totalRays;
	dest->graylized = graylized;
	dest->maskRaysOnly = maskRaysOnly;
	dest->whiteBKGD = whiteBKGD;
	if(dest->whiteBKGD)
	{
		printf("White background\n");
	}
	dest->allocateSelfMemory();

	dim3 grid((dest->totalRays + 1023) / 1024, 1);
	dim3 block(32, 32);
	shuffleToNewGroup << <grid, block >> > (*this, *dest, shuffledIndexArray);
	checkGPUStatus("Shuffle to dest rays group");
}

void _rays::freeSelf()
{
	cudaFree(cameraCenterX_Device);
	cudaFree(cameraCenterY_Device);
	cudaFree(cameraCenterZ_Device);

	cudaFree(directionX_Device);
	cudaFree(directionY_Device);
	cudaFree(directionZ_Device);

	cudaFree(near_t_Device);
	cudaFree(far_t_Device);

	cudaFree(gtMask_Device);

	cudaFree(accumulatedDensity_Device);

	cudaFree(accumulatedWeights_Device);
	
	
	if (graylized)
	{
		cudaFree(accumulatedGray_Device);
		cudaFree(gtGray_Device);
	}
	else
	{
		cudaFree(accumulatedRed_Device);
		cudaFree(accumulatedGreen_Device);
		cudaFree(accumulatedBlue_Device);

		cudaFree(gtRed_Device);
		cudaFree(gtGreen_Device);
		cudaFree(gtBlue_Device);
	}
	checkGPUStatus("Free rays");
}
