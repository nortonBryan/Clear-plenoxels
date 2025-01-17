#include "RenderUtils.cuh"

inline __device__ int getSceneIndex
(
	int col, int row, int depth,
	const int sceneWidth, const int sceneHeight, const int sceneDepth
)
{
	col = col < 0 ? 0 : (col > sceneWidth - 1 ? sceneWidth - 1 : col);
	row = row < 0 ? 0 : (row > sceneHeight - 1 ? sceneHeight - 1 : row);
	depth = depth < 0 ? 0 : (depth > sceneDepth - 1 ? sceneDepth - 1 : depth);
	return depth * sceneWidth * sceneHeight + row * sceneWidth + col;
}

__device__ __host__ inline double minf(double x1, double x2)
{
	return x1 > x2 ? x2 : x1;
}

__device__ __host__ inline double maxf(double x1, double x2)
{
	return x1 > x2 ? x1 : x2;
}

__device__ void normalizePoint(double* point)
{
	double sum = point[0] * point[0] + point[1] * point[1] + point[2] * point[2];
	sum = sqrt(sum);
	point[0] /= sum;
	point[1] /= sum;
	point[2] /= sum;
}

__device__ void getBound
(
	double* rayInfo,
	Camera camera, ObjectScene scene,
	const int pixelRow, const int pixelCol
)
{
	double xInCamera[4] =
	{
		(pixelCol + 0.5 - camera.principleDevice[0]) / camera.alphaDevice[0],
		(pixelRow + 0.5 - camera.principleDevice[1]) / camera.alphaDevice[1],
		1.0,
		1.0
	};
	double secondPoint[4];

	// get pixel's world coordinate, that is, view direction in world
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

	//if (pixelRow == 200)
	//{
	//	printf("col:%dbounds:[%4d:%-4d][%4d:%-4d][%4d:%-4d]camera:[%.6f,%.6f,%.6f]secondPoint:[%.6f,%.6f,%.6f]%.6f,%.6f,%.6f,%.6f,%.6f,%.6fnear:%.6f,far:%.6f\n", 
	//		pixelCol,scene.bbox_width_min,scene.bbox_width_max,scene.bbox_height_min,scene.bbox_height_max,scene.bbox_depth_min,scene.bbox_depth_max,
	//		camera.cameraCenterDevice[0], camera.cameraCenterDevice[1], camera.cameraCenterDevice[2],
	//		secondPoint[0], secondPoint[1], secondPoint[2],
	//		t[0], t[1], t[2], t[3], t[4], t[5], t_min, t_max);
	//}
}

__device__ void getNeighboorIndex(double* baseCoords, unsigned int* neighboors, ObjectScene scene)
{
	unsigned int baseIndex[3] =
	{
		(unsigned int)baseCoords[0],
		(unsigned int)baseCoords[1],
		(unsigned int)baseCoords[2]
	};

	for (int z = 0; z < 2; z++)
	{
		for (int y = 0; y < 2; y++)
		{
			for (int x = 0; x < 2; x++)
			{
				neighboors[z * 4 + y * 2 + x] = getSceneIndex(baseIndex[0] + x, baseIndex[1] + y, baseIndex[2] + z, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
			}
		}
	}
}

__device__ void projectWorld2Image(double* projectionMatrix, double* world, double* image)
{
	for (int row = 0; row < 3; row++)
	{
		image[row] = 0.;
		for (int col = 0; col < 4; col++)
		{
			image[row] += projectionMatrix[row * 4 + col] * world[col];
		}
	}
	image[0] /= image[2];
	image[1] /= image[2];
}

__global__ void rayTracingCarving(ObjectScene scene, Image image, Camera camera, const int silhouetteRadius = 0)
{
	int renderPixel = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	if (renderPixel >= image.width * image.height)return;

	int pixelRow = renderPixel / image.width;
	int pixelCol = renderPixel % image.width;

	unsigned int imageIndex, sceneIndex;
	int rowCur, colCur;
	for (int row = -silhouetteRadius; row < silhouetteRadius + 1; row++)
	{
		rowCur = row + pixelRow;
		if (rowCur < 0 || rowCur >= image.height)continue;
		for (int col = -silhouetteRadius; col < silhouetteRadius + 1; col++)
		{
			colCur = col + pixelCol;
			if (colCur < 0 || colCur >= image.width)continue;
			imageIndex = rowCur * image.width + colCur;

			if (image.maskDevice[imageIndex])return;

		}
	}
	double rayInfo[5];
	getBound(rayInfo, camera, scene, pixelRow, pixelCol);

	if (rayInfo[0] > rayInfo[1])return;

	double baseCoords[3], t = rayInfo[0];
	while (t < rayInfo[1])
	{
		for (int axisIndex = 0; axisIndex < 3; axisIndex++)
		{
			baseCoords[axisIndex] = camera.cameraCenterDevice[axisIndex] + t * rayInfo[axisIndex + 2];
		}

		sceneIndex = getSceneIndex(baseCoords[0], baseCoords[1], baseCoords[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		scene.adaptiveLRDevice[sceneIndex] = 0.;
		scene.sceneOpacityDevice[sceneIndex] = 0.;
		t += 1.;
	}

}

__global__ void silhouetteCarving(ObjectScene scene, Image image, Camera camera, const int silhouetteRadius = 0)
{
	int renderPixel = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	if (renderPixel >= scene.sceneWidth * scene.sceneHeight)return;

	int pixelRow = renderPixel / scene.sceneWidth;
	int pixelCol = renderPixel % scene.sceneWidth;

	double pointInWorld[4] = { pixelCol + 0.5,pixelRow + 0.5,0.,1. };
	double pointInImage[3], neighborPixel[3];
	int pixelIndex, sceneIndex;
	int imageRow, imageCol;
	bool save = false;
	for (int depth = 0; depth < scene.sceneDepth; depth++)
	{
		pointInWorld[2] = depth;
		save = false;
		projectWorld2Image(camera.projectionMatDevice, pointInWorld, pointInImage);
		imageRow = pointInImage[1];
		imageCol = pointInImage[0];
		sceneIndex = getSceneIndex(pixelCol, pixelRow, depth, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);

		pixelIndex = imageRow * image.width + imageCol;

		for (int neighborRow = -silhouetteRadius; neighborRow < silhouetteRadius + 1; neighborRow++)
		{
			neighborPixel[1] = imageRow + neighborRow;
			for (int neighborCol = -silhouetteRadius; neighborCol < silhouetteRadius + 1; neighborCol++)
			{
				neighborPixel[0] = imageCol + neighborCol;
				if (neighborPixel[0] < 0 || neighborPixel[0] >= image.width ||
					neighborPixel[1] < 0 || neighborPixel[1] >= image.height)continue;

				pixelIndex = (int)(neighborPixel[1] * image.width + neighborPixel[0]);
				if (image.maskDevice[pixelIndex] || image.redDevice[pixelIndex] || image.greenDevice[pixelIndex] || image.blueDevice[pixelIndex])
				{
					save = true;
					break;
				}
			}
			if (save)
			{
				break;
			}
		}

		if (!save)
		{
			scene.sceneOpacityDevice[sceneIndex] = 0.;
			scene.adaptiveLRDevice[sceneIndex] = 0.;
		}
		// else
		// {
		// 	double ratio;
		// 	if (image.grayImage)
		// 	{
		// 		ratio = image.grayDevice[pixelIndex] / 255.;
		// 	}
		// 	else
		// 	{
		// 		ratio = image.redDevice[pixelIndex] / 255.;
		// 		ratio = fmax(ratio,image.greenDevice[pixelIndex]/255.);
		// 		ratio = fmax(ratio,image.blueDevice[pixelIndex] / 255.);

		// 	}
		// 	scene.adaptiveLRDevice[sceneIndex]*=(1. + ratio/100);
		// 	scene.sceneOpacityDevice[sceneIndex]*=(2. - ratio/100);
		// }
	}

}

__global__ void getBoundingBoxZ(int* resMin, int* resMax, int* xoyBound, ObjectScene scene)
{
	int renderPixel = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	if (renderPixel >= scene.sceneWidth * scene.sceneHeight)return;

	int pixelRow = renderPixel / scene.sceneWidth;
	int pixelCol = renderPixel % scene.sceneWidth;
	int sceneIndex;

	int depth = 0;

	sceneIndex = getSceneIndex(pixelCol, pixelRow, depth, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
	while (scene.adaptiveLRDevice[sceneIndex] < 1e-6 && depth + 1 < scene.sceneDepth)
	{
		++depth;
		sceneIndex = getSceneIndex(pixelCol, pixelRow, depth, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
	}
	resMin[renderPixel] = depth;
	resMax[renderPixel] = depth;
	while (depth + 1 < scene.sceneDepth)
	{
		++depth;
		sceneIndex = getSceneIndex(pixelCol, pixelRow, depth, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.adaptiveLRDevice[sceneIndex] > 1e-6)
		{
			resMax[renderPixel] = depth;
		}
	}

	if (resMin[renderPixel] == depth)
	{
		xoyBound[renderPixel] = 0;
		resMin[renderPixel] = -1;
		resMax[renderPixel] = -1;
	}
	else
	{
		xoyBound[renderPixel] = 1;
	}
}

__global__ void getMaxRadius(double* maxRadius, ObjectScene scene)
{
	int renderPixel = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	if (renderPixel >= scene.sceneWidth * scene.sceneHeight)return;

	int pixelRow = renderPixel / scene.sceneWidth;
	int pixelCol = renderPixel % scene.sceneWidth;
	int sceneIndex;

	maxRadius[renderPixel] = 0.;
	double distance = 0.;
	double temp = (pixelRow - scene.sceneHeight / 2.) * (pixelRow - scene.sceneHeight / 2.) + (pixelCol - scene.sceneWidth / 2.) * (pixelCol - scene.sceneWidth / 2.);
	for (int depth = 0; depth < scene.sceneDepth; depth++)
	{
		sceneIndex = getSceneIndex(pixelCol, pixelRow, depth, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.adaptiveLRDevice[sceneIndex] > 1e-6)
		{
			distance = sqrt(temp + (depth - scene.sceneDepth / 2.) * (depth - scene.sceneDepth / 2.));
			if (distance > maxRadius[renderPixel])
			{
				maxRadius[renderPixel] = distance;
			}
		}
	}

}

double RenderUtils::bbox_silhouette
(
	Camera* cameras, Image* groundTruth, ObjectScene* scene,
	const int datasetSize, const int radius, bool isGettingScaleRatio,
	int* skipIndex, int skipNums
)
{
	double max = 0.;
	dim3 grid((scene->sceneWidth + 15) / 16, (scene->sceneHeight + 15) / 16);
	dim3 block(16, 16);

	int skipped = 0;
	for (int imageIndex = 0; imageIndex < datasetSize; imageIndex++)
	{
		if (skipped < skipNums && imageIndex == skipIndex[skipped])
		{
			++skipped;
			continue;
		}
		silhouetteCarving << <grid, block >> > (*scene, groundTruth[imageIndex], cameras[imageIndex], radius);
		cudaDeviceSynchronize();
		printf("Silhouette carving: %3d/%-3d finished\r", imageIndex + 1, datasetSize);
		fflush(stdout);
	}
	printf("\n");
	checkGPUStatus("Carving");

	dim3 grid2((scene->sceneWidth + 15) / 16, (scene->sceneHeight + 15) / 16);
	dim3 block2(16, 16);

	if (isGettingScaleRatio)
	{
		double* maxRadiusGPU, * maxRadiusHost = (double*)malloc(sizeof(double) * scene->sceneWidth * scene->sceneHeight);
		cudaMalloc((void**)&maxRadiusGPU, sizeof(double) * scene->sceneWidth * scene->sceneHeight);
		getMaxRadius << <grid2, block2 >> > (maxRadiusGPU, *scene);
		cudaMemcpy(maxRadiusHost, maxRadiusGPU, sizeof(double) * scene->sceneWidth * scene->sceneHeight, cudaMemcpyDeviceToHost);
		cudaFree(maxRadiusGPU);

		for (int i = 0; i < scene->sceneWidth * scene->sceneHeight; i++)
		{
			if (max < maxRadiusHost[i])max = maxRadiusHost[i];

		}
		free(maxRadiusHost);
	}

	int* minGPU, * minHost = (int*)malloc(sizeof(int) * scene->sceneWidth * scene->sceneHeight);
	cudaMalloc((void**)&minGPU, sizeof(int) * scene->sceneWidth * scene->sceneHeight);
	int* maxGPU, * maxHost = (int*)malloc(sizeof(int) * scene->sceneWidth * scene->sceneHeight);
	cudaMalloc((void**)&maxGPU, sizeof(int) * scene->sceneWidth * scene->sceneHeight);
	int* xoyBoundGPU, * xoyBoundHost = (int*)malloc(sizeof(int) * scene->sceneWidth * scene->sceneHeight);
	cudaMalloc((void**)&xoyBoundGPU, sizeof(int) * scene->sceneWidth * scene->sceneHeight);

	getBoundingBoxZ << <grid2, block2 >> > (minGPU, maxGPU, xoyBoundGPU, *scene);
	cudaMemcpy(minHost, minGPU, sizeof(int) * scene->sceneWidth * scene->sceneHeight, cudaMemcpyDeviceToHost);
	cudaMemcpy(maxHost, maxGPU, sizeof(int) * scene->sceneWidth * scene->sceneHeight, cudaMemcpyDeviceToHost);
	cudaMemcpy(xoyBoundHost, xoyBoundGPU, sizeof(int) * scene->sceneWidth * scene->sceneHeight, cudaMemcpyDeviceToHost);

	checkGPUStatus("Bounding box Z");

	int boundXYZ[3][2] =
	{
		{scene->sceneWidth, -1},
		{scene->sceneHeight, -1},
		{scene->sceneDepth, -1}
	};

	int index;
	for (int row = 0; row < scene->sceneHeight; row++)
	{
		for (int col = 0; col < scene->sceneWidth; col++)
		{
			index = row * scene->sceneWidth + col;
			//find z bound
			if (minHost[index] != -1 && minHost[index] < boundXYZ[2][0])
			{
				boundXYZ[2][0] = minHost[index];
			}

			if (maxHost[index] != -1 && maxHost[index] > boundXYZ[2][1])
			{
				boundXYZ[2][1] = maxHost[index];
			}

			//find x bound
			if (xoyBoundHost[index] && col < boundXYZ[0][0])
			{
				boundXYZ[0][0] = col;
			}
			if (xoyBoundHost[index] && col > boundXYZ[0][1])
			{
				boundXYZ[0][1] = col;
			}

			//find y bound
			if (xoyBoundHost[index] && row < boundXYZ[1][0])
			{
				boundXYZ[1][0] = row;
			}
			if (xoyBoundHost[index] && row > boundXYZ[1][1])
			{
				boundXYZ[1][1] = row;
			}
		}
		printf("Bounding[%4d/%-4d]...\r", row + 1, scene->sceneHeight);
		fflush(stdout);
	}

	cudaFree(minGPU);
	cudaFree(maxGPU);
	cudaFree(xoyBoundGPU);
	free(minHost);
	free(maxHost);
	free(xoyBoundHost);

	// if(isGettingScaleRatio)
	// {
	scene->bbox_width_max = boundXYZ[0][1];
	scene->bbox_width_min = boundXYZ[0][0];

	scene->bbox_height_max = boundXYZ[1][1];
	scene->bbox_height_min = boundXYZ[1][0];

	scene->bbox_depth_max = boundXYZ[2][1];
	scene->bbox_depth_min = boundXYZ[2][0];
	// }
	// else
	// {
	// 	scene->bbox_width_max = std::min(boundXYZ[0][1] + radius, scene->sceneWidth);
	// 	scene->bbox_width_min = std::max(boundXYZ[0][0] - radius, 0);

	// 	scene->bbox_height_max = std::min(boundXYZ[1][1] + radius, scene->sceneHeight);
	// 	scene->bbox_height_min = std::max(boundXYZ[1][0] - radius, 0);

	// 	scene->bbox_depth_max = std::min(boundXYZ[2][1] + radius, scene->sceneDepth);
	// 	scene->bbox_depth_min = std::max(boundXYZ[2][0] - radius, 0);

	// 	// dim3 grid((groundTruth->width+ 15) / 16, (groundTruth->height + 15) / 16);
	// 	// dim3 block(16, 16);
	// 	// for (int imageIndex = 0; imageIndex < datasetSize; imageIndex++)
	// 	// {
	// 	// 	rayTracingCarving << <grid, block >> > (*scene, groundTruth[imageIndex], cameras[imageIndex], radius);
	// 	// 	cudaDeviceSynchronize();
	// 	// 	printf("rayTracing carving: %3d/%-3d finished\r", imageIndex + 1, datasetSize);
	// 	// 	fflush(stdout);
	// 	// }
	// }


	printf("\nBoundingBox is [%4d,%4d] [%4d,%4d] [%4d,%4d]\n",
		scene->bbox_width_min, scene->bbox_width_max,
		scene->bbox_height_min, scene->bbox_height_max,
		scene->bbox_depth_min, scene->bbox_depth_max
	);
	return max / scene->sceneWidth * 2.;
}

__device__ __host__ float sigmoid(float x)
{
	return 1.f / (1.f + exp(-0.1 * x));
}

__device__ __host__ float sigmoid_derivative(float x)
{
	float sigmoidRes = sigmoid(x);
	return 0.1 * sigmoidRes * (1.f - sigmoidRes);
}

__device__ __host__ float density_activation(float x)
{
	return log(1.f + exp(x - 13.f));
	return x;
}

__device__ __host__ float density_derivation(float x)
{
	return 1.f / (1.f + exp(13.f - x));
	return 1.;
}

__device__ __host__ float colors_activation(float x)
{
	// return x>0.?x:0.;
	// return tanh(x);
	return sigmoid(x);
}

__device__ __host__ float colors_derivation(float x)
{
	// return x>0.?1.:0.;
	// return 1. - tanh(x)*tanh(x);
	return sigmoid_derivative(x);
}

__device__ void triDensity(float* tri_res, double* baseCoords, ObjectScene scene)
{
	unsigned int baseIndex[3] =
	{
		(unsigned int)baseCoords[0],
		(unsigned int)baseCoords[1],
		(unsigned int)baseCoords[2]
	};
	double x = baseCoords[0] - baseIndex[0];
	double y = baseCoords[1] - baseIndex[1];
	double z = baseCoords[2] - baseIndex[2];
	double contributes;
	unsigned int sceneIndex;

	tri_res[0] = 0.f;
	{
		//010
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1], baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = (1. - x) * (1. - y) * (1. - z);
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];

		//110
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1] + 1, baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = (1. - x) * y * (1. - z);
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];

		//000
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1], baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = x * (1. - y) * (1. - z);
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];

		//100
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1] + 1, baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = x * y * (1. - z);
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];

		//011
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1], baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = (1. - x) * (1. - y) * z;
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];

		//111
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1] + 1, baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = (1. - x) * y * z;
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];

		//001
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1], baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = x * (1. - y) * z;
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];

		//101
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1] + 1, baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = x * y * z;
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
	}

}

__device__ void trilerpGray(float* tri_res, double* baseCoords, ObjectScene scene)
{
	unsigned int baseIndex[3] =
	{
		(unsigned int)baseCoords[0],
		(unsigned int)baseCoords[1],
		(unsigned int)baseCoords[2]
	};
	double x = baseCoords[0] - baseIndex[0];
	double y = baseCoords[1] - baseIndex[1];
	double z = baseCoords[2] - baseIndex[2];
	double contributes;
	unsigned int sceneIndex;

	tri_res[0] = 0.f;
	tri_res[1] = 0.f;
	{
		//010
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1], baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = (1. - x) * (1. - y) * (1. - z);
		//contributes = 1.;
		tri_res[0] += contributes * density_activation(scene.sceneOpacityDevice[sceneIndex]);
		tri_res[1] += contributes * colors_activation(scene.AmbientGrayDevice[sceneIndex]);
		//return;

		//110
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1] + 1, baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = (1. - x) * y * (1. - z);
		tri_res[0] += contributes * density_activation(scene.sceneOpacityDevice[sceneIndex]);
		tri_res[1] += contributes * colors_activation(scene.AmbientGrayDevice[sceneIndex]);

		//000
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1], baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = x * (1. - y) * (1. - z);
		tri_res[0] += contributes * density_activation(scene.sceneOpacityDevice[sceneIndex]);
		tri_res[1] += contributes * colors_activation(scene.AmbientGrayDevice[sceneIndex]);

		//100
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1] + 1, baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = x * y * (1. - z);
		tri_res[0] += contributes * density_activation(scene.sceneOpacityDevice[sceneIndex]);
		tri_res[1] += contributes * colors_activation(scene.AmbientGrayDevice[sceneIndex]);

		//011
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1], baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = (1. - x) * (1. - y) * z;
		tri_res[0] += contributes * density_activation(scene.sceneOpacityDevice[sceneIndex]);
		tri_res[1] += contributes * colors_activation(scene.AmbientGrayDevice[sceneIndex]);

		//111
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1] + 1, baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = (1. - x) * y * z;
		tri_res[0] += contributes * density_activation(scene.sceneOpacityDevice[sceneIndex]);
		tri_res[1] += contributes * colors_activation(scene.AmbientGrayDevice[sceneIndex]);

		//001
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1], baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = x * (1. - y) * z;
		tri_res[0] += contributes * density_activation(scene.sceneOpacityDevice[sceneIndex]);
		tri_res[1] += contributes * colors_activation(scene.AmbientGrayDevice[sceneIndex]);

		//101
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1] + 1, baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = x * y * z;
		tri_res[0] += contributes * density_activation(scene.sceneOpacityDevice[sceneIndex]);
		tri_res[1] += contributes * colors_activation(scene.AmbientGrayDevice[sceneIndex]);
	}
}

__device__ bool trilerpRGB_Render(float* tri_res, double* baseCoords, float* shs, ObjectScene scene)
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
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1], baseIndex[2], scene.sceneWidth - 1, scene.sceneHeight - 1, scene.sceneDepth - 1);
		if (!scene.occupancyDevice[sceneIndex])
		{
			return false;
		}
	}

	double x = baseCoords[0] - baseIndex[0];
	double y = baseCoords[1] - baseIndex[1];
	double z = baseCoords[2] - baseIndex[2];
	double contributes;


	tri_res[0] = 0.f;
	tri_res[1] = 0.f;
	tri_res[2] = 0.f;
	tri_res[3] = 0.f;

	float sh_tri[75];
	for (int i = 0; i < scene.sh_dim * 3; i++)
	{
		sh_tri[i] = 0.f;
	}

	{
		//010
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1], baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = (1. - x) * (1. - y) * (1. - z);

		//contributes = 1.;
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = 0; i < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); i++)
		{
			sh_tri[i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			sh_tri[scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			sh_tri[2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}
		//return;

		//110
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1] + 1, baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = (1. - x) * y * (1. - z);

		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = 0; i < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); i++)
		{
			sh_tri[i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			sh_tri[scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			sh_tri[2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//000
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1], baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = x * (1. - y) * (1. - z);

		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = 0; i < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); i++)
		{
			sh_tri[i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			sh_tri[scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			sh_tri[2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//100
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1] + 1, baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = x * y * (1. - z);

		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = 0; i < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); i++)
		{
			sh_tri[i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			sh_tri[scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			sh_tri[2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//011
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1], baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = (1. - x) * (1. - y) * z;

		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = 0; i < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); i++)
		{
			sh_tri[i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			sh_tri[scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			sh_tri[2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//111
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1] + 1, baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = (1. - x) * y * z;

		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = 0; i < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); i++)
		{
			sh_tri[i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			sh_tri[scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			sh_tri[2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//001
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1], baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = x * (1. - y) * z;

		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = 0; i < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); i++)
		{
			sh_tri[i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			sh_tri[scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			sh_tri[2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//101
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1] + 1, baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = x * y * z;

		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = 0; i < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); i++)
		{
			sh_tri[i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			sh_tri[scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			sh_tri[2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}
	}

	for (int channelIdx = 0; channelIdx < 3; channelIdx++)
	{
		tri_res[channelIdx + 1] = 0.f;

		for (int dimIdx = 0; dimIdx < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); dimIdx++)
		{
			tri_res[channelIdx + 1] += sh_tri[channelIdx * scene.sh_dim + dimIdx] * shs[dimIdx];
		}

	}

	return true;
}

__device__ bool trilerpRGB(float* tri_res, double* baseCoords, float* shs, ObjectScene scene)
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
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1], baseIndex[2], scene.sceneWidth - 1, scene.sceneHeight - 1, scene.sceneDepth - 1);
		if (!scene.occupancyDevice[sceneIndex])
		{
			return false;
		}
	}
	double x = baseCoords[0] - baseIndex[0];
	double y = baseCoords[1] - baseIndex[1];
	double z = baseCoords[2] - baseIndex[2];
	double contributes;

	tri_res[0] = 0.f;
	tri_res[1] = 0.f;
	tri_res[2] = 0.f;
	tri_res[3] = 0.f;

	float sh_tri[75];
	for (int i = 0; i < scene.sh_dim * 3; i++)
	{
		sh_tri[i] = 0.f;
	}

	int start = 0;//scene.optimizingBand * scene.optimizingBand;

	{
		//010
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1], baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = (1. - x) * (1. - y) * (1. - z);
		//contributes = 1.;
		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = start; i < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); i++)
		{
			sh_tri[i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			sh_tri[scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			sh_tri[2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}
		//return;

		//110
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1] + 1, baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = (1. - x) * y * (1. - z);

		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = start; i < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); i++)
		{
			sh_tri[i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			sh_tri[scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			sh_tri[2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//000
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1], baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = x * (1. - y) * (1. - z);

		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = start; i < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); i++)
		{
			sh_tri[i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			sh_tri[scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			sh_tri[2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//100
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1] + 1, baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = x * y * (1. - z);

		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = start; i < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); i++)
		{
			sh_tri[i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			sh_tri[scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			sh_tri[2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//011
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1], baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = (1. - x) * (1. - y) * z;

		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = start; i < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); i++)
		{
			sh_tri[i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			sh_tri[scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			sh_tri[2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//111
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1] + 1, baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = (1. - x) * y * z;

		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = start; i < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); i++)
		{
			sh_tri[i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			sh_tri[scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			sh_tri[2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//001
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1], baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = x * (1. - y) * z;

		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = start; i < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); i++)
		{
			sh_tri[i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			sh_tri[scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			sh_tri[2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}

		//101
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1] + 1, baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = x * y * z;

		tri_res[0] += contributes * scene.sceneOpacityDevice[sceneIndex];
		for (int i = start; i < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); i++)
		{
			sh_tri[i] += contributes * scene.AmbientRedDevice[i][sceneIndex];
			sh_tri[scene.sh_dim + i] += contributes * scene.AmbientGreenDevice[i][sceneIndex];
			sh_tri[2 * scene.sh_dim + i] += contributes * scene.AmbientBlueDevice[i][sceneIndex];
		}
	}

	for (int channelIdx = 0; channelIdx < 3; channelIdx++)
	{
		tri_res[channelIdx + 1] = 0.f;

		for (int dimIdx = start; dimIdx < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); dimIdx++)
		{
			tri_res[channelIdx + 1] += sh_tri[channelIdx * scene.sh_dim + dimIdx] * shs[dimIdx];
		}

	}

	return true;
}

__device__ void trilerpGradientsGray
(
	float* gradients, double* baseCoords,
	ObjectScene scene,
	Optimizer density,
	Optimizer ambientGray
)
{
	unsigned int baseIndex[3] =
	{
		(unsigned int)baseCoords[0],
		(unsigned int)baseCoords[1],
		(unsigned int)baseCoords[2]
	};
	double x = baseCoords[0] - baseIndex[0];
	double y = baseCoords[1] - baseIndex[1];
	double z = baseCoords[2] - baseIndex[2];
	double contributes;
	unsigned int sceneIndex;
	/*
	{
		//010
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1], baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = (1. - x) * (1. - y) * (1. - z);
		//contributes = 1.;
		density.gradientDevice[sceneIndex] += gradients[0] * contributes * density_derivation(scene.sceneOpacityDevice[sceneIndex]);
		ambientGray.gradientDevice[sceneIndex] += gradients[1] * contributes * colors_derivation(scene.AmbientGrayDevice[sceneIndex]);
		//return;

		//110
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1] + 1, baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = (1. - x) * y * (1. - z);
		density.gradientDevice[sceneIndex] += gradients[0] * contributes * density_derivation(scene.sceneOpacityDevice[sceneIndex]);
		ambientGray.gradientDevice[sceneIndex] += gradients[1] * contributes * colors_derivation(scene.AmbientGrayDevice[sceneIndex]);
		//000
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1], baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = x * (1. - y) * (1. - z);
		density.gradientDevice[sceneIndex] += gradients[0] * contributes * density_derivation(scene.sceneOpacityDevice[sceneIndex]);
		ambientGray.gradientDevice[sceneIndex] += gradients[1] * contributes * colors_derivation(scene.AmbientGrayDevice[sceneIndex]);
		//100
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1] + 1, baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = x * y * (1. - z);
		density.gradientDevice[sceneIndex] += gradients[0] * contributes * density_derivation(scene.sceneOpacityDevice[sceneIndex]);
		ambientGray.gradientDevice[sceneIndex] += gradients[1] * contributes * colors_derivation(scene.AmbientGrayDevice[sceneIndex]);
		//011
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1], baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = (1. - x) * (1. - y) * z;
		density.gradientDevice[sceneIndex] += gradients[0] * contributes * density_derivation(scene.sceneOpacityDevice[sceneIndex]);
		ambientGray.gradientDevice[sceneIndex] += gradients[1] * contributes * colors_derivation(scene.AmbientGrayDevice[sceneIndex]);
		//111
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1] + 1, baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = (1. - x) * y * z;
		density.gradientDevice[sceneIndex] += gradients[0] * contributes * density_derivation(scene.sceneOpacityDevice[sceneIndex]);
		ambientGray.gradientDevice[sceneIndex] += gradients[1] * contributes * colors_derivation(scene.AmbientGrayDevice[sceneIndex]);
		//001
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1], baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = x * (1. - y) * z;
		density.gradientDevice[sceneIndex] += gradients[0] * contributes * density_derivation(scene.sceneOpacityDevice[sceneIndex]);
		ambientGray.gradientDevice[sceneIndex] += gradients[1] * contributes * colors_derivation(scene.AmbientGrayDevice[sceneIndex]);
		//101
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1] + 1, baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		contributes = x * y * z;
		density.gradientDevice[sceneIndex] += gradients[0] * contributes * density_derivation(scene.sceneOpacityDevice[sceneIndex]);
		ambientGray.gradientDevice[sceneIndex] += gradients[1] * contributes * colors_derivation(scene.AmbientGrayDevice[sceneIndex]);
	}
	*/
}

__device__ bool trilerpGradientsRGB
(
	float* gradients, double* baseCoords, float* shs,
	ObjectScene scene,
	Optimizer sceneOptimizer
)
{
	unsigned int baseIndex[3] =
	{
		(unsigned int)baseCoords[0],
		(unsigned int)baseCoords[1],
		(unsigned int)baseCoords[2]
	};
	double x = baseCoords[0] - baseIndex[0];
	double y = baseCoords[1] - baseIndex[1];
	double z = baseCoords[2] - baseIndex[2];
	double contributes;
	unsigned int sceneIndex;
	if (scene.sparse)
	{
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1], baseIndex[2], scene.sceneWidth - 1, scene.sceneHeight - 1, scene.sceneDepth - 1);
		if (!scene.occupancyDevice[sceneIndex])
		{
			return false;
		}
	}
	int start, end, shIdx;
	{
		//010
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1], baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = (1. - x) * (1. - y) * (1. - z);

//contributes = 1.;
		atomicAdd(&sceneOptimizer.gradientDevice[0][sceneIndex],gradients[0] * contributes);
		// sceneOptimizer.gradientDevice[0][sceneIndex] += gradients[0] * contributes;

		for (int channelIdx = 0; channelIdx < 3; channelIdx++)
		{

			start = channelIdx * scene.sh_dim + 1;
			end = start + scene.sh_dim;

			shIdx = 0;
			if (sceneOptimizer.hierarchyStrategy)
			{
				start += scene.optimizingBand * scene.optimizingBand;
				end -= (scene.sh_dim - (scene.optimizingBand + 1) * (scene.optimizingBand + 1));
				shIdx += scene.optimizingBand * scene.optimizingBand;
			}
			for (int dimIdx = start; dimIdx < end; dimIdx++)
			{
				atomicAdd(&sceneOptimizer.gradientDevice[dimIdx][sceneIndex],gradients[1 + channelIdx] * contributes * shs[shIdx]);
				// sceneOptimizer.gradientDevice[dimIdx][sceneIndex] += gradients[1 + channelIdx] * contributes * shs[shIdx];
				++shIdx;
			}
		}
		//return;

		//110
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1] + 1, baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = (1. - x) * y * (1. - z);

		atomicAdd(&sceneOptimizer.gradientDevice[0][sceneIndex],gradients[0] * contributes);
		// sceneOptimizer.gradientDevice[0][sceneIndex] += gradients[0] * contributes;

		for (int channelIdx = 0; channelIdx < 3; channelIdx++)
		{

			start = channelIdx * scene.sh_dim + 1;
			end = start + scene.sh_dim;

			shIdx = 0;
			if (sceneOptimizer.hierarchyStrategy)
			{
				start += scene.optimizingBand * scene.optimizingBand;
				end -= (scene.sh_dim - (scene.optimizingBand + 1) * (scene.optimizingBand + 1));
				shIdx += scene.optimizingBand * scene.optimizingBand;
			}
			for (int dimIdx = start; dimIdx < end; dimIdx++)
			{
				atomicAdd(&sceneOptimizer.gradientDevice[dimIdx][sceneIndex],gradients[1 + channelIdx] * contributes * shs[shIdx]);
				// sceneOptimizer.gradientDevice[dimIdx][sceneIndex] += gradients[1 + channelIdx] * contributes * shs[shIdx];
				++shIdx;
			}
		}

		//000
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1], baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = x * (1. - y) * (1. - z);

//contributes = 1.;
		atomicAdd(&sceneOptimizer.gradientDevice[0][sceneIndex],gradients[0] * contributes);
		// sceneOptimizer.gradientDevice[0][sceneIndex] += gradients[0] * contributes;

		for (int channelIdx = 0; channelIdx < 3; channelIdx++)
		{

			start = channelIdx * scene.sh_dim + 1;
			end = start + scene.sh_dim;

			shIdx = 0;
			if (sceneOptimizer.hierarchyStrategy)
			{
				start += scene.optimizingBand * scene.optimizingBand;
				end -= (scene.sh_dim - (scene.optimizingBand + 1) * (scene.optimizingBand + 1));
				shIdx += scene.optimizingBand * scene.optimizingBand;
			}
			for (int dimIdx = start; dimIdx < end; dimIdx++)
			{
				atomicAdd(&sceneOptimizer.gradientDevice[dimIdx][sceneIndex],gradients[1 + channelIdx] * contributes * shs[shIdx]);
				// sceneOptimizer.gradientDevice[dimIdx][sceneIndex] += gradients[1 + channelIdx] * contributes * shs[shIdx];
				++shIdx;
			}
		}

		//100
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1] + 1, baseIndex[2], scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = x * y * (1. - z);

//contributes = 1.;
		atomicAdd(&sceneOptimizer.gradientDevice[0][sceneIndex],gradients[0] * contributes);
		// sceneOptimizer.gradientDevice[0][sceneIndex] += gradients[0] * contributes;

		for (int channelIdx = 0; channelIdx < 3; channelIdx++)
		{

			start = channelIdx * scene.sh_dim + 1;
			end = start + scene.sh_dim;

			shIdx = 0;
			if (sceneOptimizer.hierarchyStrategy)
			{
				start += scene.optimizingBand * scene.optimizingBand;
				end -= (scene.sh_dim - (scene.optimizingBand + 1) * (scene.optimizingBand + 1));
				shIdx += scene.optimizingBand * scene.optimizingBand;
			}
			for (int dimIdx = start; dimIdx < end; dimIdx++)
			{
				atomicAdd(&sceneOptimizer.gradientDevice[dimIdx][sceneIndex],gradients[1 + channelIdx] * contributes * shs[shIdx]);
				// sceneOptimizer.gradientDevice[dimIdx][sceneIndex] += gradients[1 + channelIdx] * contributes * shs[shIdx];
				++shIdx;
			}
		}

		//011
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1], baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = (1. - x) * (1. - y) * z;

//contributes = 1.;
		atomicAdd(&sceneOptimizer.gradientDevice[0][sceneIndex],gradients[0] * contributes);
		// sceneOptimizer.gradientDevice[0][sceneIndex] += gradients[0] * contributes;

		for (int channelIdx = 0; channelIdx < 3; channelIdx++)
		{

			start = channelIdx * scene.sh_dim + 1;
			end = start + scene.sh_dim;

			shIdx = 0;
			if (sceneOptimizer.hierarchyStrategy)
			{
				start += scene.optimizingBand * scene.optimizingBand;
				end -= (scene.sh_dim - (scene.optimizingBand + 1) * (scene.optimizingBand + 1));
				shIdx += scene.optimizingBand * scene.optimizingBand;
			}
			for (int dimIdx = start; dimIdx < end; dimIdx++)
			{
				atomicAdd(&sceneOptimizer.gradientDevice[dimIdx][sceneIndex],gradients[1 + channelIdx] * contributes * shs[shIdx]);
				// sceneOptimizer.gradientDevice[dimIdx][sceneIndex] += gradients[1 + channelIdx] * contributes * shs[shIdx];
				++shIdx;
			}
		}
		//111
		sceneIndex = getSceneIndex(baseIndex[0], baseIndex[1] + 1, baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = (1. - x) * y * z;

	//contributes = 1.;
		atomicAdd(&sceneOptimizer.gradientDevice[0][sceneIndex],gradients[0] * contributes);
		// sceneOptimizer.gradientDevice[0][sceneIndex] += gradients[0] * contributes;

		for (int channelIdx = 0; channelIdx < 3; channelIdx++)
		{

			start = channelIdx * scene.sh_dim + 1;
			end = start + scene.sh_dim;

			shIdx = 0;
			if (sceneOptimizer.hierarchyStrategy)
			{
				start += scene.optimizingBand * scene.optimizingBand;
				end -= (scene.sh_dim - (scene.optimizingBand + 1) * (scene.optimizingBand + 1));
				shIdx += scene.optimizingBand * scene.optimizingBand;
			}
			for (int dimIdx = start; dimIdx < end; dimIdx++)
			{
				atomicAdd(&sceneOptimizer.gradientDevice[dimIdx][sceneIndex],gradients[1 + channelIdx] * contributes * shs[shIdx]);
				// sceneOptimizer.gradientDevice[dimIdx][sceneIndex] += gradients[1 + channelIdx] * contributes * shs[shIdx];
				++shIdx;
			}
		}
		//001
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1], baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = x * (1. - y) * z;

	//contributes = 1.;
		atomicAdd(&sceneOptimizer.gradientDevice[0][sceneIndex],gradients[0] * contributes);
		// sceneOptimizer.gradientDevice[0][sceneIndex] += gradients[0] * contributes;

		for (int channelIdx = 0; channelIdx < 3; channelIdx++)
		{

			start = channelIdx * scene.sh_dim + 1;
			end = start + scene.sh_dim;

			shIdx = 0;
			if (sceneOptimizer.hierarchyStrategy)
			{
				start += scene.optimizingBand * scene.optimizingBand;
				end -= (scene.sh_dim - (scene.optimizingBand + 1) * (scene.optimizingBand + 1));
				shIdx += scene.optimizingBand * scene.optimizingBand;
			}
			for (int dimIdx = start; dimIdx < end; dimIdx++)
			{
				atomicAdd(&sceneOptimizer.gradientDevice[dimIdx][sceneIndex],gradients[1 + channelIdx] * contributes * shs[shIdx]);
				// sceneOptimizer.gradientDevice[dimIdx][sceneIndex] += gradients[1 + channelIdx] * contributes * shs[shIdx];
				++shIdx;
			}
		}
		//101
		sceneIndex = getSceneIndex(baseIndex[0] + 1, baseIndex[1] + 1, baseIndex[2] + 1, scene.sceneWidth, scene.sceneHeight, scene.sceneDepth);
		if (scene.sparse)
		{
			sceneIndex = scene.indexOffsetsDevice[sceneIndex];
		}
		contributes = x * y * z;

//contributes = 1.;
		atomicAdd(&sceneOptimizer.gradientDevice[0][sceneIndex],gradients[0] * contributes);
		// sceneOptimizer.gradientDevice[0][sceneIndex] += gradients[0] * contributes;

		for (int channelIdx = 0; channelIdx < 3; channelIdx++)
		{

			start = channelIdx * scene.sh_dim + 1;
			end = start + scene.sh_dim;

			shIdx = 0;
			if (sceneOptimizer.hierarchyStrategy)
			{
				start += scene.optimizingBand * scene.optimizingBand;
				end -= (scene.sh_dim - (scene.optimizingBand + 1) * (scene.optimizingBand + 1));
				shIdx += scene.optimizingBand * scene.optimizingBand;
			}
			for (int dimIdx = start; dimIdx < end; dimIdx++)
			{
				atomicAdd(&sceneOptimizer.gradientDevice[dimIdx][sceneIndex],gradients[1 + channelIdx] * contributes * shs[shIdx]);
				// sceneOptimizer.gradientDevice[dimIdx][sceneIndex] += gradients[1 + channelIdx] * contributes * shs[shIdx];
				++shIdx;
			}
		}
	}
	return true;
}

/*
volume rendering
@param rayInfo: t_near,t_far,view_dir_x,view_dir_y,view_dir_z

*/
__device__ void volumeRender_gray
(
	const double* rayInfo,
	Image renderRes, Camera camera, ObjectScene scene,
	const int pixelPos
)
{
	double baseCoords[3];
	double t = rayInfo[0];
	double Ti = 1.0;
	double weights;
	double accumulatedDensity = 0.;
	float triRes[2];
	while (t < rayInfo[1])
	{
		for (int axisIndex = 0; axisIndex < 3; axisIndex++)
		{
			baseCoords[axisIndex] = camera.cameraCenterDevice[axisIndex] + t * rayInfo[axisIndex + 2];
		}

		trilerpGray(triRes, baseCoords, scene);

		//raw formula
		Ti = exp(-accumulatedDensity);
		weights = Ti * (1. - exp(-triRes[0] * scene.stepSize));
		renderRes.disparityDevice[pixelPos] += weights * t;

		//renderRes.accumulatedOpacityDevice[pixelPos] += triRes[0];
		//renderRes.accumulatedGrayDevice[pixelPos] += triRes[1];

		renderRes.grayFloatDevice[pixelPos] += weights * triRes[1];
		accumulatedDensity += triRes[0] * scene.stepSize;

		//Transformed formula
		//weights = Ti * triRes[0];
		//renderRes.disparityDevice[pixelPos] += weights * t;
		//renderRes.accumulatedOpacityDevice[pixelPos] += triRes[0];
		//renderRes.accumulatedGrayDevice[pixelPos] += triRes[1];
		//renderRes.grayFloatDevice[pixelPos] += weights * triRes[1];
		//Ti *= (1. - triRes[0]);
		t += scene.stepSize;
	}
	renderRes.grayDevice[pixelPos] = (int)(renderRes.grayFloatDevice[pixelPos] * 255 + 0.5);
}

__device__ void volumeRender_gray_train(Rays rays, ObjectScene scene, const int raysIndex)
{
	double baseCoords[3];
	double t = rays.near_t_Device[raysIndex];
	double Ti = 1.0;
	double weights;
	double accumulatedDensity = 0.;
	float triRes[2];
	while (t < rays.far_t_Device[raysIndex] - scene.stepSize)
	{
		baseCoords[0] = rays.cameraCenterX_Device[raysIndex] + t * rays.directionX_Device[raysIndex];
		baseCoords[1] = rays.cameraCenterY_Device[raysIndex] + t * rays.directionY_Device[raysIndex];
		baseCoords[2] = rays.cameraCenterZ_Device[raysIndex] + t * rays.directionZ_Device[raysIndex];

		trilerpGray(triRes, baseCoords, scene);

		//raw formula
		Ti = exp(-accumulatedDensity);
		weights = Ti * (1. - exp(-triRes[0] * scene.stepSize));

		rays.accumulatedGray_Device[raysIndex] += weights * triRes[1];
		accumulatedDensity += triRes[0] * scene.stepSize;

		t += scene.stepSize;
	}
}

__device__ void volumeRender_RGB
(
	const double* rayInfo,
	Image renderRes, Camera camera, ObjectScene scene,
	const int pixelPos
)
{
	double baseCoords[3];
	double t = rayInfo[0];
	double Ti = 1.0;
	double weights, acc = 0.;
	double accumulatedDensity = 0.;
	float triRes[4];
	float dir[3] = { rayInfo[2],rayInfo[3],rayInfo[4] };

	float shs[25] =
	{

/*
		// band 0
		0.282095f,

		// band 1
		0.488603f * dir[1],
		0.488603f * dir[2],
		0.488603f * dir[0],


		// band 2
		1.092548f * dir[0] * dir[1],  
		1.092548f * dir[1] * dir[2], 
		0.315392f * (3.f * dir[2] * dir[2] - 1.f), 
		1.092548f * dir[0] * dir[2],
		0.546274f * (dir[0] * dir[0] - dir[1] * dir[1]),

		// band 3
		0.590044f * dir[1] * (3.f * dir[0] * dir[0] - dir[1] * dir[1]), 
		2.890611f * dir[0] * dir[1] * dir[2],
		0.457046f * dir[1] * (5.f * dir[2] * dir[2] - 1.f),
		0.373176f * (5.f * dir[2] * dir[2] * dir[2] - 3.f * dir[2]), 
		0.457046f * dir[0] * (5.f * dir[2] * dir[2] - 1.f),
		1.445306f * (dir[0] * dir[0] - dir[1] * dir[1]) * dir[2],
		0.590044f * dir[0] * (dir[0] * dir[0] - 3.f * dir[1] * dir[1]),

		// band 4
		2.503343f * dir[0]*dir[1]*(dir[0]*dir[0] - dir[1]*dir[1]),
		1.770131f * dir[1] * (3.f * dir[0]*dir[0] - dir[1]*dir[1]) * dir[2],
		0.946175f * dir[0] * dir[1] * (7.f * dir[2]*dir[2] - 1.f),
		0.669047f * dir[1] * (7.f * dir[2]*dir[2]*dir[2] - 3.f * dir[2]),
		0.105786f * (35.f * dir[2]*dir[2]*dir[2]*dir[2] - 30.f * dir[2]*dir[2] + 3.f),
		0.669047f * dir[0] * (7.f * dir[2]*dir[2]*dir[2] - 3.f * dir[2]),
		0.473087f * (dir[0] * dir[0] - dir[1]*dir[1]) * (7.f * dir[2]*dir[2] - 1.),
		1.770131f * dir[0] * (dir[0]*dir[0] - 3.f * dir[1]*dir[1]) * dir[2],
		0.625836f * (dir[0] * dir[0] * (dir[0]*dir[0] - 3.f*dir[1]*dir[1]) - dir[1]*dir[1] * (3.f * dir[0]*dir[0] - dir[1]*dir[1]))

*/
		1.f,

		dir[1],
		dir[2],
		dir[0],

		dir[0] * dir[1],
		dir[1] * dir[2],
		(3.f * dir[2] * dir[2] - 1.f),
		dir[0] * dir[2],
		(dir[0] * dir[0] - dir[1] * dir[1]),

		dir[1] * (3.f * dir[0] * dir[0] - dir[1] * dir[1]),
		dir[0] * dir[1] * dir[2],
		dir[1] * (5.f * dir[2] * dir[2] - 1.f),
		(5.f * dir[2] * dir[2] * dir[2] - 3.f * dir[2]),
		dir[0] * (5.f * dir[2] * dir[2] - 1.f),
		(dir[0] * dir[0] - dir[1] * dir[1]) * dir[2],
		dir[0] * (dir[0] * dir[0] - 3.f * dir[1] * dir[1]),

		dir[0] * dir[1] * (dir[0] * dir[0] - dir[1] * dir[1]),
		dir[1] * (3.f * dir[0] * dir[0] - dir[1] * dir[1]) * dir[2],
		dir[0] * dir[1] * (7.f * dir[2] * dir[2] - 1.f),
		dir[1] * (7.f * dir[2] * dir[2] * dir[2] - 3.f * dir[2]),
		(35.f * dir[2] * dir[2] * dir[2] * dir[2] - 30.f * dir[2] * dir[2] + 3.f),
		dir[0] * (7.f * dir[2] * dir[2] * dir[2] - 3.f * dir[2]),
		(dir[0] * dir[0] - dir[1] * dir[1]) * (7.f * dir[2] * dir[2] - 1.),
		dir[0] * (dir[0] * dir[0] - 3.f * dir[1] * dir[1]) * dir[2],
		(dir[0] * dir[0] * (dir[0] * dir[0] - 3.f * dir[1] * dir[1]) - dir[1] * dir[1] * (3.f * dir[0] * dir[0] - dir[1] * dir[1]))


	};
	
	bool eval;
	while (t < rayInfo[1])
	{
		for (int axisIndex = 0; axisIndex < 3; axisIndex++)
		{
			baseCoords[axisIndex] = camera.cameraCenterDevice[axisIndex] + t * rayInfo[axisIndex + 2];
		}

		eval = trilerpRGB_Render(triRes, baseCoords, shs, scene);
		if (!eval)
		{
			t += scene.stepSize;
			continue;
		}
		triRes[0] = density_activation(triRes[0]);

		triRes[1] = colors_activation(triRes[1]);
		triRes[2] = colors_activation(triRes[2]);
		triRes[3] = colors_activation(triRes[3]);

		Ti = exp(-accumulatedDensity);
		weights = Ti * (1. - exp(-triRes[0] * scene.stepSize));
		acc += weights;

		renderRes.disparityDevice[pixelPos] += weights * t;
		
		renderRes.redFloatDevice[pixelPos] += weights * triRes[1];
		renderRes.greenFloatDevice[pixelPos] += weights * triRes[2];
		renderRes.blueFloatDevice[pixelPos] += weights * triRes[3];
		accumulatedDensity += triRes[0] * scene.stepSize;

		t += abs(scene.stepSize - weights);
		// t+= scene.stepSize;
	}

	// Generate whiteBG render res for uchar file only
	renderRes.redDevice[pixelPos] = (int)((renderRes.redFloatDevice[pixelPos] + 1. - acc) * 255 + 0.5);
	renderRes.greenDevice[pixelPos] = (int)((renderRes.greenFloatDevice[pixelPos] + 1. - acc) * 255 + 0.5);
	renderRes.blueDevice[pixelPos] = (int)((renderRes.blueFloatDevice[pixelPos] + 1. - acc) * 255 + 0.5);

	if (renderRes.whiteBG)
	{
		renderRes.redFloatDevice[pixelPos] += 1. - acc;
		renderRes.greenFloatDevice[pixelPos] += 1. - acc;
		renderRes.blueFloatDevice[pixelPos] += 1. - acc;
	}
	renderRes.maskDevice[pixelPos] = (uchar)(acc * 255);

}

__device__ void volumeRender_RGB_train(Rays rays, ObjectScene scene, const int raysIndex)
{
	double baseCoords[3];
	double t = rays.near_t_Device[raysIndex];
	double Ti = 1.0;
	double weights, acc = 0.;
	double accumulatedDensity = 0.;
	float triRes[4];
	int intT = 0;
	float dir[3] =
	{
		rays.directionX_Device[raysIndex],
		rays.directionY_Device[raysIndex],
		rays.directionZ_Device[raysIndex]
	};

	float shs[25] =
	{

/*
		// band 0
		0.282095f,

		// band 1
		0.488603f * dir[1],
		0.488603f * dir[2],
		0.488603f * dir[0],


		// band 2
		1.092548f * dir[0] * dir[1],  
		1.092548f * dir[1] * dir[2], 
		0.315392f * (3.f * dir[2] * dir[2] - 1.f), 
		1.092548f * dir[0] * dir[2],
		0.546274f * (dir[0] * dir[0] - dir[1] * dir[1]),

		// band 3
		0.590044f * dir[1] * (3.f * dir[0] * dir[0] - dir[1] * dir[1]), 
		2.890611f * dir[0] * dir[1] * dir[2],
		0.457046f * dir[1] * (5.f * dir[2] * dir[2] - 1.f),
		0.373176f * (5.f * dir[2] * dir[2] * dir[2] - 3.f * dir[2]), 
		0.457046f * dir[0] * (5.f * dir[2] * dir[2] - 1.f),
		1.445306f * (dir[0] * dir[0] - dir[1] * dir[1]) * dir[2],
		0.590044f * dir[0] * (dir[0] * dir[0] - 3.f * dir[1] * dir[1]),

		// band 4
		2.503343f * dir[0]*dir[1]*(dir[0]*dir[0] - dir[1]*dir[1]),
		1.770131f * dir[1] * (3.f * dir[0]*dir[0] - dir[1]*dir[1]) * dir[2],
		0.946175f * dir[0] * dir[1] * (7.f * dir[2]*dir[2] - 1.f),
		0.669047f * dir[1] * (7.f * dir[2]*dir[2]*dir[2] - 3.f * dir[2]),
		0.105786f * (35.f * dir[2]*dir[2]*dir[2]*dir[2] - 30.f * dir[2]*dir[2] + 3.f),
		0.669047f * dir[0] * (7.f * dir[2]*dir[2]*dir[2] - 3.f * dir[2]),
		0.473087f * (dir[0] * dir[0] - dir[1]*dir[1]) * (7.f * dir[2]*dir[2] - 1.),
		1.770131f * dir[0] * (dir[0]*dir[0] - 3.f * dir[1]*dir[1]) * dir[2],
		0.625836f * (dir[0] * dir[0] * (dir[0]*dir[0] - 3.f*dir[1]*dir[1]) - dir[1]*dir[1] * (3.f * dir[0]*dir[0] - dir[1]*dir[1]))

*/
		1.f,

		dir[1],
		dir[2],
		dir[0],

		dir[0] * dir[1],
		dir[1] * dir[2],
		(3.f * dir[2] * dir[2] - 1.f),
		dir[0] * dir[2],
		(dir[0] * dir[0] - dir[1] * dir[1]),

		dir[1] * (3.f * dir[0] * dir[0] - dir[1] * dir[1]),
		dir[0] * dir[1] * dir[2],
		dir[1] * (5.f * dir[2] * dir[2] - 1.f),
		(5.f * dir[2] * dir[2] * dir[2] - 3.f * dir[2]),
		dir[0] * (5.f * dir[2] * dir[2] - 1.f),
		(dir[0] * dir[0] - dir[1] * dir[1]) * dir[2],
		dir[0] * (dir[0] * dir[0] - 3.f * dir[1] * dir[1]),

		dir[0] * dir[1] * (dir[0] * dir[0] - dir[1] * dir[1]),
		dir[1] * (3.f * dir[0] * dir[0] - dir[1] * dir[1]) * dir[2],
		dir[0] * dir[1] * (7.f * dir[2] * dir[2] - 1.f),
		dir[1] * (7.f * dir[2] * dir[2] * dir[2] - 3.f * dir[2]),
		(35.f * dir[2] * dir[2] * dir[2] * dir[2] - 30.f * dir[2] * dir[2] + 3.f),
		dir[0] * (7.f * dir[2] * dir[2] * dir[2] - 3.f * dir[2]),
		(dir[0] * dir[0] - dir[1] * dir[1]) * (7.f * dir[2] * dir[2] - 1.),
		dir[0] * (dir[0] * dir[0] - 3.f * dir[1] * dir[1]) * dir[2],
		(dir[0] * dir[0] * (dir[0] * dir[0] - 3.f * dir[1] * dir[1]) - dir[1] * dir[1] * (3.f * dir[0] * dir[0] - dir[1] * dir[1]))


	};
	
	bool eval;

	while (t < rays.far_t_Device[raysIndex])
	{
		baseCoords[0] = rays.cameraCenterX_Device[raysIndex] + t * rays.directionX_Device[raysIndex] + scene.offsetXDevice[intT];
		baseCoords[1] = rays.cameraCenterY_Device[raysIndex] + t * rays.directionY_Device[raysIndex] + scene.offsetYDevice[intT];
		baseCoords[2] = rays.cameraCenterZ_Device[raysIndex] + t * rays.directionZ_Device[raysIndex] + scene.offsetZDevice[intT];

		eval = trilerpRGB(triRes, baseCoords, shs, scene);
		if (!eval)
		{
			t += scene.stepSize;
			++intT;
			continue;
		}

		triRes[0] = density_activation(triRes[0]);
		triRes[1] = colors_activation(triRes[1]);
		triRes[2] = colors_activation(triRes[2]);
		triRes[3] = colors_activation(triRes[3]);

		Ti = exp(-accumulatedDensity);
		weights = Ti * (1. - exp(-triRes[0] * scene.stepSize));
		acc += weights;

		rays.accumulatedDensity_Device[raysIndex] += triRes[0];

		rays.accumulatedRed_Device[raysIndex] += weights * triRes[1];
		rays.accumulatedGreen_Device[raysIndex] += weights * triRes[2];
		rays.accumulatedBlue_Device[raysIndex] += weights * triRes[3];

		accumulatedDensity += triRes[0] * scene.stepSize;
		t += abs(scene.stepSize - weights);
		// t+= scene.stepSize;
		++intT;
	}
	rays.accumulatedWeights_Device[raysIndex] = acc;

	if (rays.whiteBKGD)
	{
		
		rays.accumulatedRed_Device[raysIndex] += (1. - acc);
		rays.accumulatedGreen_Device[raysIndex] += (1. - acc);
		rays.accumulatedBlue_Device[raysIndex] += (1. - acc);
	}

}

__global__ void renderImageGPU(Image renderRes, Camera camera, ObjectScene scene)
{
	int renderPixel = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	if (renderPixel >= renderRes.width * renderRes.height)return;

	int pixelRow = renderPixel / renderRes.width;
	int pixelCol = renderPixel % renderRes.height;

	renderRes.disparityDevice[renderPixel] = 0.;
	renderRes.maskDevice[renderPixel] = 0;

	if (renderRes.grayImage)
	{
		renderRes.grayDevice[renderPixel] = 0;
		renderRes.grayFloatDevice[renderPixel] = 0.;

	}
	else
	{
		renderRes.redFloatDevice[renderPixel] = 0.;
		renderRes.greenFloatDevice[renderPixel] = 0.;
		renderRes.blueFloatDevice[renderPixel] = 0.;

		renderRes.redDevice[renderPixel] = 0;
		renderRes.greenDevice[renderPixel] = 0;
		renderRes.blueDevice[renderPixel] = 0;
	}

	double rayInfo[5];//near, far, view_dir_x,view_dir_y,view_dir_z
	getBound(rayInfo, camera, scene, pixelRow, pixelCol);

	if (rayInfo[0] > rayInfo[1])
	{
		renderRes.disparityDevice[renderPixel] = 0.;
		if (renderRes.whiteBG)
		{
			if (renderRes.grayImage)
			{
				renderRes.grayDevice[renderPixel] = 255;
			}
			else
			{
				renderRes.redDevice[renderPixel] = 255;
				renderRes.greenDevice[renderPixel] = 255;
				renderRes.blueDevice[renderPixel] = 255;

				renderRes.redFloatDevice[renderPixel] = 1.f;
				renderRes.greenFloatDevice[renderPixel] = 1.f;
				renderRes.blueFloatDevice[renderPixel] = 1.f;

			}
		}
		else
		{
			if (renderRes.grayImage)
			{
				renderRes.grayDevice[renderPixel] = 255;
			}
			else
			{
				renderRes.redDevice[renderPixel] = 255;
				renderRes.greenDevice[renderPixel] = 255;
				renderRes.blueDevice[renderPixel] = 255;
			}
		}

		return;
	}

	if (renderRes.grayImage)
	{
		//renderRes.grayDevice[renderPixel] = 255;
		volumeRender_gray(rayInfo, renderRes, camera, scene, renderPixel);
	}
	else
	{
		volumeRender_RGB(rayInfo, renderRes, camera, scene, renderPixel);
	}
	__syncthreads();
}

void RenderUtils::renderImage(Image renderRes, Camera camera, ObjectScene scene)
{
	dim3 grid((renderRes.width + 15) / 16, (renderRes.height + 15) / 16);
	dim3 block(16, 16);


	renderImageGPU << <grid, block >> > (renderRes, camera, scene);
	checkGPUStatus("forward render");
}

__global__ void renderImage_withGradientsGrayGPU
(
	Optimizer density,
	Optimizer ambientGray,
	Image renderRes, Image groundTruth,
	Camera camera, ObjectScene scene, double* losses
)
{
	int renderPixel = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (renderPixel >= renderRes.width * renderRes.height)return;

	int pixelRow = renderPixel / renderRes.width;
	int pixelCol = renderPixel % renderRes.height;

	double grayGradient = renderRes.grayFloatDevice[renderPixel] - groundTruth.grayDevice[renderPixel] / 255.;
	losses[renderPixel] = grayGradient * 255.;

	double rayInfo[5];//near, far, view_dir_x,view_dir_y,view_dir_z
	getBound(rayInfo, camera, scene, pixelRow, pixelCol);
	if (rayInfo[0] > rayInfo[1])return;

	double baseCoords[3];
	double t = rayInfo[0];
	double Ti = 1.0;
	double weights;
	double accumulatedDensity = 0.;
	double accumulatedGray = 0.;
	float triRes[2], triResActivated[2];
	float gradients[2];
	while (t < rayInfo[1])
	{
		for (int axisIndex = 0; axisIndex < 3; axisIndex++)
		{
			baseCoords[axisIndex] = camera.cameraCenterDevice[axisIndex] + t * rayInfo[axisIndex + 2];
		}
		trilerpGray(triRes, baseCoords, scene);

		triResActivated[0] = triRes[0];
		triResActivated[1] = triRes[1];

		Ti = exp(-accumulatedDensity);
		weights = Ti * (1. - exp(-triResActivated[0] * scene.stepSize));

		accumulatedGray += weights * triResActivated[1];
		gradients[1] = grayGradient * weights;
		accumulatedDensity += triResActivated[0] * scene.stepSize;
		Ti = exp(-accumulatedDensity);
		gradients[0] = Ti * triResActivated[1] - (renderRes.grayFloatDevice[renderPixel] - accumulatedGray);
		gradients[0] *= grayGradient * scene.stepSize;

		trilerpGradientsGray(gradients, baseCoords, scene, density, ambientGray);
		t += scene.stepSize;
	}
	__syncthreads();
}

void RenderUtils::renderImage_withGradients_Gray
(
	Optimizer density,
	Optimizer ambientGray,
	Image renderRes, Image groundTruth,
	Camera camera, ObjectScene scene,
	double* losses
)
{
	dim3 grid((renderRes.width + 15) / 16, (renderRes.height + 15) / 16);
	dim3 block(16, 16);

	renderImage_withGradientsGrayGPU << <grid, block >> >
		(
			density, ambientGray,
			renderRes, groundTruth,
			camera, scene, losses
			);
	checkGPUStatus("renderImage_withGradientsGrayGPU");
}

__global__ void renderImage_withGradientsRGB_GPU
(
	Optimizer density,
	Optimizer redChannel,
	Optimizer greenChannel,
	Optimizer blueChannel,
	Image renderRes, Image groundTruth,
	Camera camera, ObjectScene scene,
	double* losses
)
{

}

void RenderUtils::renderImage_withGradients_RGB
(
	Optimizer density,
	Optimizer redChannel,
	Optimizer greenChannel,
	Optimizer blueChannel,
	Image renderRes, Image groundTruth, Camera camera, ObjectScene scene, double* losses
)
{
	dim3 grid((renderRes.width + 15) / 16, (renderRes.height + 15) / 16);
	dim3 block(16, 16);

	renderImage_withGradientsRGB_GPU << <grid, block >> > (density, redChannel, greenChannel, blueChannel, renderRes, groundTruth, camera, scene, losses);
	checkGPUStatus("Get gradients");
}

__global__ void renderRaysGPU(
	Rays rays, ObjectScene scene,
	int* sampleIndex, const int startSampleIndex, const int len)
{
	int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (tid >= len)return;

	const int raysIndex = sampleIndex[tid + startSampleIndex];

	rays.accumulatedDensity_Device[raysIndex] = 0.f;

	rays.accumulatedWeights_Device[raysIndex] = 0.f;

	if (rays.graylized)
	{
		rays.accumulatedGray_Device[raysIndex] = 0.f;
	}
	else
	{
		rays.accumulatedRed_Device[raysIndex] = 0.f;
		rays.accumulatedGreen_Device[raysIndex] = 0.f;
		rays.accumulatedBlue_Device[raysIndex] = 0.f;
	}

	if (rays.near_t_Device[raysIndex] > rays.far_t_Device[raysIndex])
	{
		return;
	}

	if (rays.graylized)
	{
		volumeRender_gray_train(rays, scene, raysIndex);
	}
	else
	{
		volumeRender_RGB_train(rays, scene, raysIndex);
	}

	__syncthreads();

}

void RenderUtils::renderRays(
	Rays rays, ObjectScene scene,
	int* sampleIndex, const int startSampleIndex, const int len
)
{
	dim3 grid((len + 255) / 256, 1);
	dim3 block(16, 16);

	renderRaysGPU << <grid, block >> > (rays, scene, sampleIndex, startSampleIndex, len);
	checkGPUStatus("renderRays");
}

__global__ void renderRays_withGradients_GrayGPU
(
	Optimizer density, Optimizer ambientGray,
	Rays ray, ObjectScene scene,
	double* losses,
	int* sampleIndex, const int startSampleIndex, const int len
)
{
	int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (tid >= len)return;

	const int raysIndex = sampleIndex[tid + startSampleIndex];

	double grayGradient = ray.accumulatedGray_Device[raysIndex] - ray.gtGray_Device[raysIndex] / 255.;

	losses[tid] = grayGradient * 255.;

	if (ray.near_t_Device[raysIndex] > ray.far_t_Device[raysIndex])
	{
		return;
	}

	double baseCoords[3];
	double t = ray.near_t_Device[raysIndex];
	double Ti = 1.0;
	double weights;
	double accumulatedDensity = 0.;
	double accumulatedGray = 0.;
	float triRes[2], triResActivated[2];
	float gradients[2];
	while (t < ray.far_t_Device[raysIndex] - scene.stepSize)
	{
		baseCoords[0] = ray.cameraCenterX_Device[raysIndex] + t * ray.directionX_Device[raysIndex];
		baseCoords[1] = ray.cameraCenterY_Device[raysIndex] + t * ray.directionY_Device[raysIndex];
		baseCoords[2] = ray.cameraCenterZ_Device[raysIndex] + t * ray.directionZ_Device[raysIndex];

		trilerpGray(triRes, baseCoords, scene);

		triResActivated[0] = triRes[0];
		triResActivated[1] = triRes[1];

		Ti = exp(-accumulatedDensity);
		weights = Ti * (1. - exp(-triResActivated[0] * scene.stepSize));

		accumulatedGray += weights * triResActivated[1];
		gradients[1] = grayGradient * weights;
		accumulatedDensity += triResActivated[0] * scene.stepSize;
		Ti = exp(-accumulatedDensity);
		gradients[0] = Ti * triResActivated[1] - (ray.accumulatedGray_Device[raysIndex] - accumulatedGray);
		gradients[0] *= grayGradient * scene.stepSize;

		trilerpGradientsGray(gradients, baseCoords, scene, density, ambientGray);
		t += scene.stepSize;
	}
	__syncthreads();

}

void RenderUtils::renderRays_withGradients_Gray
(
	Optimizer density, Optimizer ambientGray,
	Rays ray, ObjectScene scene,
	double* losses,
	int* sampleIndex, const int startSampleIndex, const int len
)
{
	dim3 grid((len + 255) / 256, 1);
	dim3 block(16, 16);

	renderRays_withGradients_GrayGPU << <grid, block >> >
		(
			density, ambientGray,
			ray, scene,
			losses, sampleIndex, startSampleIndex, len
			);

}

__global__ void renderRays_withGradients_RGBGPU
(
	Optimizer sceneOptimizer,
	Rays ray, ObjectScene scene,
	double* losses,
	int* sampleIndex, const int startSampleIndex, const int len,
	const double maskRayLossCoefficient = 1.e-5,
	const float huberDelta = 1e-1
)
{
	int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (tid >= len)return;

	const int raysIndex = sampleIndex[tid + startSampleIndex];
	float dir[3] = { ray.directionX_Device[raysIndex],ray.directionY_Device[raysIndex],ray.directionZ_Device[raysIndex] };

	float shs[25] =
	{

/*
		// band 0
		0.282095f,

		// band 1
		0.488603f * dir[1],
		0.488603f * dir[2],
		0.488603f * dir[0],


		// band 2
		1.092548f * dir[0] * dir[1],  
		1.092548f * dir[1] * dir[2], 
		0.315392f * (3.f * dir[2] * dir[2] - 1.f), 
		1.092548f * dir[0] * dir[2],
		0.546274f * (dir[0] * dir[0] - dir[1] * dir[1]),

		// band 3
		0.590044f * dir[1] * (3.f * dir[0] * dir[0] - dir[1] * dir[1]), 
		2.890611f * dir[0] * dir[1] * dir[2],
		0.457046f * dir[1] * (5.f * dir[2] * dir[2] - 1.f),
		0.373176f * (5.f * dir[2] * dir[2] * dir[2] - 3.f * dir[2]), 
		0.457046f * dir[0] * (5.f * dir[2] * dir[2] - 1.f),
		1.445306f * (dir[0] * dir[0] - dir[1] * dir[1]) * dir[2],
		0.590044f * dir[0] * (dir[0] * dir[0] - 3.f * dir[1] * dir[1]),

		// band 4
		2.503343f * dir[0]*dir[1]*(dir[0]*dir[0] - dir[1]*dir[1]),
		1.770131f * dir[1] * (3.f * dir[0]*dir[0] - dir[1]*dir[1]) * dir[2],
		0.946175f * dir[0] * dir[1] * (7.f * dir[2]*dir[2] - 1.f),
		0.669047f * dir[1] * (7.f * dir[2]*dir[2]*dir[2] - 3.f * dir[2]),
		0.105786f * (35.f * dir[2]*dir[2]*dir[2]*dir[2] - 30.f * dir[2]*dir[2] + 3.f),
		0.669047f * dir[0] * (7.f * dir[2]*dir[2]*dir[2] - 3.f * dir[2]),
		0.473087f * (dir[0] * dir[0] - dir[1]*dir[1]) * (7.f * dir[2]*dir[2] - 1.),
		1.770131f * dir[0] * (dir[0]*dir[0] - 3.f * dir[1]*dir[1]) * dir[2],
		0.625836f * (dir[0] * dir[0] * (dir[0]*dir[0] - 3.f*dir[1]*dir[1]) - dir[1]*dir[1] * (3.f * dir[0]*dir[0] - dir[1]*dir[1]))

*/
		1.f,

		dir[1],
		dir[2],
		dir[0],

		dir[0] * dir[1],
		dir[1] * dir[2],
		(3.f * dir[2] * dir[2] - 1.f),
		dir[0] * dir[2],
		(dir[0] * dir[0] - dir[1] * dir[1]),

		dir[1] * (3.f * dir[0] * dir[0] - dir[1] * dir[1]),
		dir[0] * dir[1] * dir[2],
		dir[1] * (5.f * dir[2] * dir[2] - 1.f),
		(5.f * dir[2] * dir[2] * dir[2] - 3.f * dir[2]),
		dir[0] * (5.f * dir[2] * dir[2] - 1.f),
		(dir[0] * dir[0] - dir[1] * dir[1]) * dir[2],
		dir[0] * (dir[0] * dir[0] - 3.f * dir[1] * dir[1]),

		dir[0] * dir[1] * (dir[0] * dir[0] - dir[1] * dir[1]),
		dir[1] * (3.f * dir[0] * dir[0] - dir[1] * dir[1]) * dir[2],
		dir[0] * dir[1] * (7.f * dir[2] * dir[2] - 1.f),
		dir[1] * (7.f * dir[2] * dir[2] * dir[2] - 3.f * dir[2]),
		(35.f * dir[2] * dir[2] * dir[2] * dir[2] - 30.f * dir[2] * dir[2] + 3.f),
		dir[0] * (7.f * dir[2] * dir[2] * dir[2] - 3.f * dir[2]),
		(dir[0] * dir[0] - dir[1] * dir[1]) * (7.f * dir[2] * dir[2] - 1.),
		dir[0] * (dir[0] * dir[0] - 3.f * dir[1] * dir[1]) * dir[2],
		(dir[0] * dir[0] * (dir[0] * dir[0] - 3.f * dir[1] * dir[1]) - dir[1] * dir[1] * (3.f * dir[0] * dir[0] - dir[1] * dir[1]))


	};



	if (ray.near_t_Device[raysIndex] > ray.far_t_Device[raysIndex])
	{
		losses[tid] = 0.;
		return;
	}

	double rgbLosses[3] =
	{
		ray.accumulatedRed_Device[raysIndex] - ray.gtRed_Device[raysIndex] / 255.,
		ray.accumulatedGreen_Device[raysIndex] - ray.gtGreen_Device[raysIndex] / 255.,
		ray.accumulatedBlue_Device[raysIndex] - ray.gtBlue_Device[raysIndex] / 255.
	};

	losses[tid] = (abs(rgbLosses[0]) + abs(rgbLosses[1]) + abs(rgbLosses[2])) * 255. / 3.;

	if (isnan(losses[tid]))
	{
		losses[tid] = 255.;
		return;
	}

	for(int i = 0;i<3;i++)
	{
		if(rgbLosses[i]>huberDelta)
		{
			rgbLosses[i] = huberDelta - 0.5 * huberDelta * huberDelta;
		}
		else if(rgbLosses[i]<-huberDelta)
		{
			rgbLosses[i] = 0.5 * huberDelta * huberDelta - huberDelta;
		}
	}

	double baseCoords[3];
	double t = ray.near_t_Device[raysIndex];
	double Ti = 1.0;
	double weights;
	double accumulatedDensity = 0., accumulatedWeights = 0.;
	double accumulatedRGB[3] = { 0., 0., 0. };
	float triRes[4], triResActivated[4];
	float gradients[4];
	float maskLoss = 0.;

	if(ray.gtMask_Device[raysIndex]&&ray.accumulatedWeights_Device[raysIndex]<1.)
	{
		maskLoss = (ray.gtMask_Device[raysIndex] / 255.0 - ray.accumulatedWeights_Device[raysIndex])*maskRayLossCoefficient ;
		// maskLoss = (1. - ray.accumulatedWeights_Device[raysIndex]) * maskRayLossCoefficient;
	}
	float weightsPre = maxf(1e-6,ray.accumulatedWeights_Device[raysIndex]);
	weightsPre = minf(1-1e-6,weightsPre);
	maskLoss += (log(weightsPre)-log(1.-weightsPre))*1e-3;
	int intT = 0;
	int totalStep = (ray.far_t_Device[raysIndex] - t) / scene.stepSize;
	bool eval;

	while (t < ray.far_t_Device[raysIndex])
	{
		baseCoords[0] = ray.cameraCenterX_Device[raysIndex] + t * ray.directionX_Device[raysIndex] + scene.offsetXDevice[intT];
		baseCoords[1] = ray.cameraCenterY_Device[raysIndex] + t * ray.directionY_Device[raysIndex] + scene.offsetYDevice[intT];
		baseCoords[2] = ray.cameraCenterZ_Device[raysIndex] + t * ray.directionZ_Device[raysIndex] + scene.offsetZDevice[intT];

		eval = trilerpRGB(triRes, baseCoords, shs, scene);
		if (!eval)
		{
			t += scene.stepSize;
			++intT;
			continue;
		}

		triResActivated[0] = density_activation(triRes[0]);

		triResActivated[1] = colors_activation(triRes[1]);
		triResActivated[2] = colors_activation(triRes[2]);
		triResActivated[3] = colors_activation(triRes[3]);
		//triResActivated[1] = triRes[1];
		//triResActivated[2] = triRes[2];
		//triResActivated[3] = triRes[3];

		Ti = exp(-accumulatedDensity);
		weights = Ti * (1. - exp(-triResActivated[0] * scene.stepSize));
		accumulatedWeights += weights;

		accumulatedRGB[0] += weights * triResActivated[1];
		accumulatedRGB[1] += weights * triResActivated[2];
		accumulatedRGB[2] += weights * triResActivated[3];

		gradients[1] = 0.6667f * weights * rgbLosses[0] * colors_derivation(triRes[1]);
		gradients[2] = 0.6667f * weights * rgbLosses[1] * colors_derivation(triRes[2]);
		gradients[3] = 0.6667f * weights * rgbLosses[2] * colors_derivation(triRes[3]);

		accumulatedDensity += triResActivated[0] * scene.stepSize;
		Ti = exp(-accumulatedDensity);

		gradients[0] = rgbLosses[0] * (Ti * triResActivated[1] - (ray.accumulatedRed_Device[raysIndex] - accumulatedRGB[0]));
		gradients[0] += rgbLosses[1] * (Ti * triResActivated[2] - (ray.accumulatedGreen_Device[raysIndex] - accumulatedRGB[1]));
		gradients[0] += rgbLosses[2] * (Ti * triResActivated[3] - (ray.accumulatedBlue_Device[raysIndex] - accumulatedRGB[2]));

		gradients[0] *= 0.6667 * scene.stepSize * density_derivation(triRes[0]);// * cos(ray.directionZ_Device[raysIndex]);

		// if (ray.whiteBKGD)
		// {
		// 	
		// }

		// gradients[0] += 0.000001 * ray.accumulatedDensity_Device[raysIndex]/totalStep;
		gradients[0] += scene.stepSize * density_derivation(triRes[0]) * (ray.accumulatedWeights_Device[raysIndex] - accumulatedWeights - Ti) * maskLoss;

		trilerpGradientsRGB(gradients, baseCoords, shs, scene, sceneOptimizer);
		t += abs(scene.stepSize - weights);
		// t+= scene.stepSize;
		++intT;
	}
	__syncthreads();
}

void RenderUtils::renderRays_withGradients_RGB
(
	Optimizer sceneOptimizer,
	Rays ray, ObjectScene scene,
	double* losses,
	int* sampleIndex, const int startSampleIndex, const int len,
	const double maskRayLossCoefficient,
	const float huberDelta
)
{
	dim3 grid((len + 255) / 256, 1);
	dim3 block(16, 16);

	renderRays_withGradients_RGBGPU << <grid, block >> >
		(
			sceneOptimizer,
			ray, scene,
			losses, sampleIndex, startSampleIndex, len, maskRayLossCoefficient,huberDelta
			);
	checkGPUStatus("renderRays_withGradients_RGBGPU ");
}

__global__ void bounddRays(Rays ray, ObjectScene scene, int* sampleIndex, const int startSampleIndex, const int len, const double threshold)
{
	int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (tid >= len)return;

}

void RenderUtils::boundRays(Rays rays, ObjectScene scene, int* sampleIndex, const int startSampleIndex, const int len, const double threshold)
{
	dim3 grid((len + 255) / 256, 1);
	dim3 block(16, 16);

}

__global__ void gatherDensitiesGPU(Camera camera, ObjectScene scene, float* densities, double* rayInfo, const int len)
{
	//int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	//if (tid >= len)return;

	//double baseCoords[3];
	//for (int axisIndex = 0; axisIndex < 3; axisIndex++)
	//{
	//	baseCoords[axisIndex] = camera.cameraCenterDevice[axisIndex] + (rayInfo[0] + tid * scene.stepSize) * rayInfo[axisIndex + 2];
	//}
	//float triRes[4];
	//trilerpRGB(triRes, baseCoords, scene);
	//densities[tid] = density_activation(triRes[0]);
	//__syncthreads();
}

void getBoundHost(Camera camera, ObjectScene scene, double* rayInfo)
{
	double xInCamera[4] =
	{
		(400 - camera.principle[0]) / camera.alpha[0],
		(250 - camera.principle[1]) / camera.alpha[1],
		1.0,
		1.0
	};
	double secondPoint[4];

	for (int row = 0; row < 4; row++)
	{
		secondPoint[row] = 0.;
		for (int col = 0; col < 4; col++)
		{
			secondPoint[row] += camera.c2wExternalMat[row * 4 + col] * xInCamera[col];
		}
		secondPoint[row] -= camera.cameraCenterHost[row];
	}
	normalize(secondPoint, 3);

	double t[6] =
	{
		(scene.bbox_width_max - camera.cameraCenterHost[0]) / secondPoint[0],
		(scene.bbox_width_min - camera.cameraCenterHost[0]) / secondPoint[0],
		(scene.bbox_height_max - camera.cameraCenterHost[1]) / secondPoint[1],
		(scene.bbox_height_min - camera.cameraCenterHost[1]) / secondPoint[1],
		(scene.bbox_depth_max - camera.cameraCenterHost[2]) / secondPoint[2],
		(scene.bbox_depth_min - camera.cameraCenterHost[2]) / secondPoint[2]
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

}

float* RenderUtils::gatherDensitiesAlongRay(Camera camera, ObjectScene scene, int* effectivePts)
{
	double rayInfo[5], * rayInfoDevice;
	cudaMalloc((void**)&rayInfoDevice, sizeof(double) * 5);
	getBoundHost(camera, scene, rayInfo);
	cudaMemcpy(rayInfoDevice, rayInfo, sizeof(double) * 5, cudaMemcpyHostToDevice);
	*effectivePts = (rayInfo[1] - rayInfo[0]) / scene.stepSize;
	float* densities = (float*)malloc(sizeof(float) * *effectivePts);
	float* densitiesDevice;
	cudaMalloc((void**)&densitiesDevice, sizeof(float) * *effectivePts);

	dim3 grid((*effectivePts + 255) / 256, 1);
	dim3 block(16, 16);

	gatherDensitiesGPU << <grid, block >> > (camera, scene, densitiesDevice, rayInfoDevice, *effectivePts);
	checkGPUStatus("Gather rayInfo");

	cudaMemcpy(densities, densitiesDevice, sizeof(float) * *effectivePts, cudaMemcpyDeviceToHost);
	cudaFree(densitiesDevice);
	return densities;
}

__global__ void pruneSceneByWeightsGPU(Rays ray, ObjectScene scene, const double threshold)
{
	//int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	//if (tid >= ray.totalRays)return;

	//const int raysIndex = tid;

	//if (ray.near_t_Device[raysIndex] > ray.far_t_Device[raysIndex])
	//{
	//	return;
	//}
	//double baseCoords[3];
	//unsigned int neighboors[8];
	//double t = ray.near_t_Device[raysIndex];
	//double Ti = 1.0;
	//double weights;
	//double accumulatedDensity = 0.;
	//double accumulatedRGB[3] = { 0., 0., 0. };
	//float triRes[4], triResActivated[4];

	//while (t < ray.far_t_Device[raysIndex] - scene.stepSize)
	//{
	//	baseCoords[0] = ray.cameraCenterX_Device[raysIndex] + t * ray.directionX_Device[raysIndex];
	//	baseCoords[1] = ray.cameraCenterY_Device[raysIndex] + t * ray.directionY_Device[raysIndex];
	//	baseCoords[2] = ray.cameraCenterZ_Device[raysIndex] + t * ray.directionZ_Device[raysIndex];

	//	trilerpRGB(triRes, baseCoords, scene);
	//	triResActivated[0] = density_activation(triRes[0]);
	//	triResActivated[1] = colors_activation(triRes[1]);
	//	triResActivated[2] = colors_activation(triRes[2]);
	//	triResActivated[3] = colors_activation(triRes[3]);

	//	//triResActivated[1] = triRes[1];
	//	//triResActivated[2] = triRes[2];
	//	//triResActivated[3] = triRes[3];

	//	Ti = exp(-accumulatedDensity);
	//	weights = Ti * (1. - exp(-triResActivated[0] * scene.stepSize));
	//	accumulatedDensity += triResActivated[0] * scene.stepSize;
	//	if (weights > threshold)break;

	//	getNeighboorIndex(baseCoords, neighboors, scene);
	//	for (int i = 0; i < 8; i++)
	//	{
	//		scene.adaptiveLRDevice[neighboors[i]] = 0.;
	//		// scene.sceneOpacityDevice[neighboors[i]] = 0.;
	//	}

	//	t += scene.stepSize;
	//}
}

void RenderUtils::pruneSceneByWeights(Rays ray, ObjectScene scene, const double threshold)
{
	dim3 grid((ray.totalRays + 255) / 256, 1);
	dim3 block(16, 16);
	printf("Pruning scene by weights...\n");
	pruneSceneByWeightsGPU << <grid, block >> > (ray, scene, threshold);
	checkGPUStatus("pruneSceneByWeights");
}

__device__ void getVoxelGridIndexes
(
	unsigned int* res,
	const int queryX, const int queryY, const int queryZ,
	ObjectScene scene
)
{
	for (int depth = 0; depth < 2; depth++)
	{
		for (int height = 0; height < 2; height++)
		{
			for (int width = 0; width < 2; width++)
			{
				res[depth * 4 + height * 2 + width] = getSceneIndex
				(
					queryX + width, queryY + height, queryZ + depth,
					scene.sceneWidth, scene.sceneHeight, scene.sceneDepth
				);
				if (scene.sparse)
				{
					res[depth * 4 + height * 2 + width] = scene.indexOffsetsDevice[res[depth * 4 + height * 2 + width]];
				}
			}
		}
	}
}

__global__ void addVarianceLossGPU
(
	ObjectScene scene,
	Optimizer sceneOptimizer,
	double densityDisparityLossCoefficient,
	double shDisparityLossCoefficient
)
{
	int voxelPos = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	if (voxelPos >= (scene.sceneWidth - 1) * (scene.sceneHeight - 1))return;
	int voxelRow = voxelPos / (scene.sceneWidth - 1);
	int voxelCol = voxelPos % (scene.sceneWidth - 1);

	float sumDensity, sumColorChannel[3][25];
	unsigned int eightVoxelIndex[8], voxelIdx;
	const int startSHIdx = (sceneOptimizer.hierarchyStrategy? scene.optimizingBand * scene.optimizingBand:0);
	for (int depth = 0; depth < scene.sceneDepth - 1; depth++)
	{
		voxelIdx = getSceneIndex(voxelCol, voxelRow, depth, scene.sceneWidth - 1, scene.sceneHeight - 1, scene.sceneDepth - 1);
		if (scene.sparse && !scene.occupancyDevice[voxelIdx])continue;

		getVoxelGridIndexes(eightVoxelIndex, voxelCol, voxelRow, depth, scene);

		if (scene.optimizeStep <= 1)
		{
			for (int shIdx = startSHIdx; shIdx < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); shIdx++)
			{
				sumColorChannel[0][shIdx] = 0.f;
				sumColorChannel[1][shIdx] = 0.f;
				sumColorChannel[2][shIdx] = 0.f;

			}

			for (int i = 0; i < 8; i++)
			{
				for (int shIdx = startSHIdx; shIdx < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); shIdx++)
				{
					sumColorChannel[0][shIdx] += scene.AmbientRedDevice[shIdx][eightVoxelIndex[i]] / 8.f;
					sumColorChannel[1][shIdx] += scene.AmbientGreenDevice[shIdx][eightVoxelIndex[i]] / 8.f;
					sumColorChannel[2][shIdx] += scene.AmbientBlueDevice[shIdx][eightVoxelIndex[i]] / 8.f;
				}
			}

			for (int i = 0; i < 8; i++)
			{
				for (int shIdx = startSHIdx; shIdx < (scene.optimizingBand + 1) * (scene.optimizingBand + 1); shIdx++)
				{
					sceneOptimizer.gradientDevice[1 + shIdx][eightVoxelIndex[i]] += shDisparityLossCoefficient * (scene.AmbientRedDevice[shIdx][eightVoxelIndex[i]] - sumColorChannel[0][shIdx]);
					sceneOptimizer.gradientDevice[1 + scene.sh_dim * 1 + shIdx][eightVoxelIndex[i]] += shDisparityLossCoefficient * (scene.AmbientGreenDevice[shIdx][eightVoxelIndex[i]] - sumColorChannel[1][shIdx]);
					sceneOptimizer.gradientDevice[1 + scene.sh_dim * 2 + shIdx][eightVoxelIndex[i]] += shDisparityLossCoefficient * (scene.AmbientBlueDevice[shIdx][eightVoxelIndex[i]] - sumColorChannel[2][shIdx]);
				}
			}

		}

		// setZero
		sumDensity = 0.f;

		for (int i = 0; i < 8; i++)
		{
			sumDensity += scene.sceneOpacityDevice[eightVoxelIndex[i]];
		}

		// get mean
		sumDensity /= 8.f;

		// get disparity loss
		for (int i = 0; i < 8; i++)
		{
			sceneOptimizer.gradientDevice[0][eightVoxelIndex[i]] += densityDisparityLossCoefficient * (scene.sceneOpacityDevice[eightVoxelIndex[i]] - sumDensity);
			// if(scene.sparse)
			// {
			// 	sceneOptimizer.gradientDevice[0][eightVoxelIndex[i]] += densityDisparityLossCoefficient * (sumDensity - 13.f);
			// }
			
		}
	}
}

__device__ void getNeighboorGridIndexes
(
	unsigned int* res,
	const int centroidX, const int centroidY, const int centroidZ,
	ObjectScene scene
)
{
	for (int depth = -1; depth < 2; depth++)
	{
		for (int height = -1; height < 2; height++)
		{
			for (int width = -1; width < 2; width++)
			{
				res[(depth + 1) * 9 + (height + 1) * 3 + width + 1] = getSceneIndex
				(
					centroidX + width, centroidY + height, centroidZ + depth,
					scene.sceneWidth, scene.sceneHeight, scene.sceneDepth
				);
				if (scene.sparse)
				{
					res[(depth + 1) * 9 + (height + 1) * 3 + width + 1] = scene.indexOffsetsDevice[res[(depth + 1) * 9 + (height + 1) * 3 + width + 1]];
				}

			}
		}
	}
}

__global__ void totalVarianceLoss
(
	ObjectScene scene,
	Optimizer sceneOptimizer,
	double densityVarianceLossCoefficient,
	double shVarianceCoefficientCoefficient,
	const int neighboorNums = 6
)
{
	int voxelPos = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	if (voxelPos >= scene.sceneWidth * scene.sceneHeight)return;
	int voxelRow = voxelPos / scene.sceneWidth;
	int voxelCol = voxelPos % scene.sceneWidth;
	if (voxelRow == 0 || voxelRow == scene.sceneHeight - 1 || voxelCol == 0 || voxelCol == scene.sceneWidth - 1)return;

	unsigned int indexs[27];
	/*
		6 indexes: 4 10 12 [13] 14 16 22
	*/
	float sumDifference = 0.f, selfValue, disparity;
	float neighbourValues[27];
	const int startSHIdx = (sceneOptimizer.hierarchyStrategy? scene.optimizingBand * scene.optimizingBand:0);
	if(neighboorNums == 3)
	{
		for (int depth = 0; depth < scene.sceneDepth - 1; depth++)
		{
			getNeighboorGridIndexes(indexs, voxelCol, voxelRow, depth, scene);
			if (scene.sparse && !scene.inCageInfoDevice[indexs[13]])continue;

			sumDifference = 0.f;
			disparity = 0.f;
			neighbourValues[13] = scene.sceneOpacityDevice[indexs[13]];

			selfValue = neighbourValues[13];
	
			neighbourValues[14] = (indexs[14] != 0 ? scene.sceneOpacityDevice[indexs[14]] : selfValue);
			sumDifference += (selfValue - neighbourValues[14]) * (selfValue - neighbourValues[14]);
			disparity += selfValue - neighbourValues[4];

			neighbourValues[16] = (indexs[16] != 0 ? scene.sceneOpacityDevice[indexs[16]] : selfValue);
			sumDifference += (selfValue - neighbourValues[16]) * (selfValue - neighbourValues[16]);
			disparity += selfValue - neighbourValues[16];

			neighbourValues[22] = (indexs[22] != 0 ? scene.sceneOpacityDevice[indexs[22]] : selfValue);
			disparity = selfValue - neighbourValues[22];
			sumDifference += (selfValue - neighbourValues[22]) * (selfValue - neighbourValues[22]);

			sumDifference = sqrt(sumDifference) / 3.f;

			sceneOptimizer.gradientDevice[0][indexs[13]] += (densityVarianceLossCoefficient / (sumDifference + 1.e-6) * disparity);
			// continue;
			for (int colorChannelIdx = 0; colorChannelIdx < 3; colorChannelIdx++)
			{
				for (int shIdx = colorChannelIdx * scene.sh_dim + startSHIdx; shIdx < colorChannelIdx * scene.sh_dim + (scene.optimizingBand + 1) * (scene.optimizingBand + 1); shIdx++)
				{
					sumDifference = 0.f;
					disparity = 0.f;

					neighbourValues[13] = sceneOptimizer.paramsDevice[1 + shIdx][indexs[13]];
					selfValue = neighbourValues[13];
					
					neighbourValues[14] = (indexs[14] != 0 ? sceneOptimizer.paramsDevice[1 + shIdx][indexs[14]] : selfValue);
					disparity += selfValue - neighbourValues[14];
					sumDifference += (selfValue - neighbourValues[14]) * (selfValue - neighbourValues[14]);
					// sumDifference += neighbourValues[14];

					neighbourValues[16] = (indexs[16] != 0 ? sceneOptimizer.paramsDevice[1 + shIdx][indexs[16]] : selfValue);
					disparity += selfValue - neighbourValues[16];
					sumDifference += (selfValue - neighbourValues[16]) * (selfValue - neighbourValues[16]);
					// sumDifference += neighbourValues[16];

					neighbourValues[22] = (indexs[4] != 0 ? sceneOptimizer.paramsDevice[1 + shIdx][indexs[22]] : selfValue);
					disparity = selfValue - neighbourValues[22];
					sumDifference += (selfValue - neighbourValues[22]) * (selfValue - neighbourValues[22]);
					// sumDifference += neighbourValues[4];


					sumDifference = sqrt(sumDifference) / 3.f;
					// sumDifference /= 7.;

					sceneOptimizer.gradientDevice[1 + shIdx][indexs[13]] += (shVarianceCoefficientCoefficient /(sumDifference + 1.e-6) * disparity);
				}
			}
		}
	}
	else if (neighboorNums == 6)
	{
		for (int depth = 1; depth < scene.sceneDepth - 1; depth++)
		{

			getNeighboorGridIndexes(indexs, voxelCol, voxelRow, depth, scene);
			if (scene.sparse && !scene.inCageInfoDevice[indexs[13]])continue;

			sumDifference = 0.f;
			disparity = 0.f;
			neighbourValues[13] = scene.sceneOpacityDevice[indexs[13]];

			selfValue = scene.sceneOpacityDevice[indexs[13]];

			neighbourValues[4] = (indexs[4] != 0 ? scene.sceneOpacityDevice[indexs[4]] : selfValue);
			sumDifference += (selfValue - neighbourValues[4]) * (selfValue - neighbourValues[4]);
			disparity = selfValue - neighbourValues[4];
			// sumDifference += neighbourValues[4];

			neighbourValues[10] = (indexs[10] != 0 ? scene.sceneOpacityDevice[indexs[10]] : selfValue);
			sumDifference += (selfValue - neighbourValues[10]) * (selfValue - neighbourValues[10]);
			disparity += selfValue - neighbourValues[10];
			// sumDifference += neighbourValues[10];

			neighbourValues[12] = (indexs[12] != 0 ? scene.sceneOpacityDevice[indexs[12]] : selfValue);
			sumDifference += (selfValue - neighbourValues[12]) * (selfValue - neighbourValues[12]);
			disparity += selfValue - neighbourValues[12];
			// sumDifference += neighbourValues[12];

			neighbourValues[14] = (indexs[14] != 0 ? scene.sceneOpacityDevice[indexs[14]] : selfValue);
			sumDifference += (selfValue - neighbourValues[14]) * (selfValue - neighbourValues[14]);
			disparity += selfValue - neighbourValues[14];
			// sumDifference += neighbourValues[14];

			neighbourValues[16] = (indexs[16] != 0 ? scene.sceneOpacityDevice[indexs[16]] : selfValue);
			sumDifference += (selfValue - neighbourValues[16]) * (selfValue - neighbourValues[16]);
			disparity += selfValue - neighbourValues[16];
			// sumDifference += neighbourValues[16];

			neighbourValues[22] = (indexs[22] != 0 ? scene.sceneOpacityDevice[indexs[22]] : selfValue);
			sumDifference += (selfValue - neighbourValues[22]) * (selfValue - neighbourValues[22]);
			disparity += selfValue - neighbourValues[22];
			// sumDifference += neighbourValues[22];


			sumDifference = sqrt(sumDifference) / 6.f;
			// sumDifference /= 7.f;

			// sceneOptimizer.gradientDevice[0][indexs[13]] += varianceCoefficient * sumDifference;
			sceneOptimizer.gradientDevice[0][indexs[13]] += densityVarianceLossCoefficient / (sumDifference + 1.e-6) * disparity;


			for (int colorChannelIdx = 0; colorChannelIdx < 3; colorChannelIdx++)
			{
				for (int shIdx = colorChannelIdx * scene.sh_dim + startSHIdx; shIdx < colorChannelIdx * scene.sh_dim + (scene.optimizingBand + 1) * (scene.optimizingBand + 1); shIdx++)
				{
					sumDifference = 0.f;
					disparity = 0.f;

					neighbourValues[13] = sceneOptimizer.paramsDevice[1 + shIdx][indexs[13]];
					selfValue = neighbourValues[13];
					neighbourValues[4] = (indexs[4] != 0 ? sceneOptimizer.paramsDevice[1 + shIdx][indexs[4]] : selfValue);
					disparity = selfValue - neighbourValues[4];
					sumDifference += (selfValue - neighbourValues[4]) * (selfValue - neighbourValues[4]);
					// sumDifference += neighbourValues[4];

					neighbourValues[10] = (indexs[10] != 0 ? sceneOptimizer.paramsDevice[1 + shIdx][indexs[10]] :selfValue);
					disparity += selfValue - neighbourValues[10];
					sumDifference += (selfValue - neighbourValues[10]) * (selfValue - neighbourValues[10]);
					// sumDifference += neighbourValues[10];

					neighbourValues[12] = (indexs[12] != 0 ? sceneOptimizer.paramsDevice[1 + shIdx][indexs[12]] : selfValue);
					disparity += selfValue - neighbourValues[12];
					sumDifference += (selfValue - neighbourValues[12]) * (selfValue - neighbourValues[12]);
					// sumDifference += neighbourValues[12];

					neighbourValues[14] = (indexs[14] != 0 ? sceneOptimizer.paramsDevice[1 + shIdx][indexs[14]] : selfValue);
					disparity += selfValue - neighbourValues[14];
					sumDifference += (selfValue - neighbourValues[14]) * (selfValue - neighbourValues[14]);
					// sumDifference += neighbourValues[14];

					neighbourValues[16] = (indexs[16] != 0 ? sceneOptimizer.paramsDevice[1 + shIdx][indexs[16]] : selfValue);
					disparity += selfValue - neighbourValues[16];
					sumDifference += (selfValue - neighbourValues[16]) * (selfValue - neighbourValues[16]);
					// sumDifference += neighbourValues[16];

					neighbourValues[22] = (indexs[22] != 0 ? sceneOptimizer.paramsDevice[1 + shIdx][indexs[22]] : selfValue);
					disparity += selfValue - neighbourValues[22];
					sumDifference += (selfValue - neighbourValues[22]) * (selfValue - neighbourValues[22]);
					// sumDifference += neighbourValues[22];

					sumDifference = sqrt(sumDifference) / 6.f;
					// sumDifference /= 7.;

					sceneOptimizer.gradientDevice[1 + shIdx][indexs[13]] += shVarianceCoefficientCoefficient / (sumDifference + 1.e-6) * disparity;


				}
			}
		}
	}
	else
	{
		for (int depth = 1; depth < scene.sceneDepth - 1; depth++)
		{
			getNeighboorGridIndexes(indexs, voxelCol, voxelRow, depth, scene);
			if (scene.sparse && !scene.inCageInfoDevice[indexs[13]])continue;
			sumDifference = 0.f;
			disparity = 0.f;
			selfValue = scene.sceneOpacityDevice[indexs[13]];
			neighbourValues[13] = selfValue;

			for (int neighbborIdx = 0; neighbborIdx < 27; neighbborIdx++)
			{
				if (neighbborIdx == 13)continue;

				neighbourValues[neighbborIdx] = (indexs[neighbborIdx] != 0 ? scene.sceneOpacityDevice[indexs[neighbborIdx]] : selfValue);
				sumDifference += (selfValue - neighbourValues[neighbborIdx]) * (selfValue - neighbourValues[neighbborIdx]);
				disparity += selfValue - neighbourValues[neighbborIdx];
				// sumDifference += neighbourValues[neighbborIdx];
			}
			sumDifference = sqrt(sumDifference) / 26.f;
			// sumDifference /= 27.;
			sceneOptimizer.gradientDevice[0][indexs[13]] += densityVarianceLossCoefficient / (sumDifference + 1.e-6) * disparity;
			
			for (int colorChannelIdx = 0; colorChannelIdx < 3; colorChannelIdx++)
			{
				for (int shIdx = colorChannelIdx * scene.sh_dim + startSHIdx; shIdx < colorChannelIdx * scene.sh_dim + (scene.optimizingBand + 1) * (scene.optimizingBand + 1); shIdx++)
				{
					sumDifference = 0.f;
					disparity = 0.f;
					selfValue = sceneOptimizer.paramsDevice[1 + shIdx][indexs[13]];
					neighbourValues[13] = selfValue;

					for (int neighbborIdx = 0; neighbborIdx < 27; neighbborIdx++)
					{
						if (neighbborIdx == 13)continue;

						neighbourValues[neighbborIdx] = (indexs[neighbborIdx] != 0 ? sceneOptimizer.paramsDevice[1 + shIdx][indexs[neighbborIdx]] : selfValue);
						sumDifference += (selfValue - neighbourValues[neighbborIdx]) * (selfValue - neighbourValues[neighbborIdx]);
						disparity += selfValue - neighbourValues[4];
						// sumDifference += neighbourValues[neighbborIdx];
					}
					sumDifference = sqrt(sumDifference) / 26.f;
					// sumDifference /= 27.f;
					sceneOptimizer.gradientDevice[1 + shIdx][indexs[13]] += shVarianceCoefficientCoefficient / (sumDifference + 1.e-6) * disparity;
				}
			}
		}
	}
}

void RenderUtils::addVarianceLoss
(
	ObjectScene scene,
	Optimizer sceneOptimizer,
	double densityCoefficient,
	double shCoefficient,
	const int varianceType,
	const int neighboorNums
)
{
	if (varianceType == 0)
	{
		assert(neighboorNums == 3||neighboorNums == 6 || neighboorNums == 26);
		dim3 grid((scene.sceneWidth + 15) / 16, (scene.sceneHeight + 15) / 16);
		dim3 block(16, 16);

		totalVarianceLoss << <grid, block >> > (scene, sceneOptimizer, densityCoefficient, shCoefficient, neighboorNums);
	}
	else if (varianceType == 1)
	{
		/*average nums is width/height/depth - 1*/
		dim3 grid((scene.sceneWidth - 1 + 15) / 16, (scene.sceneHeight - 1 + 15) / 16);
		dim3 block(16, 16);
		addVarianceLossGPU << <grid, block >> > (scene, sceneOptimizer, densityCoefficient, shCoefficient);
	}
	else if(varianceType == 2)
	{
		dim3 grid((scene.sceneWidth - 1 + 15) / 16, (scene.sceneHeight - 1 + 15) / 16);
		dim3 block(16, 16);

		totalVarianceLoss << <grid, block >> > (scene, sceneOptimizer, densityCoefficient, shCoefficient, neighboorNums);
		addVarianceLossGPU << <grid, block >> > (scene, sceneOptimizer, densityCoefficient, shCoefficient);

	}
	checkGPUStatus("Add variance loss");

}