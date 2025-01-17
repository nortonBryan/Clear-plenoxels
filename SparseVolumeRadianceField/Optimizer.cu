#include "Optimizer.cuh"

__global__ void setZero(float* datas, const int len)
{
	int paramPos = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (paramPos >= len)return;

	datas[paramPos] = 0.f;	
}

void bandMemoryManage
(
	float**& targetGPUPtr, 
	const int dim1Size, 
	const int dim2Size,
	const int optimizingBand,
	const bool densityInitialized =true
)
{
	dim3 grid((dim2Size + 1023) / 1024, 1);
	dim3 block(32, 32);

	float** gpuPtr = (float**)malloc(sizeof(float*) * dim1Size);

	if (optimizingBand != 0&&densityInitialized)
	{
		cudaMemcpy(gpuPtr, targetGPUPtr, sizeof(float*) * dim1Size, cudaMemcpyDeviceToHost);
		// setZero << <grid, block >> > (gpuPtr[0], dim2Size);
	}
	else
	{
		float* densityGPUptr;
		cudaMalloc((void**)&densityGPUptr, sizeof(float) * dim2Size);
		setZero << <grid, block >> > (densityGPUptr, dim2Size);

		gpuPtr[0] = densityGPUptr;
	}

	const int channelDim = (dim1Size - 1) / 3;
	for (int channelIdx = 0; channelIdx < 3; channelIdx++)
	{
		for (int dimIdx = (optimizingBand * optimizingBand); dimIdx < (optimizingBand + 1) * (optimizingBand + 1); dimIdx++)
		{
			float* temp;
			cudaMalloc((void**)&temp, sizeof(float) * dim2Size);
			setZero << <grid, block >> > (temp, dim2Size);

			gpuPtr[1 + channelDim * channelIdx + dimIdx] = temp;
		}
	}

	if (optimizingBand != 0&&densityInitialized)
	{
		int previousBand = optimizingBand - 1;
		for (int channelIdx = 0; channelIdx < 3; channelIdx++)
		{
			for (int dimIdx = (previousBand * previousBand); dimIdx < (previousBand + 1) * (previousBand + 1); dimIdx++)
			{
				cudaFree(gpuPtr[1 + channelDim * channelIdx + dimIdx]);
				gpuPtr[1 + channelDim * channelIdx + dimIdx] = 0;
			}
		}
	}
	cudaMemcpy(targetGPUPtr, gpuPtr, sizeof(float*) * dim1Size, cudaMemcpyHostToDevice);
	free(gpuPtr);
}

void allocateNone(float**& gpuPtr,const int dim1Size)
{

	float** gpu = (float**)malloc(sizeof(float*)*dim1Size);
	cudaMemcpy(gpu, gpuPtr, sizeof(float*) * dim1Size, cudaMemcpyDeviceToHost);
	for(int dim1Idx = 0;dim1Idx<dim1Size;dim1Idx++)
	{
		gpu[dim1Idx] = 0;
	}
	cudaMemcpy(gpuPtr, gpu, sizeof(float*) * dim1Size, cudaMemcpyHostToDevice);
	free(gpu);
}

void optimizer::initialSelf
(
	const int paramsLen, 
	OptimizeMethods methods, 
	float** paramsPtr, 
	const int totalVariableNum
)
{
	variablesNum = totalVariableNum;
	paramLen = paramsLen;
	printf("each params len = %d,total param = %d\n",paramLen,variablesNum);
	
	optimizeMethod = methods;
	paramsDevice = paramsPtr;

	cudaMalloc((void**)&gradientDevice, sizeof(float*) * totalVariableNum);
	allocateNone(gradientDevice,totalVariableNum);

	if (optimizeMethod == MomentumSGD)
	{
		cudaMalloc((void**)&momentumDevice, sizeof(float*) * totalVariableNum);
		allocateNone(momentumDevice,totalVariableNum);
	}
	else if (optimizeMethod == RMSprop)
	{
		cudaMalloc((void**)&momentumDevice, sizeof(float*) * totalVariableNum);
		allocateNone(momentumDevice,totalVariableNum);
	}
	else if (optimizeMethod == Adam)
	{
		cudaMalloc((void**)&momentumDevice, sizeof(float*) * totalVariableNum);
		allocateNone(momentumDevice,totalVariableNum);
		cudaMalloc((void**)&varianceDevice, sizeof(float*) * totalVariableNum);
		allocateNone(varianceDevice,totalVariableNum);
	}

	checkGPUStatus("Initial optimizer",true);
}

void allocateAll(float** gpuDevicePtr, const int dim1Size, const int dim2Size)
{
	dim3 grid((dim2Size + 1023) / 1024, 1);
	dim3 block(32, 32);

	float** gpuPtr = (float**)malloc(sizeof(float*) * dim1Size);

	for (int varIndex = 0; varIndex < dim1Size; varIndex++)
	{
		float* temp;
		cudaMalloc((void**)&temp, sizeof(float) * dim2Size);
		setZero << <grid, block >> > (temp, dim2Size);
		gpuPtr[varIndex] = temp;
	}
	cudaMemcpy(gpuDevicePtr, gpuPtr, sizeof(float*) * dim1Size, cudaMemcpyHostToDevice);
	free(gpuPtr);
	
}

void optimizer::initialBand(const int optimizingBand,const bool densityInitialized)
{
	hierarchyStrategy = true;
	bandMemoryManage(gradientDevice, variablesNum, paramLen, optimizingBand,densityInitialized);

	if (optimizeMethod == MomentumSGD)
	{
		bandMemoryManage(momentumDevice, variablesNum, paramLen, optimizingBand,densityInitialized);
	}
	else if (optimizeMethod == RMSprop)
	{
		bandMemoryManage(momentumDevice, variablesNum, paramLen, optimizingBand,densityInitialized);
	}
	else if (optimizeMethod == Adam)
	{
		bandMemoryManage(momentumDevice, variablesNum, paramLen, optimizingBand,densityInitialized);
		bandMemoryManage(varianceDevice, variablesNum, paramLen, optimizingBand,densityInitialized);
	}

	checkGPUStatus("bandMemoryManage", true);
}

void optimizer::initialAll()
{
	printGPUMemoryInfo();
	hierarchyStrategy = false;
	allocateAll(gradientDevice, variablesNum, paramLen);
	checkGPUStatus("allocate gradient");
	if (optimizeMethod == MomentumSGD)
	{
		allocateAll(momentumDevice, variablesNum, paramLen);
		checkGPUStatus("allocate momentum");
	}
	else if (optimizeMethod == RMSprop)
	{
		allocateAll(momentumDevice, variablesNum, paramLen);
		checkGPUStatus("allocate momentum");
	}
	else if (optimizeMethod == Adam)
	{
		allocateAll(momentumDevice, variablesNum, paramLen);
		checkGPUStatus("allocate momentum");

		printGPUMemoryInfo();
		allocateAll(varianceDevice, variablesNum, paramLen);
		checkGPUStatus("allocate variance");
	}
	checkGPUStatus("allocate all memory", true);
}

void optimizer::setLearningRate(double* lr,double* shLR)
{
	learningRate = lr;
	shLearningRate = shLR;
}

__global__ void naiveSGD(float* params, float* gradients, double lr, const int paramLen)
{
	//int paramPos = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	//if (paramPos >= paramLen)return;

	//params[paramPos] -= lr * gradients[paramPos];
	//gradients[paramPos] = 0.;
}

__global__ void momentumSGD(float* params, float* gradients, float* momentum, double lr, float gamma, const int paramLen)
{
	//int paramPos = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	//if (paramPos >= paramLen)return;

	//momentum[paramPos] = gamma * momentum[paramPos] + lr * gradients[paramPos];
	//params[paramPos] -= momentum[paramPos];
	//gradients[paramPos] = 0.;
}

__global__ void adamUpdate(
	float* params,
	float* gradients,
	float* momentum,
	float* variance,
	double lr,
	int timeStep,
	const int paramLen,
	const float beta1 = 0.9, const float beta2 = 0.999, const float epsilon = 1.e-8
)
{
	//int paramPos = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	//if (paramPos >= paramLen)return;

	//momentum[paramPos] = beta1 * momentum[paramPos] + (1. - beta1) * gradients[paramPos];
	//variance[paramPos] = beta2 * variance[paramPos] + (1. - beta2) * gradients[paramPos] * gradients[paramPos];

	//float correctedMomentum = momentum[paramPos] / (1. - powf(beta1, timeStep));
	//float correctedVariance = variance[paramPos] / (1. - powf(beta2, timeStep));
	//params[paramPos] -= lr * correctedMomentum / (sqrtf(correctedVariance) + epsilon);

	//gradients[paramPos] = 0.;
}

void optimizer::updateSelf(const int timeStep)
{
	dim3 grid((paramLen + 1023) / 1024, 1);
	dim3 block(32, 32);
	//if (optimizeMethod == NoneOptimize || optimizeMethod == NaiveSGD)
	//{
	//	naiveSGD << <grid, block >> >
	//		(
	//			paramsDevice, gradientDevice,
	//			*learningRate, paramLen
	//			);
	//}
	//else if (optimizeMethod == MomentumSGD)
	//{
	//	momentumSGD << <grid, block >> >
	//		(
	//			paramsDevice, gradientDevice, momentumDevice,
	//			*learningRate,
	//			momentum_gamma,
	//			paramLen
	//			);
	//}
	//else if (optimizeMethod == Adam)
	//{
	//	adamUpdate << <grid, block >> >
	//		(
	//			paramsDevice, gradientDevice, momentumDevice, varianceDevice,
	//			*learningRate,
	//			timeStep,
	//			paramLen
	//		);
	//}

	checkGPUStatus("UpdateParams");
}

void free2D(float**& gpuPtr, const int nums)
{
	float** ptr = (float**)malloc(sizeof(float*) * nums);
	cudaMemcpy(ptr, gpuPtr, sizeof(float*) * nums, cudaMemcpyDeviceToHost);
	/*for (int dim = 0; dim < nums; dim++)
	{
		if(ptr[dim])
		{
			cudaFree(ptr[dim]);
		}
		
	}
	checkGPUStatus("Free sub dimension",true);*/
	cudaFree(gpuPtr);
	free(ptr);
}

void optimizer::freeSelf()
{
	free2D(gradientDevice, variablesNum);
	//cudaFree(gradientDevice);
	// checkGPUStatus("Free gradient",true);
	if (optimizeMethod == MomentumSGD)
	{
		//cudaFree(momentumDevice);
		free2D(momentumDevice, variablesNum);
	}
	else if (optimizeMethod == RMSprop)
	{
		//cudaFree(momentumDevice);
		free2D(momentumDevice, variablesNum);
	}
	else if (optimizeMethod == Adam)
	{
		//cudaFree(momentumDevice);
		free2D(momentumDevice, variablesNum);
		// checkGPUStatus("Free momentum",true);
		//cudaFree(varianceDevice);
		free2D(varianceDevice, variablesNum);
		// checkGPUStatus("Free variance",true);
	}
	checkGPUStatus("Free Optimizer");
}

__global__ void updateGraySceneGPU_ADAM(
	optimizer density, optimizer ambientGray,
	ObjectScene scene,
	double lr,
	int timeStep,
	const int paramLen,
	const float beta1 = 0.9, const float beta2 = 0.999, const float epsilon = 1.e-8)
{
	//int paramPos = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	//if (paramPos >= paramLen)return;
	//float correctedMomentum, correctedVariance;

	//density.momentumDevice[paramPos] = beta1 * density.momentumDevice[paramPos] + (1. - beta1) * density.gradientDevice[paramPos];
	//density.varianceDevice[paramPos] = beta2 * density.varianceDevice[paramPos] + (1. - beta2) * density.gradientDevice[paramPos] * density.gradientDevice[paramPos];
	//correctedMomentum = density.momentumDevice[paramPos] / (1. - powf(beta1, timeStep));
	//correctedVariance = density.varianceDevice[paramPos] / (1. - powf(beta2, timeStep));
	//scene.sceneOpacityDevice[paramPos] -= lr * correctedMomentum / (sqrtf(correctedVariance) + epsilon);
	//density.gradientDevice[paramPos] = 0.;

	//ambientGray.momentumDevice[paramPos] = beta1 * ambientGray.momentumDevice[paramPos] + (1. - beta1) * ambientGray.gradientDevice[paramPos];
	//ambientGray.varianceDevice[paramPos] = beta2 * ambientGray.varianceDevice[paramPos] + (1. - beta2) * ambientGray.gradientDevice[paramPos] * ambientGray.gradientDevice[paramPos];
	//correctedMomentum = ambientGray.momentumDevice[paramPos] / (1. - powf(beta1, timeStep));
	//correctedVariance = ambientGray.varianceDevice[paramPos] / (1. - powf(beta2, timeStep));
	//scene.AmbientGrayDevice[paramPos] -= lr * correctedMomentum / (sqrtf(correctedVariance) + epsilon);
	//ambientGray.gradientDevice[paramPos] = 0.;
}

void optimizer::updateGrayScene(
	optimizer density, optimizer ambientGray, 
	ObjectScene scene, int timeStep)
{
	dim3 grid((density.paramLen + 255) / 256, 1);
	dim3 block(16,16);

	updateGraySceneGPU_ADAM << <grid, block >> > (
		density, ambientGray,
		scene, *density.learningRate, timeStep, density.paramLen
	);
	checkGPUStatus("Update Graylized scene params");
}

__global__ void updateRGBSceneGPU_ADAM(
	optimizer sceneOptimizer,
	ObjectScene scene, 
	double lr,double shLR,
	int timeStep,
	const int paramLen,
	const float beta1 = 0.9, const float beta2 = 0.999, const float epsilon = 1.e-6)
{
	int paramPos = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (paramPos >= paramLen)return;

	if(!scene.sparse)
	{
		lr = lr * scene.adaptiveLRDevice[paramPos];
	}

	float correctedMomentum, correctedVariance;
	double temp;
	
	// if(scene.optimizingBand==0)
	{
		sceneOptimizer.momentumDevice[0][paramPos] = beta1 * sceneOptimizer.momentumDevice[0][paramPos] + (1. - beta1) * sceneOptimizer.gradientDevice[0][paramPos];
		sceneOptimizer.varianceDevice[0][paramPos] = beta2 * sceneOptimizer.varianceDevice[0][paramPos] + (1. - beta2) * sceneOptimizer.gradientDevice[0][paramPos] * sceneOptimizer.gradientDevice[0][paramPos];
		correctedMomentum = sceneOptimizer.momentumDevice[0][paramPos] / (1. - powf(beta1, timeStep));
		correctedVariance = sceneOptimizer.varianceDevice[0][paramPos] / (1. - powf(beta2, timeStep));

		temp = sceneOptimizer.paramsDevice[0][paramPos] - lr * correctedMomentum / (sqrtf(correctedVariance) + epsilon);
		sceneOptimizer.paramsDevice[0][paramPos] = temp < 0. ? 0. : temp;

		sceneOptimizer.gradientDevice[0][paramPos] = 0.;
	}
	
	lr  = shLR;

	int start, end;
	for (int channelIdx = 0; channelIdx < 3; channelIdx++)
	{
		start = channelIdx * scene.sh_dim + 1;
		end = start + scene.sh_dim;
		if (sceneOptimizer.hierarchyStrategy)
		{
			start += scene.optimizingBand * scene.optimizingBand;
			end -= (scene.sh_dim - (scene.optimizingBand + 1) * (scene.optimizingBand + 1));
		}

		for (int i = start; i < end; i++)
		{
			sceneOptimizer.momentumDevice[i][paramPos] = beta1 * sceneOptimizer.momentumDevice[i][paramPos] + (1. - beta1) * sceneOptimizer.gradientDevice[i][paramPos];
			sceneOptimizer.varianceDevice[i][paramPos] = beta2 * sceneOptimizer.varianceDevice[i][paramPos] + (1. - beta2) * sceneOptimizer.gradientDevice[i][paramPos] * sceneOptimizer.gradientDevice[i][paramPos];
			correctedMomentum = sceneOptimizer.momentumDevice[i][paramPos] / (1. - powf(beta1, timeStep));
			correctedVariance = sceneOptimizer.varianceDevice[i][paramPos] / (1. - powf(beta2, timeStep));
			sceneOptimizer.paramsDevice[i][paramPos] -= lr * correctedMomentum / (sqrtf(correctedVariance) + epsilon);
			sceneOptimizer.gradientDevice[i][paramPos] = 0.;

		}
	}
}

__global__ void updateRGBSceneGPU_RMSProp
(

	optimizer density,
	optimizer ambientRed, optimizer ambientGreen, optimizer ambientBlue,
	ObjectScene scene,
	double lr,
	const int paramLen,
	const float decayRate = 0.9
)
{
	int paramPos = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (paramPos >= paramLen)return;

	//lr = lr * scene.adaptiveLRDevice[paramPos];

	////treat momentum as past gradients
	//density.momentumDevice[paramPos] += (1. - decayRate) * density.gradientDevice[paramPos] * density.gradientDevice[paramPos];
	//scene.sceneOpacityDevice[paramPos] -= lr * density.gradientDevice[paramPos] / sqrt(density.momentumDevice[paramPos] + 1e-8);
	//density.gradientDevice[paramPos] = 0.;

	//ambientRed.momentumDevice[paramPos] += (1. - decayRate) * ambientRed.gradientDevice[paramPos] * ambientRed.gradientDevice[paramPos];
	//scene.AmbientRedDevice[paramPos] -= lr * ambientRed.gradientDevice[paramPos] / sqrt(ambientRed.momentumDevice[paramPos] + 1e-8);
	//ambientRed.gradientDevice[paramPos] = 0.;

	//ambientGreen.momentumDevice[paramPos] += (1. - decayRate) * ambientGreen.gradientDevice[paramPos] * ambientGreen.gradientDevice[paramPos];
	//scene.AmbientGreenDevice[paramPos] -= lr * ambientGreen.gradientDevice[paramPos] / sqrt(ambientGreen.momentumDevice[paramPos] + 1e-8);
	//ambientGreen.gradientDevice[paramPos] = 0.;

	//ambientBlue.momentumDevice[paramPos] += (1. - decayRate) * ambientBlue.gradientDevice[paramPos] * ambientBlue.gradientDevice[paramPos];
	//scene.AmbientBlueDevice[paramPos] -= lr * ambientBlue.gradientDevice[paramPos] / sqrt(ambientBlue.momentumDevice[paramPos] + 1e-8);
	//ambientBlue.gradientDevice[paramPos] = 0.;

}
void optimizer::updateRGBScene
(
	optimizer sceneOptimizer,
	ObjectScene scene, 
	int timeStep
)
{
	dim3 grid((sceneOptimizer.paramLen + 255) / 256, 1);
	dim3 block(16, 16);

	if (sceneOptimizer.optimizeMethod == Adam)
	{
		updateRGBSceneGPU_ADAM << <grid, block >> > (
			sceneOptimizer,
			scene,
			*sceneOptimizer.learningRate, *sceneOptimizer.shLearningRate, 
			timeStep, sceneOptimizer.paramLen
			);
	}
	else
	{
		//updateRGBSceneGPU_RMSProp << <grid, block >> >
		//(
		//	density, ambientRed, ambientGreen, ambientBlue,
		//	scene, *density.learningRate, density.paramLen
		//);
	}
	checkGPUStatus("Update RGB scene params");
}
