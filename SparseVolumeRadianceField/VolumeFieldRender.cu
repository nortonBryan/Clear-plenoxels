#include "Render.cuh"

/*
================Training rays==================
800 * 800 * 100 * 4 * 17 / 1024 / 1024 = 4151MB
================Training rays==================

================Training views=================
800 * 800 * 100 * 4 / 1024 / 1024 = 245MB, can be freed after transfered to rays
================Training views=================

======================================Scene============================================
Band 0:(1 + 3 * 1) * 4 = 16
Band 1:(1 + 3 * 4) * 4 = 52
	   (1 + 3 * 3) * 4 + 3 * 1 = 43
Band 2:(1 + 3 * 9) * 4 = 112
       (1 + 3 * 5) * 4 + 3 * 4 = 76
Band 3:(1 + 3 * 16) * 4 = 196
       (1 + 3 * 7) * 4 + 3 * 9 = 115
Band 4:(1 + 3 * 25) * 4 = 304
       (1 + 3 * 9) * 4 + 3 * 16 = 160 


no Hierarchy strategy
[64]:
64*64*64*4*4/1024/1024 * (1 + 3 * 9) = 4MB * (1 + 3 * 9) = 112MB
#64: resolution x
#64: resolution y
#64: resolution z
#4: sizeof(float) = 4bytes
#4: self + gradients + momentum + variance,(in adam optimizer case)
#1: density
#3: Red + green + blue = 3 channels
#9: spherical harmonics coefficients for each channel

[128]:
128*128*128*4*4/1024/1024 * (1 + 3 * 9) = 32MB * (1 + 3 * 9) = 896MB

[256]:
256*256*256*4*4/1024/1024 * (1 + 3 * 9) = 256MB * (1 + 3 * 9) = 7168MB = 7GB

[512]:
512*512*512*4*4/1024/1024 * (1 + 3 * 9) = 2048MB * (1 + 3 * 9) = 57344MB = 56GB......

use hierarchy strategy
[64]
64 * 64 * 64 * 4 / 1024 / 1024 = 1MB
For band2
scene: 1 * (1 + 3 * 9) = 28MB
Optimizer: 1 * (1 + 3 * 5) * 3 = 48MB at most
For band3
scene: 1 * (1 + 3 * 16) = 49MB
Optimizer: 1 * (1 + 3 * 7) * 3 = 66MB
For band4
scene: 1 * (1 + 3 * 25) = 76MB
Optimizer: 1 * (1 + 3 * 9) * 3 = 84MB

[128]
128 * 128 * 128 * 4 / 1024 / 1024 = 8MB
For band2
scene: 8 * (1 + 3 * 9) = 224MB
Optimizer: 384
For band3
scene: 392MB
Optimizer:528MB
For band4
scene: 608MB
Optimizer:672MB

[256]
256 * 256 * 256 * 4 / 1024 / 1024 = 64MB
For band2
scene:512
Optimizer:3072
For band3
scene:3136
Optimizer:4224
For band4
scene:4864
Optimizer:5376

256,band4 most:4864+5376+4400=14640

======================================Scene============================================
*/

void testSecondOrderGPUPtr()
{
	int** testGPU, ** testHost = (int**)malloc(sizeof(int*) * 10);
	cudaMalloc((void**)&testGPU, sizeof(int*) * 10);

	for (int i = 0; i < 10; i++)
	{
		int* test;
		cudaMalloc((void**)&test, sizeof(int) * 10);
		testHost[i] = test;
	}
	
	cudaMemcpy(testGPU, testHost, sizeof(int*) * 10, cudaMemcpyHostToDevice);
	free(testHost);

	int** testHostGPU = (int**)malloc(sizeof(int*) * 10);
	cudaMemcpy(testHostGPU, testGPU, sizeof(int*) * 10, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 10; i++)
	{
		cudaFree(testHostGPU[i]);
	}
	cudaFree(testGPU);
}

void testRender()
{
	const int samples = 10;
	double opacity[samples] =
	{ 0.35082109, 0.49677446, 0.10159139, 0.699579, 0.70196073, 0.28737161, 0.88559283, 0.04455409, 0.53510863, 0.52025912 };
	//double opacity[samples] =
	//{ 0., 0., 0., 1., 0., 0., 0., 0., 0., 0. };
	//for (int i = 0; i < samples; i++)
	//{
	//	opacity[i] = 1. - exp(-opacity[i]);
	//}
	double density[3][samples] =
	{
		{ 0.25436352, 0.41225156, 0.62032918, 0.33643962, 0.48815315, 0.10610127, 0.85595129, 0.8505483, 0.69670436, 0.23520701},
		{ 0.40833749, 0.97833265, 0.36634089, 0.91709214, 0.1882598,  0.36327668, 0.51981918, 0.4919779, 0.04402864, 0.72275637},
		{ 0.63738149, 0.66299872, 0.44536173, 0.04668531, 0.76686737, 0.66925641, 0.01505942, 0.68553697, 0.81408741, 0.58484179}
	};
	//double density[3][samples] =
	//{
	//	{0., 0., 0., 0.6, 0., 0., 0., 0., 0., 0.},
	//	{0., 0., 0., 0.6, 0., 0., 0., 0., 0., 0.},
	//	{0., 0., 0., 0.6, 0., 0., 0., 0., 0., 0.}
	//};
	double accumulatedOpacity = 0.;//sum(sigma_i * delta_i)
	double Ti = 1.0;
	double distsNP[samples] = { 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
	double weights[samples];
	double renderRes[3] = { 0., 0., 0. };
	for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
	{
		Ti = exp(-accumulatedOpacity);

		weights[sampleIndex] = Ti * (1. - exp(-opacity[sampleIndex] * distsNP[sampleIndex]));

		accumulatedOpacity += opacity[sampleIndex] * distsNP[sampleIndex];
		for (int channelIndex = 0; channelIndex < 3; channelIndex++)
		{
			renderRes[channelIndex] += weights[sampleIndex] * density[channelIndex][sampleIndex];
		}
	}

	printArray(renderRes, 1, 3, "RenderRes");
	printArray(weights, 1, samples, "weights");

	double accumulatedRenderRes[3] = { 0., 0., 0. };
	double GT[3] = { 0.4, 0.5, 0.6 };
	accumulatedOpacity = 0.;
	double densityGradientDevice[3][samples] = { 0. }, opacityGradientDevice[samples] = { 0. };
	for (int sceneIndex = 0; sceneIndex < samples; sceneIndex++)
	{
		Ti = exp(-accumulatedOpacity);
		weights[sceneIndex] = Ti * (1. - exp(-opacity[sceneIndex] * distsNP[sceneIndex]));

		accumulatedOpacity += opacity[sceneIndex] * distsNP[sceneIndex];
		for (int channelIndex = 0; channelIndex < 3; channelIndex++)
		{
			accumulatedRenderRes[channelIndex] += weights[sceneIndex] * density[channelIndex][sceneIndex];
			densityGradientDevice[channelIndex][sceneIndex] += 0.667 * (renderRes[channelIndex] - GT[channelIndex]) * weights[sceneIndex];
		}

		Ti = exp(-accumulatedOpacity);
		for (int channelIndex = 0; channelIndex < 3; channelIndex++)
		{
			opacityGradientDevice[sceneIndex] += 0.667 * (renderRes[channelIndex] - GT[channelIndex]) * distsNP[sceneIndex] * Ti * density[channelIndex][sceneIndex];
			opacityGradientDevice[sceneIndex] -= 0.667 * (renderRes[channelIndex] - GT[channelIndex]) * distsNP[sceneIndex] * (renderRes[channelIndex] - accumulatedRenderRes[channelIndex]);
		}
	}
	printArray(opacityGradientDevice, 1, samples, "OpacityGradient");
	printArray(densityGradientDevice[0], 1, samples, "redGradient");
	printArray(densityGradientDevice[1], 1, samples, "greenGradient");
	printArray(densityGradientDevice[2], 1, samples, "blueGradient");

}

void getObjectRegion(Configuration* config, Image* gt)
{
	double objectCenter[3] = 
	{
		config->width / pow(2,config->minimumHierarchy + 1),
		config->height / pow(2,config->minimumHierarchy + 1),
		config->depth / pow(2,config->minimumHierarchy + 1)
	};

	int targetImageSize[2] =
	{
		config->renderWidth,
		config->renderHeight
	};

	Camera* trainingCamera = Camera::readFromMyBlender
	(
		config->cameraPoseFile,
		objectCenter, targetImageSize,
		config->viewCount,
		config->cameraDistance
	);

	ObjectScene scene;
	scene.initSelf
	(
		objectCenter[0]*2,objectCenter[1]*2,objectCenter[2]*2, 
		true, true,
		0.1, -10., 0.5,1);
	printf("After initial temp scene:\n");
	printGPUMemoryInfo();
	double ratio = RenderUtils::bbox_silhouette
	(
		trainingCamera, gt, &scene, 
		config->viewCount, config->silhouetteRadius,
		true,config->skipIndex,config->skips
	);
	scene.freeSelf();

	for (int cameraIndex = 0; cameraIndex < config->viewCount; cameraIndex++)
	{
		trainingCamera[cameraIndex].freeSelf();
	}
	double xRadius = std::max(scene.sceneWidth / 2. - scene.bbox_width_min, scene.bbox_width_max - scene.sceneWidth / 2.);
	double yRadius = std::max(scene.sceneHeight / 2. - scene.bbox_height_min, scene.bbox_height_max - scene.sceneHeight / 2.);
	// double zRadius = std::max(scene.sceneDepth / 2. - scene.bbox_depth_min, scene.bbox_depth_max - scene.sceneDepth / 2.);
	double zRadius = scene.bbox_depth_max - scene.sceneDepth / 2.;
	double ratio1 = std::max(std::max(xRadius, yRadius), zRadius) / objectCenter[0];

	double radius = sqrt(xRadius*xRadius+yRadius*yRadius+zRadius*zRadius);
	double ratio2 = radius / objectCenter[0];
	
	printf("farestRatio = %.10f, bound ratio = %.10f,dignoal ratio = %.10f\n",ratio, ratio1, ratio2);
	config->cameraDistance *= (ratio1 + (config->stepSize * 2)/objectCenter[0]);
}

void testRegion(const char* baseDir = "config_linux.txt")
{
	Configuration* config = readConfig(baseDir);

	char imageFiles[256];
	bool graylizeImage = config->graylized;
	int targetImageSize[2] =
	{
		config->renderWidth,
		config->renderHeight
	};

	Image* gtImages = (Image*)malloc(sizeof(Image) * config->viewCount);
	for (int imageIndex = 0; imageIndex < config->viewCount; imageIndex++)
	{
		strcpy(imageFiles, config->viewsDirectory);

		if (graylizeImage)
		{
			sprintf(imageFiles, "%s%shierarchy0%strain%s_%d.txt", imageFiles, PATH_CONCATENATOR, PATH_CONCATENATOR, PATH_CONCATENATOR, imageIndex);
		}
		else
		{
			sprintf(imageFiles, "%s%shierarchy0%strain%s_%d", imageFiles, PATH_CONCATENATOR, PATH_CONCATENATOR, PATH_CONCATENATOR, imageIndex);
		}

		gtImages[imageIndex].initSelfFromFile
		(
			imageFiles,
			targetImageSize[0],
			targetImageSize[1],
			graylizeImage, config->MaskAreaOnly
		);
		cudaDeviceSynchronize();
		checkGPUStatus("Loading dataset");
		printf("successfully loaded %dth images:%s_*.txt\r", imageIndex + 1, imageFiles + 1);
		fflush(stdout);
	}
	printf("\n");

	getObjectRegion(config, gtImages);

	for (int i = 0; i < config->viewCount; i++)
	{
		gtImages[i].freeSelf();
	}
	free(config->skipIndex);
	free(config);
}

void getOrder(Configuration* config,int* idx)
{
	double objectCenter[3] = 
	{
		config->width / pow(2,config->minimumHierarchy + 1),
		config->height / pow(2,config->minimumHierarchy + 1),
		config->depth / pow(2,config->minimumHierarchy + 1)
	};

	int targetImageSize[2] =
	{
		config->renderWidth,
		config->renderHeight
	};

	Camera* trainingCamera = Camera::readFromMyBlender
	(
		config->cameraPoseFile,
		objectCenter, targetImageSize,
		config->viewCount,
		config->cameraDistance
	);

	Camera::sortCameraByZPos(trainingCamera,idx,config->viewCount);

	for (int cameraIndex = 0; cameraIndex < config->viewCount; cameraIndex++)
	{
		trainingCamera[cameraIndex].freeSelf();
	}
}

void getTraingImage(Image* gtImages,Configuration* config)
{
	char imageFiles[256];
	bool graylizeImage = config->graylized;
	int targetImageSize[2] =
	{
		config->renderWidth,
		config->renderHeight
	};

	for (int imageIndex = 0; imageIndex < config->viewCount; imageIndex++)
	{
		strcpy(imageFiles, config->viewsDirectory);

		if (graylizeImage)
		{
			sprintf(imageFiles, "%s%shierarchy0%strain%s_%d.txt", imageFiles, PATH_CONCATENATOR, PATH_CONCATENATOR, PATH_CONCATENATOR, imageIndex);
		}
		else
		{
			sprintf(imageFiles, "%s%shierarchy0%strain%s_%d", imageFiles, PATH_CONCATENATOR, PATH_CONCATENATOR, PATH_CONCATENATOR, imageIndex);
		}

		gtImages[imageIndex].initSelfFromFile
		(
			imageFiles,
			targetImageSize[0],
			targetImageSize[1],
			graylizeImage, config->MaskAreaOnly
		);
		cudaDeviceSynchronize();
		checkGPUStatus("Loading dataset");
		printf("successfully loaded %dth images:%s_*.txt\r", imageIndex + 1, imageFiles + 1);
		fflush(stdout);
	}
	printf("\n");
}

void trainSynthesisBlender(const char* baseDir = "config.txt")
{
	Configuration* config = readConfig(baseDir);
	int* idx = (int*)malloc(sizeof(int)*config->viewCount);
	getOrder(config,idx);

	if(!config->shuffleRays)
	{		
		printf("\norder:\n");
		for(int i = 0;i<config->viewCount;i++)
		{
			printf("%d\t",idx[i]);
			if((i+1)%10==0)
			{
				printf("\n");
			}
		}
		printf("\n");
	}
	else
	{
		printf("Shuffle images...\n");
		for(int i = 0;i<config->viewCount;i++)
		{
			idx[i] = i;
		}
	}

	Image* gtImages = (Image*)malloc(sizeof(Image) * config->viewCount);
	getTraingImage(gtImages,config);
	
	if(config->silhouetteBound)
	{
		getObjectRegion(config, gtImages);
	}
		
	ObjectScene hierarchyObjScene, scene;

	printf("Before initial scene:\n");
	printGPUMemoryInfo();

	if (config->initialFromFile)
	{
		hierarchyObjScene.initFromFile
		(
			config->initialFileDirBase,
			(int)(config->width / pow(2, config->hierarchy)),
			(int)(config->height / pow(2, config->hierarchy)),
			(int)(config->depth / pow(2, config->hierarchy)),
			config->graylized,
			config->stepSize,
			(config->availableSHBand + 1) * (config->availableSHBand + 1),
			(config->sphericalHarmonicsBand + 1)*(config->sphericalHarmonicsBand + 1),
			true,true
		);
		//hierarchyObjScene.pruneSelf(config->pruneThreshold);
		if (!config->renderOnly&&(config->availableSHBand==config->sphericalHarmonicsBand||config->availableSHBand==0&&config->SHsForTargetResoOnly))
		{
			printf("Trained params for hierarchy %d,upSampling automatically...\n", config->hierarchy);
			config->hierarchy -= 1;
			scene.initSelfByUpsample
			(
				hierarchyObjScene, 
				config->pruneThreshold,
				config->SHsForTargetResoOnly&&  config->hierarchy!= config->minimumHierarchy,
				config->hierarchy <= config->minimumHierarchy - config->toSparseHierarchy,
				(config->sphericalHarmonicsBand + 1)*(config->sphericalHarmonicsBand + 1)
			);
			// hierarchyObjScene.freeSelf();
			hierarchyObjScene = scene;
			
			config->upsampled = true;
			// config->silhouetteBBOX = false;
			
		}
	}
	else
	{
		hierarchyObjScene.initSelf
		(
			(int)(config->width / pow(2, config->hierarchy)),
			(int)(config->height / pow(2, config->hierarchy)),
			(int)(config->depth / pow(2, config->hierarchy)),
			config->graylized,
			true,
			config->initialDensity,
			config->initialColor,
			config->stepSize,
			config->SHsForTargetResoOnly?1:(config->sphericalHarmonicsBand + 1) * (config->sphericalHarmonicsBand + 1),0
		);
	}
	printf("After initial scene:\n");
	printGPUMemoryInfo();

	int rawSaveInterval = config->saveIntervals;
	double rawlr = config->learningRate;
	double shLR = config->shLearningRate;

	int rawBatchSize = config->batchSize;
	config->epochs += 3 - config->hierarchy;

	int rawTestSkip = config->testSkip;
	int rawTrainingProcessIdx = 0;

	for (int i = config->hierarchy; i >= config->minimumHierarchy; i--)
	{
		config->learningRate = rawlr;
		config->shLearningRate = shLR;

		// if(config->sphericalHarmonicsBand!=0&&i==config->minimumHierarchy)
		// {
		// 	config->shLearningRate = 0.1;
		// }
		printf("current lr = %.6f, shLR = %.6f\n",config->learningRate,config->shLearningRate);
		config->testSkip = rawTestSkip;
		if(config->shuffleRays)
		{
			// if(i==config->minimumHierarchy)
			// {
			// 	config->epochs = 20;
			// }
			config->epochs += 1;
			
		}
		else
		{
			config->epochs = 10;
		}
		
		config->saveIntervals = config->epochs;

		if (!config->fixBatchSize)
		{
			printf("Double batchsize...\n");
			config->batchSize = rawBatchSize << (3 - i);
		}

		Render hierarchyRender(config, i, idx);
		hierarchyRender.setDataset(gtImages);
		hierarchyRender.setScene
		(
			hierarchyObjScene,
			!config->renderOnly,
			(OptimizeMethods)config->densityOptimizer,
			(OptimizeMethods)config->colorChannelOptimizer
		);

		if (config->trainProcessViewIndex != -1)
		{
			printf("hierarchy index:\033[4;33m%d\033[m\n",rawTrainingProcessIdx);
			hierarchyRender.trainingProcessIdx = rawTrainingProcessIdx;
		}
		hierarchyRender.objectSceneLearner
		(
			config
		);
		rawTrainingProcessIdx = hierarchyRender.trainingProcessIdx;
		if (!config->renderOnly)
		{
			hierarchyRender.freeOptimizers();
		}
		

		// if (i <= config->hierarchy)
		// {
		// 	for (int imageIndex = 0; imageIndex < config->viewCount; imageIndex++)
		// 	{
		// 		RenderUtils::getExpectTermination(gtImages[imageIndex], hierarchyRender.train_cameras[imageIndex], hierarchyObjScene);
		// 		printf("Aquire learned depth info %3d/%3d...\r", imageIndex + 1, config->viewCount);
		// 		fflush(stdout);
		// 	}
		// }

		//hierarchyObjScene.getEffectiveCount(hierarchyObjScene, false);
		//char modelSavePath[255];
		//sprintf(modelSavePath, "%s%shierarchy%dModel.ply", config->resBaseDirectory, PATH_CONCATENATOR, i);
		//hierarchyObjScene.saveSelf2ply(modelSavePath,false);

		int saveScene = 0;
		// printf("Save scene?");
		// scanf("%d", &saveScene);
		if (saveScene)
		{
			char savePath[1024];

			sprintf(savePath,"mkdir %s%shierarchy_%dModel",config->resBaseDirectory,PATH_CONCATENATOR,i);
			system(savePath);
			sprintf(savePath,"%s%shierarchy_%dModel",config->resBaseDirectory,PATH_CONCATENATOR,i);
			hierarchyObjScene.saveSelf(savePath,false,config->sphericalHarmonicsBand);
		}

		if (i > config->minimumHierarchy)
		{
			//hierarchyObjScene.pruneSelf(config->pruneThreshold);
			scene.initSelfByUpsample
			(
				hierarchyObjScene,
				config->pruneThreshold,
				config->SHsForTargetResoOnly&& i != config->minimumHierarchy + 1,//lowBandOnly
				i <= config->minimumHierarchy - config->toSparseHierarchy,//toSparse
				(config->sphericalHarmonicsBand + 1)*(config->sphericalHarmonicsBand + 1)
			);
			if(i > config->minimumHierarchy - config->toSparseHierarchy)
			{
				hierarchyObjScene.freeSelf();
			}			

			hierarchyObjScene = scene;
			
			// config->silhouetteBBOX = false;
		}
		// config->varianceLossInterval = config->varianceLossInterval/2>0?config->varianceLossInterval/2:1;

		// config->varianceCoefficient*=pow(10.,(config->hierarchy-i+1)/4.);
		// config->shCoefficient*=pow(10.,(config->hierarchy-i+1)/4.);
		// config->maskLossCoefficient*=pow(10.,(config->hierarchy-i)*1.);
		// config->varianceLossInterval *= 2;
	}

	for (int i = 0; i < config->viewCount; i++)
	{
		gtImages[i].freeSelf();
	}

	if (config->exportModel)
	{
		if (hierarchyObjScene.sceneWidth > 512)
		{
			printf("scene is too big!\n");
		}
		else if (hierarchyObjScene.sceneWidth == 512)
		{
			hierarchyObjScene.getEffectiveCount(hierarchyObjScene, false, 13.);
			char modelName[512];
			sprintf(modelName, "%s%sexportedModel_%d.ply", config->resBaseDirectory, PATH_CONCATENATOR, 512);
			hierarchyObjScene.saveSelf2ply(modelName, false, 13.);
		}
		else
		{
			ObjectScene upsampled;
			upsampled.initSelfByUpsample(hierarchyObjScene, config->pruneThreshold,true);
			upsampled.getEffectiveCount(upsampled, false, 13.);
			char modelName[512];
			sprintf(modelName, "%s%sexportedModel_%d.ply", config->resBaseDirectory, PATH_CONCATENATOR, upsampled.sceneWidth);
			upsampled.saveSelf2ply(modelName, false, 13.);
			upsampled.freeSelf();
		}
	}
	int saveScene = 0;
	printf("Save scene?");
	scanf("%d", &saveScene);
	if (saveScene)
	{
		hierarchyObjScene.saveSelf(config->resBaseDirectory,false,config->sphericalHarmonicsBand);
	}

	free(idx);
	hierarchyObjScene.freeSelf();
	free(config->skipIndex);
	free(config);

}

void testBlenderDatasets()
{
	const int imageNums[3] = { 100,100,200 };
	double objectCenter[3] = { 0., 0., 0. };
	int targetImageSize[2] = { 800,800 };
	const char types[3][256] = { "train","val","test" };
	const char* baseDir = "/norton/Datasets/finalDataset/saucer/";
	char viewInfoDir[256];
	char centerInfoFile[256];
	for (int i = 0; i < 3; i++)
	{
		sprintf(viewInfoDir, "%s%s%s%scameraInfo.txt", baseDir, PATH_CONCATENATOR, types[i], PATH_CONCATENATOR);
		sprintf(centerInfoFile, "%s%s%s%scenterInfo.txt", baseDir, PATH_CONCATENATOR, types[i], PATH_CONCATENATOR);
		Camera* cameras = Camera::readFromMyBlender(viewInfoDir, objectCenter, targetImageSize, imageNums[i], 50., false, false);
		Camera::writeJasonFile(cameras, baseDir, imageNums[i], types[i]);
		for (int j = 0; j < imageNums[i]; j++)
		{
			//cameras[j].saveSelfCenterInfo(centerInfoFile);
			cameras[j].freeSelf();
		}
		free(cameras);
		printf("%s finished\n", types[i]);
	}

}

void testUniform()
{
	const int len = 64;
	double* rowDevice, * rowHost = (double*)malloc(sizeof(double) * len);
	double* colDevice, * colHost = (double*)malloc(sizeof(double) * len);
	cudaMalloc((void**)&rowDevice, sizeof(double) * len);
	cudaMalloc((void**)&colDevice, sizeof(double) * len);

	char rowsFileName[512], colsFileName[512];
	char commands[512] = "mkdir randoms";
	system(commands);
	for (int i = 0; i < 10; i++)
	{
		randGPU(rowDevice, len, 0.166, 1);
		randGPU(colDevice, len, 0.166, 1);
		cudaMemcpy(rowHost, rowDevice, sizeof(double) * len, cudaMemcpyDeviceToHost);
		cudaMemcpy(colHost, colDevice, sizeof(double) * len, cudaMemcpyDeviceToHost);
		sprintf(rowsFileName, "randoms%srows_%d.txt", PATH_CONCATENATOR, i);
		sprintf(colsFileName, "randoms%scols_%d.txt", PATH_CONCATENATOR, i);
		saveDouble2File(rowHost, len, 1, rowsFileName);
		saveDouble2File(colHost, len, 1, colsFileName);
	}

	cudaFree(rowDevice);
	cudaFree(colDevice);
	free(rowHost);
	free(colHost);

}

int main(int argc, char** argv)
{
	// system("mode con cols=200");
	// for(int i=0;i<20;i++)
	// 	printf("........../");
	// printf("\n");
	// return 0;
	// testRegion();
	// return 0;
	//testUpsample();
	//return 0;
	// testUniform();
	// return 0;
	// testBlenderDatasets();
	// return 0;

#if defined(__linux__)
	if (argc < 3)
	{
		printf("Both \033[1;33mconfiguration file path\033[m and \033[1;33mGPU device id\033[m are needed!\n");
		return 0;
	}
	char baseDir[256];
	strcpy(baseDir, argv[1]);
	cudaSetDevice(atoi(argv[2]));
	if (cudaFree(0) != cudaSuccess)
	{
		printf("Unable to initialize GPU...\n");
		return 0;
	}

	trainSynthesisBlender(baseDir);

#else
	cudaSetDevice(atoi(argv[2]));
	if (cudaFree(0) != cudaSuccess)
	{
		printf("Unable to initialize GPU...\n");
		return 0;
	}
	trainSynthesisBlender();
#endif

	return 0;
}
