#include "Render.cuh"

Render::~Render()
{
	freeCameras();

	cudaFree(this->sampleRaysIndexGPU);
	free(sampleRaysIndexHost);

	checkGPUStatus("free render");
	this->renderRes.freeSelf();
	this->ray.freeSelf();

}

Render::Render(Configuration* config, const int hierarchy,int* idx)
{
	this->idx = idx;
	this->loadCameras(config, hierarchy);


	this->grayLized = config->graylized;
	this->datasetSize = config->viewCount;

	this->currentHierarchy = hierarchy;
}

void Render::shuffleRays(const int maxIndex, bool initialized,const bool shuffle)
{
	printf("Shuffle rays...\n");
	if (!initialized)
	{
		this->sampleRaysIndexHost = (int*)malloc(sizeof(int) * maxIndex);
		cudaMalloc((void**)&this->sampleRaysIndexGPU, sizeof(int) * maxIndex);	
	}

	ShuffleRank(this->sampleRaysIndexHost, maxIndex, !initialized,0,shuffle);
	cudaMemcpy(this->sampleRaysIndexGPU, this->sampleRaysIndexHost, sizeof(int) * maxIndex, cudaMemcpyHostToDevice);
	
	checkGPUStatus("ShuffleRays");
}

void Render::regenerateSamples()
{
	for(int i = 0;i<this->ray.totalRays;i++)
	{
		this->sampleRaysIndexHost[i] = i;
	}
	cudaMemcpy(this->sampleRaysIndexGPU, this->sampleRaysIndexHost, sizeof(int) * this->ray.totalRays, cudaMemcpyHostToDevice);
}

void Render::setDataset(Image* groundTruth)
{
	this->groundTruth = groundTruth;
}

void Render::setScene(ObjectScene& scene, bool initialOptimizer,
	OptimizeMethods densityOptimizer, OptimizeMethods colorChannelOptimizer)
{
	this->scene = &scene;
	if (!initialOptimizer)return;

	this->sceneOptimizer.initialSelf(scene.actualGridNums, densityOptimizer, scene.totalVariableGPUPtrsDevice, scene.sh_dim * 3 + 1);
}

void Render::gatherInfoAlongARay(Configuration* configs, const int viewIndex)
{
	char fileName[1024];
	sprintf(fileName, "%s%shierarchy_%dDensities.txt", configs->resBaseDirectory, PATH_CONCATENATOR, currentHierarchy);

	int effectivePts;
	float* densities = RenderUtils::gatherDensitiesAlongRay(this->test_cameras[viewIndex], *this->scene, &effectivePts);
	saveFloaters2File(densities, 1, effectivePts, fileName, true);
	free(densities);
}

void Render::objectSceneLearner(Configuration* configs)
{
	if (configs->silhouetteInitial)
	{
		if(!this->scene->sparse)
		{
			RenderUtils::bbox_silhouette
			(
				this->train_cameras,
				this->groundTruth,
				this->scene,
				this->datasetSize,
				configs->silhouetteRadius,false,
				configs->skipIndex,configs->skips
			);
		}
		else
		{
			ObjectScene tempScene;
			tempScene.initSelf
			(
				this->scene->sceneWidth,this->scene->sceneHeight,this->scene->sceneDepth, 
				true, true,
				0.1, -10., 0.5, 1
			);
			RenderUtils::bbox_silhouette
			(
				this->train_cameras,this->groundTruth,
				&tempScene, 
				configs->viewCount, configs->silhouetteRadius,
				true,configs->skipIndex,configs->skips
			);
			this->scene->bbox_depth_max = tempScene.bbox_depth_max;
			this->scene->bbox_depth_min = tempScene.bbox_depth_min;
			this->scene->bbox_height_max= tempScene.bbox_height_max;
			this->scene->bbox_height_min = tempScene.bbox_height_min;
			this->scene->bbox_width_max = tempScene.bbox_width_max;
			this->scene->bbox_width_min = tempScene.bbox_width_min;

			cudaMemcpy(this->scene->adaptiveLRDevice, tempScene.adaptiveLRDevice, sizeof(float) * this->scene->sceneWidth * this->scene->sceneHeight * this->scene->sceneDepth, cudaMemcpyDeviceToDevice);
			checkGPUStatus("memcpy adaptive lr",true);
			tempScene.freeSelf();

		}

	}
	else
	{
		ObjectScene tempScene;
		tempScene.initSelf
		(
			this->scene->sceneWidth,this->scene->sceneHeight,this->scene->sceneDepth, 
			true, true,
			0.1, -10., 0.5, 1
		);
		RenderUtils::bbox_silhouette
		(
			this->train_cameras,this->groundTruth,
			&tempScene, 
			configs->viewCount, configs->silhouetteRadius,
			true,configs->skipIndex,configs->skips
		);
		this->scene->bbox_depth_max = tempScene.bbox_depth_max;
		this->scene->bbox_depth_min = tempScene.bbox_depth_min;
		this->scene->bbox_height_max= tempScene.bbox_height_max;
		this->scene->bbox_height_min = tempScene.bbox_height_min;
		this->scene->bbox_width_max = tempScene.bbox_width_max;
		this->scene->bbox_width_min = tempScene.bbox_width_min;
		tempScene.freeSelf();
	}
	//RenderUtils::renderImage
	//(
	//	this->renderRes,
	//	this->train_cameras[0],
	//	this->scene
	//);
	//this->renderRes.copy2HostMemory();
	//this->renderRes.saveSelf("test");
	//char additionDirs[255];
	//sprintf(additionDirs, "_%dEpochsRes", 0);
	//saveExpRes(configs, additionDirs, false);
	//this->shuffleRays(this->renderRes.width * this->renderRes.height);
	//return;

	printGPUMemoryInfo();

	this->renderRes.initSelf
	(
		configs->renderWidth, configs->renderHeight,
		configs->graylized,
		configs->whiteBG,
		false,
		true,
		true
	);

	if (configs->renderOnly)
	{

		this->scene->optimizingBand = configs->sphericalHarmonicsBand;
		int speicifyPath = 0;
		printf("Specify path?");
		scanf("%d",&speicifyPath);
		if(speicifyPath)
		{
			while(speicifyPath)
			{
				this->renderSpecifiedPath(configs);
				
				printf("Try again?");
				scanf("%d",&speicifyPath);
			}
			
			printf("Get metrics?");
			scanf("%d",&speicifyPath);
			if(!speicifyPath)
			{
				return;
			}
			else
			{
				configs->testSkip = 1;
				printf("Getting metrics, set test skip to 1 automatically...\n");
			}
		}
	

		if(configs->testSkip==-1)return;

		char saveDir[255] = "RenderOnlyRes";
		configs->testSkip = 1;
		int renderTrainingView = 1;
		
		// printf("Render Training View?");
		// scanf("%d",&renderTrainingView);
		saveExpRes(configs, saveDir, false, renderTrainingView);

		return;
	}

	_rays temp;
	temp.initialSelf
	(
		this->train_cameras, 
		this->groundTruth, 
		*this->scene, 
		this->datasetSize,
		configs->MaskAreaOnly, 
		false,
		configs->whiteBG,
		this->idx
	);
	this->shuffleRays(temp.totalRays,false,configs->shuffleRays);
	temp.shuffleTo(&this->ray, this->sampleRaysIndexGPU);
	temp.freeSelf();
	regenerateSamples();

	// this->ray.initialSelf(this->train_cameras, this->groundTruth, *this->scene, this->datasetSize);
	// this->shuffleRays(this->ray.totalRays);

	printf("After Initial training rays:\n");
	printGPUMemoryInfo();

	double losses;
	double maxLoss = 0., minLoss = 255.;
	double* lossesPerEpochs = (double*)calloc(configs->epochs * 4, sizeof(double));
	double* lossesHost, * lossesDevice;
	lossesHost = (double*)calloc(configs->batchSize, sizeof(double));
	cudaMalloc((void**)&lossesDevice, sizeof(double) * configs->batchSize);
	
	int startSampleIndex = 0;

	this->sceneOptimizer.setLearningRate(&configs->learningRate,&configs->shLearningRate);

	printf("Extral loss type:\033[1;31m%s\033[m\n",configs->vairanceType==0?"total variance loss":"Voxel disparity loss");
	printf("Density:%.10lf,SH:%.10f,mask:%.10lf\n",configs->varianceCoefficient,configs->shCoefficient,configs->maskLossCoefficient);
	
	getCurrentTime(configs->startTime);
	char lossFileDirectory[256];
	int viewIndex, timeStep;

	const int batchCount = (this->ray.totalRays + configs->batchSize - 1) / configs->batchSize;
	int actualBatchSize;
	printf("Batch size = %-6d, batchCount = %6d\n",configs->batchSize,batchCount);
	int trainProcessTestIndex = 0;
	int rayDensitiesSaveIndex = 0;

	this->scene->optimizeStep = currentHierarchy;
	double rawLR = configs->learningRate;
	double rawSHLR = configs->shLearningRate;
	int startBand = (configs->initialFromFile?(configs->upsampled?0:configs->availableSHBand + 1):0);
	
	if(!configs->hierarchyBandStrategy)
	{
		startBand = configs->sphericalHarmonicsBand;
		if(configs->SHsForTargetResoOnly&&currentHierarchy!=configs->minimumHierarchy)
		{
			startBand=0;
		}
	}

	this->sceneOptimizer.densityOptimizedStep = 0;
	if(this->scene->sparse)
	{
		printf("Sparse scene\n");
	}
	for (int optimizingBand = startBand ; optimizingBand <= configs->sphericalHarmonicsBand; optimizingBand++)
	{
		this->scene->optimizingBand = optimizingBand;
		if (configs->hierarchyBandStrategy||(!configs->hierarchyBandStrategy&&configs->SHsForTargetResoOnly&&currentHierarchy!=configs->minimumHierarchy))
		{
			bool densityInitialized = true;
			if(configs->initialFromFile&&optimizingBand==configs->availableSHBand+1)
			{
				densityInitialized = false;
				
			}
			
			this->sceneOptimizer.initialBand(optimizingBand,densityInitialized);
		}
		else
		{

			this->sceneOptimizer.initialAll();
		}

		size_t freeGPUMemory, totalMemory;
		cudaMemGetInfo(&freeGPUMemory, &totalMemory);
		
		if(!configs->justModel)
		{
			char initialState[255];
			sprintf(initialState,"InitialState_band%d",optimizingBand);			
			saveExpRes(configs, initialState, false, false);
		}

		printf("Optimizing band:%d,total sh_dim:%d\n", optimizingBand,this->scene->sh_dim);
		configs->learningRate = rawLR;
		configs->shLearningRate = rawSHLR;
		for (int epoch = 0; epoch < configs->epochs; epoch++)
		{
			// if(configs->hierarchy==configs->minimumHierarchy&&epoch==2&&!this->scene->sparse)
			// {
			// 	this->scene->transferSelf2Sparse(configs->pruneThreshold);
			// }
			if (!configs->justModel&&configs->trainProcessViewIndex == -1)
			{
				srand((unsigned int)time(NULL));
				RenderUtils::renderImage
				(
					this->renderRes,
					this->val_cameras[rand()%configs->validationCount],
					*this->scene
				);
				srand(configs->randomSeed);
				this->renderRes.copy2HostMemory();
				char epochInfo[512];
				sprintf
				(
					epochInfo, "%s%strainProcess_hierarchy%d%s_band%d_%d",
					configs->resBaseDirectory,
					PATH_CONCATENATOR, this->currentHierarchy, PATH_CONCATENATOR,
					optimizingBand, epoch
				);
				this->renderRes.saveSelf(epochInfo);

			}
			losses = 0.;
			maxLoss = 0., minLoss = 65025.;

			cudaEvent_t startTimeGPU, endTimeGPU;
			cudaEventCreate(&startTimeGPU);
			cudaEventCreate(&endTimeGPU);
			cudaEventRecord(startTimeGPU, 0);

			for (int batchIndex = 0; batchIndex < batchCount; batchIndex++)
			{
				if (configs->trainProcessViewIndex != -1 && batchIndex % (min((int)pow(2,this->currentHierarchy),(int)pow(2, this->trainingProcessIdx / configs->testCount))) == 0)
				{
					if(configs->trainProcessViewIndex<100&&configs>=0)
					{
						RenderUtils::renderImage
						(
							this->renderRes,
							this->train_cameras[configs->trainProcessViewIndex],
							// this->test_cameras[trainProcessTestIndex % configs->testCount],
							*this->scene
						);
					}
					else if(configs->trainProcessViewIndex>=100&&configs->trainProcessViewIndex<200)
					{
						RenderUtils::renderImage
						(
							this->renderRes,
							this->val_cameras[configs->trainProcessViewIndex-100],
							// this->test_cameras[trainProcessTestIndex % configs->testCount],
							*this->scene
						);
					}
					else if(configs->trainProcessViewIndex>=200&&configs->trainProcessViewIndex<400)
					{
						RenderUtils::renderImage
						(
							this->renderRes,
							this->test_cameras[configs->trainProcessViewIndex-200],
							// this->test_cameras[trainProcessTestIndex % configs->testCount],
							*this->scene
						);
					}

					this->renderRes.copy2HostMemory();
					char epochInfo[512];
					sprintf
					(
						epochInfo, "%s%strainProcess_hierarchy%d%s_band%d_%d",
						configs->resBaseDirectory,
						PATH_CONCATENATOR, configs->minimumHierarchy, PATH_CONCATENATOR,
						optimizingBand, this->trainingProcessIdx++
					);
					this->renderRes.saveSelf(epochInfo,true);
				}
				// this->gatherInfoAlongARay(configs,50);
				startSampleIndex = batchIndex * configs->batchSize;
				actualBatchSize = startSampleIndex + configs->batchSize < this->ray.totalRays ? configs->batchSize : this->ray.totalRays - startSampleIndex;

				if (!configs->solidStrategy)
				{
					this->scene->setSelfSampleOffset(this->scene->stepSize/2.);
				}

				RenderUtils::renderRays
				(
					this->ray,
					*this->scene,
					this->sampleRaysIndexGPU,
					startSampleIndex,
					actualBatchSize
				);

				if (this->renderRes.grayImage)
				{
					//RenderUtils::renderRays_withGradients_Gray
					//(
					//	this->densityOptimizer, this->ambientGrayOptimizer,
					//	this->ray, *this->scene,
					//	lossesDevice,
					//	this->sampleRaysIndexGPU,
					//	startSampleIndex,
					//	actualBatchSize
					//);
				}
				else
				{
					RenderUtils::renderRays_withGradients_RGB
					(
						this->sceneOptimizer,
						this->ray, *this->scene,
						lossesDevice,
						this->sampleRaysIndexGPU,
						startSampleIndex,
						actualBatchSize,
						configs->maskLossCoefficient,
						configs->huberDelta
					);
				}

				cudaMemcpy
				(
					lossesHost, lossesDevice,
					sizeof(double) * actualBatchSize,
					cudaMemcpyDeviceToHost
				);
				double loss = getMSEError(lossesHost, actualBatchSize);
				if (isnan(loss))
				{
					printf("nan occurred...exitting\n");
					free(lossesPerEpochs);
					cudaFree(lossesDevice);
					free(lossesHost);
					return;
				}
				if (loss > maxLoss)maxLoss = loss;
				if (minLoss > loss)minLoss = loss;
				losses += loss;
				printf("%3d/%-3d Epochs,%6d/%-6d \033[1;33m%.10lf\033[m,\033[1;36m%.6lf\033[m,\033[1;32m%.6lf\033[m,\033[1;35m%.6f\033[m\r",
					epoch + 1, configs->epochs,
					batchIndex + 1, batchCount,
					loss, maxLoss, minLoss,
					10. * log10(65025. / loss));
				fflush(stdout);
				timeStep = epoch * batchCount + batchIndex + 1;

				if (!this->scene->sparse||batchIndex % configs->varianceLossInterval == 0)
				{	
					RenderUtils::addVarianceLoss
					(
						*this->scene,
						this->sceneOptimizer,
						configs->varianceCoefficient,
						configs->shCoefficient,
						configs->vairanceType,
						configs->neighboorNums
					);
				}

				if (this->scene->grayImage)
				{
					//Optimizer::updateGrayScene
					//(
					//	this->densityOptimizer, this->ambientGrayOptimizer,
					//	*this->scene, timeStep
					//);
				}
				else
				{
					Optimizer::updateRGBScene
					(
						this->sceneOptimizer,
						*this->scene, timeStep
					);
				}

				configs->learningRate /= pow(2, 1. / batchCount / (epoch + 1.));
				configs->shLearningRate /= pow(2, 1. / batchCount / (epoch + 1.));
				//configs->halfLRIntervals = 2;

			}
			lossesPerEpochs[epoch * 4 + 0] = minLoss;
			lossesPerEpochs[epoch * 4 + 1] = maxLoss;
			lossesPerEpochs[epoch * 4 + 2] = losses / batchCount;
			cudaEventRecord(endTimeGPU, 0);
			cudaEventSynchronize(endTimeGPU);
			float elapsedTime;
			cudaEventElapsedTime(&elapsedTime, startTimeGPU, endTimeGPU);
			lossesPerEpochs[epoch * 4 + 3] = elapsedTime;
			printf("%5d/%-5d Epochs, ROI's mse =\033[1;31m %.10lf\033[m,maxLoss=\033[1;36m%.6lf\033[m,minLoss=\033[1;32m%.6lf\033[m,PSNR =\033[1;35m %.6f\033[m,lr=\033[1;34m%.6lf\033[m,cost %.6f ms\n",
				epoch + 1, configs->epochs, losses / batchCount, maxLoss, minLoss,
				10. * log10(65025. / (losses / batchCount)),
				configs->learningRate, elapsedTime);
			fflush(stdout);
			// this->shuffleRays(this->ray.totalRays, true);

			if (!configs->justModel&&(epoch + 1) % configs->saveIntervals == 0)
			{
				char additionDir[255];
				sprintf(additionDir, "Band_%d", optimizingBand);
				saveExpRes(configs, additionDir, false, false);
			}

			// if(!configs->justModel&&currentHierarchy==configs->minimumHierarchy&&epoch>=8&&epoch!=configs->epochs-1)
			// {
			// 	char additionDir[255];
			// 	sprintf(additionDir, "Band_%d_Epoch_%d", optimizingBand,epoch);
			// 	configs->testSkip = 1;
			// 	saveExpRes(configs, additionDir, false, false);
			// }


			// if(currentHierarchy!=configs->minimumHierarchy)
			// {
			// 	printf("Breaking...\n");
			// 	break;
			// }

			// if (epoch + 1 == configs->pruneEpoch)
			// {
			// 	this->scene->pruneSelf(configs->pruneThreshold);
			// 	RenderUtils::pruneSceneByWeights(this->ray, *this->scene, configs->pruneThreshold);
			// 	this->scene->optimizeStep = 10;
			// }
			// this->shuffleRays(temp.totalRays,true,configs->shuffleRays);
		}
		if (!configs->justModel&&configs->trainProcessViewIndex == -1)
		{
			RenderUtils::renderImage
			(
				this->renderRes,
				this->train_cameras[rand()%configs->validationCount],
				*this->scene
			);
			this->renderRes.copy2HostMemory();
			char epochInfo[512];
			sprintf
			(
				epochInfo, "%s%strainProcess_hierarchy%d%s_band%d_%d",
				configs->resBaseDirectory,
				PATH_CONCATENATOR, this->currentHierarchy, PATH_CONCATENATOR,
				optimizingBand, configs->epochs
			);
			this->renderRes.saveSelf(epochInfo);

		}
		printf("\n");

/*
		if (!this->scene->sparse&&optimizingBand == 0&&currentHierarchy==configs->minimumHierarchy)
		{
			printf("Pruning...\n");
			this->scene->pruneSelf(configs->pruneThreshold);	
		}
		 if (optimizingBand == 0&&currentHierarchy==configs->minimumHierarchy)
		 {
			 for (int batchIndex = 0; batchIndex < batchCount; batchIndex++)
			 {
				 startSampleIndex = batchIndex * configs->batchSize;
				 actualBatchSize = startSampleIndex + configs->batchSize < this->ray.totalRays ? configs->batchSize : this->ray.totalRays - startSampleIndex;
				 RenderUtils::boundRays
				 (
					 this->ray,
					 *this->scene,
					 this->sampleRaysIndexGPU,
					 startSampleIndex,
					 actualBatchSize,
					 configs->pruneThreshold
				 );
				 printf("%4d/%-4d Get tighter bounds...\r", batchIndex + 1, batchCount);
				 fflush(stdout);
			 }
			printf("\n");
		 }
		when arriving target grid resolution, save each band's density individually 
		to inspect spherical harmonics band impact.
*/

		// if (configs->exportModel&&currentHierarchy == 1)
		// {
		// 	this->scene->saveSelf(configs->resBaseDirectory, false, optimizingBand);
		// }
		
		sprintf(lossFileDirectory, "%s%slosses_%s.txt", configs->optimizedResDirectory, PATH_CONCATENATOR, configs->startTime);
		saveDouble2File(lossesPerEpochs, configs->epochs, 4, lossFileDirectory,true);
		if(configs->SHsForTargetResoOnly&&currentHierarchy!=configs->minimumHierarchy)
		{
			printf("Spherical harmonics for target resolution only, no higher bands optimizations......\n");

			break;
		}
		this->sceneOptimizer.densityOptimizedStep += timeStep;

		if(configs->sphericalHarmonicsBand==4&&configs->hierarchyBandStrategy&&currentHierarchy==configs->minimumHierarchy)
		{
			char finalDir[255];
			sprintf(finalDir, "final_band%d",optimizingBand);
			configs->testSkip = 1;
			int saveTrainingView = 1;
			// printf("Save training views?");
			// scanf("%d",&saveTrainingView);

			size_t freeGPUMemory, totalMemory;
			cudaMemGetInfo(&freeGPUMemory, &totalMemory);

			saveExpRes(configs, finalDir,false,saveTrainingView!=0);

		}
	}

	if(!configs->justModel&&configs->minimumHierarchy==currentHierarchy)
	{
		int saveFinal = 1;
		// printf("Save hierarchy %d final res?",this->currentHierarchy);
		// scanf("%d", &saveFinal);
		if (saveFinal)
		{
			char finalDir[255];
			sprintf(finalDir, "final");
			configs->testSkip = 1;
			int saveTrainingView = 1;
			// printf("Save training views?");
			// scanf("%d",&saveTrainingView);

			size_t freeGPUMemory, totalMemory;
			cudaMemGetInfo(&freeGPUMemory, &totalMemory);

			saveExpRes(configs, finalDir,false,saveTrainingView!=0);
		}
	}

	free(lossesPerEpochs);
	cudaFree(lossesDevice);
	free(lossesHost);

	if(this->currentHierarchy==configs->minimumHierarchy)
	{
		int speicifyPath = 0;
		printf("Specify path?");
		scanf("%d",&speicifyPath);
		if(speicifyPath)
		{
			while(speicifyPath)
			{
				this->renderSpecifiedPath(configs);
				
				printf("Try again?");
				scanf("%d",&speicifyPath);
			}
		}
		speicifyPath = 0;
		if(configs->justModel)
		{
			// printf("Get metrics?");
			// scanf("%d",&speicifyPath);
			if(!speicifyPath)
			{
				return;
			}
			else
			{
				configs->testSkip = 1;
				printf("Getting metrics, set test skip to 1 automatically...\n");
				char saveDir[255] = "RenderOnlyRes";
				// configs->testSkip = 1;
				int renderTrainingView = 1;
				
				// printf("Render Training View?");
				// scanf("%d",&renderTrainingView);
				saveExpRes(configs, saveDir, false, renderTrainingView);
			}

		}
	}


}

void Render::saveExpRes
(
	Configuration* configs, const char* additionInfo, bool validatingRes, bool saveAll)
{
	char saveDir[512];
	char command[1024];
	sprintf(saveDir, "%s%shierachy%d%s", configs->optimizedResDirectory, PATH_CONCATENATOR, this->currentHierarchy, additionInfo);
	sprintf(command, "mkdir %s%s", saveDir, PATH_CONCATENATOR);

	system(command);

	if (validatingRes)
	{
		renderViews
		(
			saveDir, this->val_cameras, "validation",
			configs->validationCount, this->renderRes.width, this->renderRes.height
		);
	}

	if (saveAll)
	{
		renderViews
		(
			saveDir,
			this->train_cameras, "fittingRes",
			this->datasetSize, this->renderRes.width, this->renderRes.height
		);

		renderViews
		(
			saveDir,
			this->val_cameras, "validation",
			configs->validationCount, this->renderRes.width, this->renderRes.height
		);
	}
	//char modelSavePath[256], fileName[64];
	//if (configs->rayTracingSilhouetteInitial)
	//{
	//	this->scene->getEffectiveCount(this->scene);
	//	sprintf(modelSavePath, "%s%stextureModel.ply", saveDir, PATH_CONCATENATOR);
	//	this->scene->saveSelf2ply(modelSavePath);
	//}
	//this->scene->getEffectiveCount(this->scene, false);
	//sprintf(modelSavePath, "%s%sopacityModel.ply", saveDir, PATH_CONCATENATOR, fileName);
	//this->scene->saveSelf2ply(modelSavePath, false);

	renderViews
	(
		saveDir,
		this->test_cameras, "test",
		configs->testCount, this->renderRes.width, this->renderRes.height, configs->testSkip
	);
}

void Render::loadCameras(Configuration* config, const int i)
{
	this->freeCameras();

	double objectCenter[3] =
	{
		config->width / pow(2,i) / 2,
		config->height / pow(2,i) / 2,
		config->depth / pow(2,i) / 2
	};

	int targetImageSize[2] =
	{
		config->renderWidth,
		config->renderHeight
	};

	this->valCamerasCount = config->validationCount;
	this->testCamerasCount = config->testCount;

	this->train_cameras = Camera::readFromMyBlender
	(
		config->cameraPoseFile,
		objectCenter, targetImageSize,
		config->viewCount,
		config->cameraDistance
	);

	this->val_cameras = Camera::readFromMyBlender
	(
		config->validationPoseFile,
		objectCenter, targetImageSize,
		config->validationCount,
		config->cameraDistance
	);

	this->test_cameras = Camera::readFromMyBlender
	(
		config->novelViewPoseFile,
		objectCenter, targetImageSize,
		config->testCount,
		config->cameraDistance
	);
}

void Render::freeOptimizers()
{
	sceneOptimizer.freeSelf();
}

void Render::freeCameras()
{
	if (this->train_cameras)
	{
		for (int cameraIndex = 0; cameraIndex < this->datasetSize; cameraIndex++)
		{
			this->train_cameras[cameraIndex].freeSelf();
		}
		free(this->train_cameras);
	}

	if (this->val_cameras)
	{
		for (int cameraIndex = 0; cameraIndex < this->valCamerasCount; cameraIndex++)
		{
			this->val_cameras[cameraIndex].freeSelf();
		}
		free(this->val_cameras);
	}

	if (this->test_cameras)
	{
		for (int cameraIndex = 0; cameraIndex < testCamerasCount; cameraIndex++)
		{
			this->test_cameras[cameraIndex].freeSelf();
		}
		free(this->test_cameras);
	}

}

void Render::renderViews
(
	const char* saveLearnedSceneDirectory,
	Camera* testCameras,
	const char* viewName, const int viewNums,
	const int renderWidth, const int renderHeight, const int skip)
{
	char novelViewDirectory[255];
	sprintf(novelViewDirectory, "%s%s%s%s", saveLearnedSceneDirectory, PATH_CONCATENATOR, viewName, PATH_CONCATENATOR);
	char command[256] = "mkdir ";
	sprintf(command, "%s%s", command, novelViewDirectory);
	system(command);

	Render::travelScene
	(
		*this->scene,
		novelViewDirectory,
		viewNums, &this->renderRes,
		testCameras, skip
	);
	printf("%s saved successfully\n", novelViewDirectory);
}

void Render::renderSpecifiedPath(Configuration* configs)
{
	double cameraPos[3],lineStart[3],lineEnd[3];
	const char* coordinates = "XYZ";

	char savePath[256],directoryName[256];
	printf("Specify save Dirctory:");
	scanf("%s",directoryName);
	sprintf(savePath,"%s%sSpecifiedRenderPath_%s",configs->resBaseDirectory,PATH_CONCATENATOR,directoryName);
	char command[1024];
	sprintf(command, "mkdir %s", savePath);
	system(command);

	printf("Specify rotate Center:\n");
	for(int i = 0;i<3;i++)
	{
		printf("Specify rotate Center's %c:",coordinates[i]);
		scanf("%lf",&lineStart[i]);
	}

	printf("Specify this line's another point:\n");
	for(int i = 0;i<3;i++)
	{
		printf("Specify this line's another point's %c:",coordinates[i]);
		scanf("%lf",&lineEnd[i]);
	}

	printf("Specify camera pos:\n");
	for(int i = 0;i<3;i++)
	{
		printf("Specify camera pos's %c:",coordinates[i]);
		scanf("%lf",&cameraPos[i]);
	}

	int viewNums;
	printf("Speicfy view nums:");
	scanf("%d",&viewNums);

	double objectCenter[3] = {this->scene->sceneWidth/2.,this->scene->sceneHeight/2.,this->scene->sceneDepth/2.},rotatedCameraPose[3];
	Camera* cameras = (Camera*)malloc(sizeof(Camera)*viewNums);
	double alpha[2] = { this->train_cameras[0].alpha[0],this->train_cameras[0].alpha[1] };
	double principle[2] = {this->train_cameras[0].principle[0],this->train_cameras[0].principle[1] };
	
	for(int i = 0;i<viewNums;i++)
	{
		GetRotatedPointAroundAxis(cameraPos,rotatedCameraPose,lineStart,lineEnd,i * 360./viewNums);
		for (int axis = 0; axis < 3; axis++)
		{
			//To make full use of Voxel...
			//resize first
			rotatedCameraPose[axis] *= (objectCenter[axis] / configs->cameraDistance);
			//then translate
			rotatedCameraPose[axis] += objectCenter[axis];
		}

		cameras[i].zReversed = this->train_cameras->zReversed;
		cameras[i].setCameraModelParams(alpha,principle,rotatedCameraPose,rotatedCameraPose,objectCenter);
		cameras[i].formulateProjectionMat();
		cameras[i].transferSelf2GPU();

	}


	Render::travelScene(*this->scene,savePath,viewNums,&this->renderRes,cameras);

	objectCenter[0] = 0.;
	objectCenter[1] = 0.;
	objectCenter[2] = 0.;

	for(int i = 0;i<viewNums;i++)
	{
		GetRotatedPointAroundAxis(cameraPos,rotatedCameraPose,lineStart,lineEnd,i * 360./viewNums);
		cameras[i].zReversed = false;
		cameras[i].setCameraModelParams(alpha,principle,rotatedCameraPose,rotatedCameraPose,objectCenter);
		cameras[i].formulateProjectionMat();
		cameras[i].transferSelf2GPU();

	}
	Camera::writeJasonFile(cameras,savePath,viewNums,"selfDefinePath");
	for(int i = 0;i<viewNums;i++)
	{
		cameras[i].freeSelf();
	}

}

void Render::travelScene
(
	ObjectScene scene,
	const char* saveDirectory,
	const int totalImages,
	Image* renderRes,
	Camera* cameras, const int skip
)
{
	char saveFileDirectory[256];
	for (int i = 0; i < totalImages; i += skip)
	{
		sprintf(saveFileDirectory, "%s//_%d", saveDirectory, i);

		RenderUtils::renderImage(*renderRes, cameras[i], scene);
		cudaDeviceSynchronize();
		renderRes->copy2HostMemory();
		renderRes->saveSelf(saveFileDirectory);
		printf("%d finished\r", i);
		fflush(stdout);
	}
	printf("\n");
}
