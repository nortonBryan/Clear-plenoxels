#pragma once
#include "SceneUtils.cuh"
#include "ImageUtils.cuh"
#include "Optimizer.cuh"
#include "Rays.cuh"

class RenderUtils
{
public:
	static double bbox_silhouette
	(
		Camera* camera, Image* groundTruth, 
		ObjectScene* scene, const int datasetSize, 
		const int radius = 1,bool isGettingScaleRatio = false,int* skipIndex = 0,int skipNums = 0
	);
	
	static void renderImage(Image renderRes, Camera camera, ObjectScene scene);
	static void renderImage_withGradients_Gray
	(
		Optimizer density,
		Optimizer ambientGray,
		Image renderRes, Image groundTruth,
		Camera camera, ObjectScene scene,
		double* losses
	);
	static void renderImage_withGradients_RGB
	(
		Optimizer density,
		Optimizer redChannel,
		Optimizer greenChannel,
		Optimizer blueChannel,
		Image renderRes, Image groundTruth,
		Camera camera, ObjectScene scene,
		double* losses
	);

	static void renderRays
	(
		Rays rays, ObjectScene scene, 
		int* sampleIndex, const int startSampleIndex, const int len
	);

	static void renderRays_withGradients_Gray
	(
		Optimizer density, Optimizer ambientGray,
		Rays ray, ObjectScene scene,
		double* losses,
		int* sampleIndex, const int startSampleIndex, const int len
	);

	static void renderRays_withGradients_RGB
	(
		Optimizer sceneOptimizer,
		Rays ray, ObjectScene scene,
		double* losses,
		int* sampleIndex, const int startSampleIndex, const int len,const double maskRayLossCoefficient = 5.e-5,
		const float huberDelta = 0.01
	);

	static void boundRays
	(
		Rays rays, ObjectScene scene,
		int* sampleIndex, const int startSampleIndex, const int len, const double threshold
	);

	static float* gatherDensitiesAlongRay(Camera camera, ObjectScene scene, int* effectivePts);
	static void pruneSceneByWeights(Rays ray, ObjectScene scene, const double threshold = 0.01);
	/*
	@param scene:
	@param sceneOptimizer:
	@param varianceCoefficient:
	@param shCoefficient:
	@param varianceType:0 for tv loss, 1 for voxel disparity loss
	@param neighboorNums:6,26 only,default 6
	*/
	static void addVarianceLoss
	(
		ObjectScene scene,
		Optimizer sceneOptimizer,
		double varianceCoefficient = 0.001,
		double shCoefficient = 0.001,
		const int varianceType = 0,
		const int neighboorNums = 6
	);
};