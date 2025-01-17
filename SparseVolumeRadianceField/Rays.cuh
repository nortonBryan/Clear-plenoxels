#pragma once

#include "Utils.cuh"
#include "ImageUtils.cuh"
#include "CameraUtils.cuh"
#include "SceneUtils.cuh"

typedef struct _rays
{
	int totalRays = 0;
	bool graylized;
	bool whiteBKGD;
	bool maskRaysOnly;
	//800*800*100/1024/1024=61.035MB
	//61.035 * (4*11+3)=2868.652MB

	float* cameraCenterX_Device = 0;
	float* cameraCenterY_Device = 0;
	float* cameraCenterZ_Device = 0;

	float* directionX_Device = 0;
	float* directionY_Device = 0;
	float* directionZ_Device = 0;

	float* near_t_Device = 0;
	float* far_t_Device = 0;
	//8

	float* accumulatedGray_Device = 0;

	float* accumulatedRed_Device = 0;
	float* accumulatedGreen_Device = 0;
	float* accumulatedBlue_Device = 0;
	//11

	float* accumulatedDensity_Device = 0;
	float* accumulatedWeights_Device = 0;
	//13

	uchar* gtMask_Device = 0;
	float* gtGray_Device = 0;
	uchar* gtRed_Device = 0;
	uchar* gtGreen_Device = 0;
	uchar* gtBlue_Device = 0;
	//17

	void allocateSelfMemory();
	void initialSelf
	(
		Camera* cameras, Image* groundTruths,
		ObjectScene scene, const int datasetSize = 100,
		bool masksRaysOnly = false, bool depthPriorLearned = false, bool whiteBKGD = false, int* order = 0
	);

	void shuffleTo(_rays* dest, int* shuffledIndexArray);
	void freeSelf();
}Rays;