#pragma once
#include "CameraUtils.cuh"

typedef struct objectScene
{
	int sceneWidth;
	int sceneHeight;
	int sceneDepth;
	int maxSamplePoint;

	int sh_dim = 1;

	int bbox_width_min, bbox_width_max;
	int bbox_height_min, bbox_height_max;
	int bbox_depth_min, bbox_depth_max;

	double* offsetXDevice, * offsetYDevice, * offsetZDevice;
	bool grayImage = true;
	bool sparse = false;
	int effectiveVoxel = 0;
	bool solid;
	int actualGridNums = 0;

	int optimizeStep = 0;
	int optimizingBand = 0;
	uchar* occupancyDevice = 0, * occupancyHost = 0;

	float* adaptiveLRDevice = 0;

	float* sceneOpacityHost = 0;
	float* sceneOpacityDevice = 0;

	float* AmbientGrayHost = 0;
	float* AmbientGrayDevice = 0;

	float* DiffuseGrayHost = 0;
	float* DiffuseGrayDevice = 0;

	float stepSize = 1.;

	float** AmbientRedHost = 0;
	float** AmbientRedDevice = 0;

	float** AmbientGreenHost = 0;
	float** AmbientGreenDevice = 0;

	float** AmbientBlueHost = 0;
	float** AmbientBlueDevice = 0;

	float** totalVariableGPUPtrs = 0, **totalVariableGPUPtrsDevice = 0;

	unsigned int* indexOffsetsDevice = 0, * indexOffsetsHost = 0;
	unsigned char* inCageInfoDevice = 0, * inCageInfoHost = 0;
	unsigned int inCageNums = 0;

	void initSelf
	(
		int width = 64, int height = 64, int depth = 64,
		bool graylized = true, bool solid = false,
		float initialDensity = 0.00001, float initialColor = 0.00001,
		float gridStepSize = 0.5, const int sh_dim = 9,
		const int cagedNums = 0
	);

	void initSelfFromStackImage(const char* imageName, int width, int height, int depth, bool graylized = true);
	void initSelfByUpsample
	(
		objectScene src, 
		float threshold, 
		const bool lowBandOnly = false,
		const bool toSparse = false,
		const int targetSH_dim = 1
	);
	
	void initFromFile
	(
		const char* sceneDirectory,
		int width = 640, int height = 480, int depth = 512,
		bool graylized = true, float gridStepSize = 0.5,
		const int sh_dim = 1,const int targetSH_Dim = 1,
		const bool sparse = false,const bool haveOccupancyInfo = false
	);

	void setSelfSampleOffset(float radius);

	void pruneSelf(float densityThreshold = 0.2);

	void transferSelf2Sparse(float densityThreshold = 7.f);

	void getCagedNums();
	
	void getEffectiveCount(objectScene scene, bool densityOnly = true, float threshold = 0.5f);
	void copy2HostMemory(const int shDims = 1);
	void saveSelf2ply(const char* fileName, bool densityOnly = true, float threshold = 0.5f);
	void saveSelf(const char* saveDirectory, bool densityOnly = true, const int band = 0,const bool hierarchyMode = false);
	void freeSelf();

}ObjectScene;