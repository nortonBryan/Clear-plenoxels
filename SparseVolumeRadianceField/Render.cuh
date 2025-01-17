#pragma once
#include "RenderUtils.cuh"

class Render
{
public:
	Image renderRes;
	Image* groundTruth = 0;
	Camera* train_cameras = 0;
	Camera* val_cameras = 0;
	int valCamerasCount = 100;
	Camera* test_cameras = 0;
	int testCamerasCount = 200;

	bool grayLized = true;
	int datasetSize;
	ObjectScene* scene;

	_rays ray;
	Optimizer sceneOptimizer;

	int currentHierarchy;

	int trainingProcessIdx = 0;
	int* sampleRaysIndexGPU = 0, * sampleRaysIndexHost = 0;
	~Render();

	int* idx = 0;

	/*
	load cameras from configuration file info
	*/
	Render(Configuration* config,const int hierarchy,int* idx);

	void shuffleRays(const int maxIndex, bool initialized = false,const bool shuffle = true);

	void setDataset(Image* groundTruth);

	void setScene
	(
		ObjectScene& scene, bool initialOptimizer = true,
		OptimizeMethods densityOptimizer = Adam, OptimizeMethods colorChannelOptimizer = Adam
	);
	void objectSceneLearner(Configuration* configs);
	void regenerateSamples();

	void gatherInfoAlongARay(Configuration* configs, const int viewIndex);

	void saveExpRes
	(
		Configuration* configs,
		const char* additionInfo,
		bool validatingRes = false,bool saveAll = false
	);

	void renderSpecifiedPath(Configuration* configs);

	void renderViews
	(
		const char* saveBaseDir,
		Camera* cameras,
		const char* viewName,
		const int viewNums = 120,
		const int renderWidth = 640, const int renderHeight = 480, const int skip = 1
	);

	void loadCameras(Configuration* configs, const int hierarchy);

	void freeOptimizers();
	void freeCameras();

	static void travelScene
	(
		ObjectScene scene,
		const char* saveDirectory,
		const int totalImages = 360,
		Image* renderRes = 0,
		Camera* cameras = 0, const int skip = 1
	);
};
