#pragma once
#include "Utils.cuh"
#include "SceneUtils.cuh"

typedef struct optimizer
{
	OptimizeMethods optimizeMethod;
	int variablesNum;
	int paramLen;
	bool hierarchyStrategy;

	int densityOptimizedStep;
	float** paramsDevice = 0;
	float** gradientDevice = 0;
	float** momentumDevice = 0;
	float** varianceDevice = 0;

	double* learningRate = 0;
	double* shLearningRate = 0;
	float momentum_gamma = 0.9;
	float adam_beta1 = 0.9;
	float adam_beta2 = 0.999;

	void initialSelf(const int paramsLen, OptimizeMethods methods, float** paramsPtr,const int totalVariableNum);
	void initialBand(const int optimizingBand,const bool densityInitialized=true);
	void initialAll();
	void setLearningRate(double* lr,double* shLR);
	void updateSelf(const int timeStep = 1);
	void freeSelf();

	static void updateGrayScene(
		optimizer density, optimizer ambientGray,
		ObjectScene scene, int timeStep);

	static void updateRGBScene(
		optimizer sceneOptimizer,
		ObjectScene scene, int timeStep);
}Optimizer;