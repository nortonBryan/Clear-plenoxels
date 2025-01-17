#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cuda_runtime_api.h"
#include <curand_kernel.h>
#include <curand.h>
/*
项目-属性-链接器-输入-附加依赖项中添加一项"curand.lib"
*/
#include <curand_mtgp32_host.h>
#include <cassert>

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include <unistd.h>
#include "iostream"

#define PI 3.141592653589793

#ifdef __linux__
const char PATH_CONCATENATOR[3] = "/";
#else
const char PATH_CONCATENATOR[3] = "\\";
#endif // !PATH_CONCATENATOR

enum OptimizeMethods
{
	NoneOptimize = 0,
	NaiveSGD = 1,
	MomentumSGD = 2,
	NesterovSGD = 3,
	AdaGrad = 4,
	RMSprop = 5,
	Adam = 6
};

typedef unsigned char uchar;

#define INFINITY 1.e10

typedef struct
{
	int width;
	int height;
	int depth;
	int renderWidth;
	int renderHeight;
	bool graylized;
	int sphericalHarmonicsBand;
	bool hierarchyBandStrategy;
	bool SHsForTargetResoOnly;
	int hierarchy;
	int minimumHierarchy;
	int toSparseHierarchy;
	bool silhouetteBound;
	bool silhouetteInitial;
	bool MaskAreaOnly;
	int silhouetteRadius;
	bool initialFromFile;
	bool upsampled;
	char initialFileDirBase[512];
	int availableSHBand;
	bool renderOnly;
	char viewsDirectory[512];
	char cameraPoseFile[512];
	char validationPoseFile[512];
	char novelViewPoseFile[512];
	double cameraDistance;
	int viewCount;
	int validationCount;
	int testCount;
	int trainProcessViewIndex;
	int testSkip;
	bool whiteBG;
	int repeatLayers;
	int randomSeed;
	double learningRate;
	double shLearningRate;
	double decayRate;
	int densityOptimizer;
	int colorChannelOptimizer;
	double momentumCoefficient;
	int epochs;
	int validationInterval;
	int batchSize;
	bool fixBatchSize;
	char optimizedResDirectory[512];
	char silhouetteResDirectory[512];
	char resBaseDirectory[512];
	int saveIntervals;
	int halfLRIntervals;
	int pruneEpoch;
	float pruneThreshold;
	double lowestLR;
	int skips;
	int* skipIndex;
	float stepSize;
	bool solidStrategy;
	float initialDensity;
	float initialColor;
	int varianceLossInterval;
	double varianceCoefficient;
	double shCoefficient;
	double maskLossCoefficient;
	int vairanceType;//0 for tv loss, 1 for voxel disparity loss
	int neighboorNums;//6,26 only for tvLoss,default 6
	double diffuseRatio;
	bool exportModel;
	bool shuffleRays;
	char startTime[512];
	bool justModel = false;
	float huberDelta= 0.1f;
}Configuration;

__device__ __host__ void printArray(double* arrays, const int rows, const int cols, const char* description);

template <typename T>
void __device__ __host__
embedCopy(
	T* dest, T* src,
	int destRow, int destCol, int srcRow, int srcCol,
	int rowOffset = 0, int colOffset = 0
);

template <typename T>
void __host__
arrayAppend(
	T*& matrix,
	T* appendArray,
	int matRow, int matCol,
	int axis = 0, int appendPosi = -1
);

template<typename T>
void __device__ __host__
flatternMat(T** matrix, T* dest, int matRow, int matCol);

//__device__ __host__ inline double minf(double x1, double x2);
//__device__ __host__ inline double maxf(double x1, double x2);

//__device__ __host__ inline float sigmoid(float x);
//__device__ __host__ inline float sigmoid_derivative(float x);
//
//__device__ __host__ inline float colors_activation(float x);
//__device__ __host__ inline float colors_derivation(float x);
//
//__device__ __host__ inline float density_activation(float x);
//__device__ __host__ inline float density_derivation(float x);

void __device__ __host__
swapRowForMat(double* mat, int row1, int row2, int rows, int cols);

void __device__ __host__
setMatIdentity(double* src, int rows, int cols);

bool __device__ __host__
tinyMatMul(double* left, double* right, double* res, int leftRows, int leftCols, int rightRows, int rightCols);

void __device__ __host__
eulerAngle2RotationMat(double angles[3], double* rotationMat);

bool __host__
getMatInvert(double* src, double* dest, int rows);

double __device__ __host__
getEuclidDistance(double* data1, double* data2, const int dimension);

double __device__ __host__
getMSEError(double* losses, int len);

void __host__
GetRotatedPointAroundAxis(double* point, double* res, double* axisLineFirstPoint, double* axisLineSecondPoint, double rotateAngle);

void __device__ __host__
normalize(double* arrays, int length);

void __device__ __host__
crossProduct(double* src1, double* src2, double* dest);

template<typename T>
void arrayMulScalar(T* src, int rows, int cols, T defaultScalar);

void __host__
setCameraLookAtXOY(double* cameraPoint, double* objectPoint, double* rotateMat);

void setCameraLookAt(double* cameraPoint, double* objectPoint, double* rotateMat);

template<typename T, typename T2>
void stackArray(T* src, T2* dest, int rows, int cols, int depth);

template<typename T>
void setArrayValue(T* src, int rows, int cols, T defaultValue = 0);

void ShuffleRank(int* dest, int maxIndex, bool regenerate = true, int minIndex = 0, const bool shuffle = true);

void readMask(uchar*& arrays, int rows, int cols, const char* fileName, int* bounds);

void readUcharFromFile(uchar*& arrays, int rows, int cols, const char* fileName);
void saveUchar2File(uchar* arrays, int rows, int cols, const char* fileName);

void readIntFromFile(unsigned int*& arrays, int rows, int cols, const char* fileName);
void saveInt2File(unsigned int* arrays, int rows, int cols, const char* fileName);

void readFloatersFromFile(float*& arrays, int rows, int cols, const char* fileName);
void saveFloaters2File(float* arrays, int rows, int cols, const char* fileName, bool append = false,bool needProgress = false);

void saveDouble2File(double* arrays, int rows, int cols, const char* fileName, bool append = false);

void checkGPUStatus(const char* whereCalled, bool printSuccessInfo = false);

void printGPUMemoryInfo();

void getCurrentTime(char* currentTime);

Configuration* readConfig(const char* configFile);

/*
* rand by different types
@param randType:0:uniform,1:normal
*/
void randGPU(double* sampleIndexGPUPtr, const int len, const float sigma, const int randType=0);

bool isSquareNumber(int i);