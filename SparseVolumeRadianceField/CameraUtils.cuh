#pragma once
#include "Utils.cuh"

typedef struct Camera
{
	bool zReversed = true;
	double alpha[2] = { 0 }, principle[2] = { 0 }, skew, intrinsicMat[9] = { 0 };
	double translation[3] = { 0 }; // camera center position, camera to world translate transform vector
	double rotation[3] = { 0 };
	double* rotation2 = 0;
	double rotationMat[9] = { 0 };// world to camera rotate transform
	double w2cExternalMat[16] = { 0 };// world to camera transform
	double c2wExternalMat[16] = { 0 };// camera to world transform
	double projectionMatHost[12] = { 0. };// world to image pixel projection matrix

	double M[9] = { 0. }, invertMHost[9] = { 0. }, p_4Host[3] = { 0. };
	double* c2wExternalMatDevice;
	double* principleDevice, * alphaDevice;
	double* invertMDevice = 0;//invert M in device
	double* p_4Device = 0;// the fourth column of projectionMatHost in device
	double cameraCenterHost[4]; // camera center
	double* cameraCenterDevice = 0; //camera center
	double* projectionMatDevice = 0; //world to image pixel projection matrix
	/*
	@param alphas: alpha
	@param principles:
	@param cameraTranslations:
	@param cameraRotations:
	@param cameraRotationsDirection:
	*/
	void setCameraModelParams(
		double alphas[2],
		double principles[2],
		double cameraTranslations[3],
		double cameraRotations[3],
		double* cameraRotations2 = 0,
		double skew = 0.,
		double* projectionMat = 0);
	void formulateProjectionMat();
	/*
	@param imageNums:
	@param rotateAxis:
	@param scaleRatioX:
	@param scaleRatioY:
	@param translationX:
	@param translationY:
	@param translationZ:
	@param sceneWidth:
	@param sceneHeight:
	@param sceneDepth:
	*/
	static Camera* setUpCameras(
		const int imageNums,
		int rotateAxis,
		double scaleRatioX,
		double scaleRatioY,
		double translationX,
		double translationY,
		double translationZ,
		int width = 640, int height = 480, int depth = 512);
	static Camera* initialSelfFromBlenderJsonFile(const char* fileName, const int cameraNums = 0);
	static void writeJasonFile(Camera* cameras, const char* destFilePath, const int cameraNums, const char* types);
	static Camera* readFromMyBlender(
		const char* fileName,
		double* objectCenter,
		int* targetImageSize,
		const int cameraNums = 84,
		double cameraDistance = 50.,
		bool needReScale = true,
		bool needReverseZ = true
	);
	static Camera* readFromDTUDataset(
		const char* cameraFileName,
		int* targetImageSize,
		const int cameraNums = 84
	);

	static void sortCameraByZPos(Camera* cameras, int* idx, const int cameraNums);
	static void sortCameras(Camera* cameras, int* idx, const int cameraNums);
	
	void transferSelf2GPU();
	void freeSelf();
	void printSelfInfo();
}Camera;