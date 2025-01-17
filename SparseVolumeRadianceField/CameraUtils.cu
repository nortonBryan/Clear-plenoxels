#include "CameraUtils.cuh"

void Camera::setCameraModelParams
(
	double alphas[2],
	double principles[2],
	double cameraTranslations[3],
	double cameraRotations[3],
	double* cameraRotations2,
	double Skew,
	double* externalMat_w2cRotation
) {
	embedCopy(alpha, alphas, 2, 1, 2, 1);
	embedCopy(principle, principles, 2, 1, 2, 1);
	embedCopy(translation, cameraTranslations, 3, 1, 3, 1);
	embedCopy(rotation, cameraRotations, 3, 1, 3, 1);
	rotation2 = cameraRotations2;
	skew = Skew;

	double K[3][3] =
	{
		{alpha[0], skew, principle[0]},
		{0., alpha[1], principle[1]},
		{0., 0., 1.}
	}, * kPtr[3];
	for (int i = 0; i < 3; i++)
	{
		kPtr[i] = K[i];
	}
	flatternMat(kPtr, intrinsicMat, 3, 3);

	if (externalMat_w2cRotation == 0)
	{
		if (rotation2 == 0)
		{
			eulerAngle2RotationMat(rotation, rotationMat);
		}
		else
		{
			setCameraLookAt(rotation, rotation2, rotationMat);
		}
	}
}

void Camera::formulateProjectionMat()
{
	double zeros[4] = { 0., 0., 0., 0. };

	// step 1 formulate K[I|0]
	double* K_I_0 = (double*)malloc(sizeof(double) * 9);
	embedCopy(K_I_0, intrinsicMat, 3, 3, 3, 3);
	arrayAppend(K_I_0, zeros, 3, 3, 1, -1);
#if DEBUG
	printArray(K_I_0, 3, 4, "intrinsic mat append null vector");
#endif

	// step2 formulate w2cExternalMat
	// step2.1 rotationMat
	double* temp = (double*)malloc(sizeof(double) * 9);
	embedCopy(temp, rotationMat, 3, 3, 3, 3);

	// step2.1.2 openGL2opencv
	if (zReversed)
	{
		double openGL2OpenCVRotateAngle[3] = { 180.,0.,0. }, openGL2OpenCVRotateMat[9];
		eulerAngle2RotationMat(openGL2OpenCVRotateAngle, openGL2OpenCVRotateMat);
		double finalRotateMat[9];
		tinyMatMul(openGL2OpenCVRotateMat, temp, finalRotateMat, 3, 3, 3, 3);
		embedCopy(temp, finalRotateMat, 3, 3, 3, 3);
	}


#if DEBUG
	printArray(temp, 3, 3, "rotationMat in temp");
#endif
	arrayAppend<double>(temp, zeros, 3, 3, 1, -1);
#if DEBUG
	printArray(temp, 3, 4, "temp after append translation");
#endif
	double lastOne[4] = { 0., 0.,0.,1. };
	arrayAppend<double>(temp, lastOne, 3, 4, 0, -1);
	// step2.2 translationMat
	double w2cTranslation[3] = { -translation[0], -translation[1], -translation[2] };
	double w2cTranslationMat[16];
	setMatIdentity(w2cTranslationMat, 4, 4);
	embedCopy(w2cTranslationMat, w2cTranslation, 4, 4, 3, 1, 0, 3);
	tinyMatMul(temp, w2cTranslationMat, w2cExternalMat, 4, 4, 4, 4);
#if DEBUG
	printArray(temp, 4, 4, "temp after append null row");
#endif
#if DEBUG
	printArray(c2wExternalMat, 4, 4, "Whole external Matrix");
#endif
	getMatInvert(w2cExternalMat, c2wExternalMat, 4);

	// step3 formulate projection mat = K[I|0]*[R|t] 
	// that project point in world coordinate to point in image pixel coordinate
	tinyMatMul(K_I_0, w2cExternalMat, projectionMatHost, 3, 4, 4, 4);
#if DEBUG
	printArray(w2cExternalMat, 4, 4, "external mat interval");
	printArray(projectionMatHost, 3, 4, "Projection matrix");
#endif

	free(K_I_0);
	free(temp);
}

Camera* Camera::setUpCameras(
	const int imageNums,
	int rotateAxis,
	double focalX,
	double focalY,
	double translationX,
	double translationY,
	double translationZ,
	int Width, int Height, int Depth)
{
	double alphas[2] = { focalX, focalY };
	double principles[2] = { Width / 2., Height / 2. };
	double translations[3] = { translationX, translationY, translationZ };

	double rotationAngles[3] = { 0.,0.,0. };
	double rotation2[3] = { -1.,0.,0. };

	double translatedPoints[3];
	double rotateLinePt1[3] = { Width / 2., Height / 2., Depth / 2. };
	double rotateLinePt2[3] = { Width / 2., Height / 2., Depth / 2. };
	if (rotateAxis == 0)
	{
		rotateLinePt2[0] = Width;
	}
	else if (rotateAxis == 1)
	{
		rotateLinePt2[1] = Height;
	}
	else if (rotateAxis == 2)
	{
		rotateLinePt2[2] = Depth;
	}


	double worldCenter[3] = { Width / 2., Height / 2., Depth / 2. };

	Camera* cameras = (Camera*)malloc(sizeof(Camera) * imageNums);
	for (int i = 0; i < imageNums; i++)
	{
		GetRotatedPointAroundAxis
		(
			translations,
			translatedPoints,
			rotateLinePt1,
			rotateLinePt2,
			i * 360 / imageNums
		);

		cameras[i].setCameraModelParams
		(
			alphas,
			principles,
			translatedPoints,
			translatedPoints,
			worldCenter
		);
		cameras[i].zReversed = true;
		cameras[i].formulateProjectionMat();
		cameras[i].transferSelf2GPU();
		//printf("angle:%d\n", i * (360 / imageNums));
		//cameras[i].printSelfInfo();

		cudaDeviceSynchronize();
		checkGPUStatus("Initialize Cameras");
	}

	return cameras;
}

Camera* Camera::initialSelfFromBlenderJsonFile(const char* fileName, const int cameraNums)
{
	FILE* blendCameraParams = fopen(fileName, "r");
	if (!blendCameraParams)
	{
		printf("Can not find %s!\n", fileName);
		exit(-1);
	}
	Camera* cameras = (Camera*)malloc(sizeof(Camera) * cameraNums);
	char temp[512];
	// skip first row
	fscanf(blendCameraParams, "%[^\n]", temp);
	double cameraKernel[9];
	double rotateMat[9];
	double translateMat[3];

	for (int cameraIndex = 0; cameraIndex < cameraNums; cameraIndex++)
	{
		fscanf(blendCameraParams, "  \"%d\": \{\"cam_K\": [", temp);
		//fscanf(cameraFiles, "%[^][",temp);
		for (int i = 0; i < 8; i++)
		{
			fscanf(blendCameraParams, "%lf, ", &cameraKernel[i]);
		}
		fscanf(blendCameraParams, "%lf], \"cam_R_w2c\": [", &cameraKernel[8]);
		for (int i = 0; i < 8; i++)
		{
			fscanf(blendCameraParams, "%lf, ", &rotateMat[i]);
		}
		fscanf(blendCameraParams, "%lf], \"cam_t_w2c\": [", &rotateMat[8]);
		for (int i = 0; i < 2; i++)
		{
			fscanf(blendCameraParams, "%lf, ", &translateMat[i]);
		}
		fscanf(blendCameraParams, "%lf]%[^\n]", &translateMat[2]);
		embedCopy<double>(cameras[cameraIndex].intrinsicMat, cameraKernel, 3, 3, 3, 3);
		embedCopy<double>(cameras[cameraIndex].rotationMat, rotateMat, 3, 3, 3, 3);
		embedCopy<double>(cameras[cameraIndex].translation, translateMat, 3, 1, 3, 1);
		cameras[cameraIndex].formulateProjectionMat();
		cameras[cameraIndex].transferSelf2GPU();
		cameras[cameraIndex].printSelfInfo();
	}

	fclose(blendCameraParams);
	return cameras;
}

void Camera::writeJasonFile(Camera* cameras, const char* destFilePath, const int cameraNums, const char* types)
{
	char fileName[256];
	sprintf(fileName, "%s%stransforms_%s.json", destFilePath,PATH_CONCATENATOR, types);
	FILE* file = fopen(fileName, "w");

	//double camera_angle_x = 0.9464029168768066;
	double camera_angle_x = 2. * atan(cameras[0].principle[0] / cameras[0].alpha[0]);
	fprintf(file, "{\n\t\"camera_angle_x\": %.16lf,\n\t\"frames\": [\n", camera_angle_x);
	for (int cameraIndex = 0; cameraIndex < cameraNums - 1; cameraIndex++)
	{
		fprintf(file, "\t\t{\n\t\t\t\"file_path\": \"./%s/r_%d\",\n", types, cameraIndex);
		fprintf(file, "\t\t\t\"transform_matrix\": [\n");
		for (int rows = 0; rows < 3; rows++)
		{
			fprintf(file, "\t\t\t\t[\n\t\t\t\t\t%.16lf,\n\t\t\t\t\t%.16lf,\n\t\t\t\t\t%.16lf,\n\t\t\t\t\t%.16lf\n\t\t\t\t],\n",
				cameras[cameraIndex].c2wExternalMat[rows * 4],
				cameras[cameraIndex].c2wExternalMat[rows * 4 + 1],
				cameras[cameraIndex].c2wExternalMat[rows * 4 + 2],
				cameras[cameraIndex].c2wExternalMat[rows * 4 + 3]);
		}
		fprintf(file, "\t\t\t\t[\n\t\t\t\t\t%.16lf,\n\t\t\t\t\t%.16lf,\n\t\t\t\t\t%.16lf,\n\t\t\t\t\t%.16lf\n\t\t\t\t]\n\t\t\t]\n\t\t},\n",
			cameras[cameraIndex].c2wExternalMat[3 * 4],
			cameras[cameraIndex].c2wExternalMat[3 * 4 + 1],
			cameras[cameraIndex].c2wExternalMat[3 * 4 + 2],
			cameras[cameraIndex].c2wExternalMat[3 * 4 + 3]);
	}
	fprintf(file, "\t\t{\n\t\t\t\"file_path\": \"./%s/r_%d\",\n", types, cameraNums - 1);
	fprintf(file, "\t\t\t\"transform_matrix\": [\n");
	for (int rows = 0; rows < 3; rows++)
	{
		fprintf(file, "\t\t\t\t[\n\t\t\t\t\t%.16lf,\n\t\t\t\t\t%.16lf,\n\t\t\t\t\t%.16lf,\n\t\t\t\t\t%.16lf\n\t\t\t\t],\n",
			cameras[cameraNums - 1].c2wExternalMat[rows * 4],
			cameras[cameraNums - 1].c2wExternalMat[rows * 4 + 1],
			cameras[cameraNums - 1].c2wExternalMat[rows * 4 + 2],
			cameras[cameraNums - 1].c2wExternalMat[rows * 4 + 3]);
	}
	fprintf(file, "\t\t\t\t[\n\t\t\t\t\t%.16lf,\n\t\t\t\t\t%.16lf,\n\t\t\t\t\t%.16lf,\n\t\t\t\t\t%.16lf\n\t\t\t\t]\n\t\t\t]\n\t\t}\n",
		cameras[cameraNums - 1].c2wExternalMat[3 * 4],
		cameras[cameraNums - 1].c2wExternalMat[3 * 4 + 1],
		cameras[cameraNums - 1].c2wExternalMat[3 * 4 + 2],
		cameras[cameraNums - 1].c2wExternalMat[3 * 4 + 3]);

	fprintf(file, "\t]\n}\n");
	fclose(file);
}

Camera* Camera::readFromMyBlender(
	const char* fileName,
	double* objectCenter,
	int* targetImageSize,
	const int cameraNums,
	double cameraDistance, bool needReScale, bool needReverseZ)
{
	FILE* cameraFiles = fopen(fileName, "r");
	if (!cameraFiles)
	{
		printf("Can not find %s!\n", fileName);
		exit(-1);
	}
	int datasetSize;
	fscanf(cameraFiles, "total %d\n", &datasetSize);
	if (cameraNums != 0)
	{
		datasetSize = cameraNums;
	}
	Camera* cameras = (Camera*)malloc(sizeof(Camera) * datasetSize);
	double cameraKernel[9];
	double translateMat[3];

	fscanf(cameraFiles, "%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n",
		&cameraKernel[0],
		&cameraKernel[1],
		&cameraKernel[2],
		&cameraKernel[3],
		&cameraKernel[4],
		&cameraKernel[5],
		&cameraKernel[6],
		&cameraKernel[7],
		&cameraKernel[8]);

	double scaleTransform[9];
	setMatIdentity(scaleTransform, 3, 3);

	if (fabs(cameraKernel[2] * 2 - targetImageSize[0]) > 1.e-8)
	{
		scaleTransform[0] = targetImageSize[0] / cameraKernel[2] / 2;
	}

	if (fabs(cameraKernel[5] * 2 - targetImageSize[1]) > 1.e-8)
	{
		scaleTransform[4] = targetImageSize[1] / cameraKernel[5] / 2;
	}
	double temp[9];
	tinyMatMul(scaleTransform, cameraKernel, temp, 3, 3, 3, 3);
	embedCopy(cameraKernel, temp, 3, 3, 3, 3);

	double alpha[2] = { cameraKernel[0] ,cameraKernel[4] };
	double principle[2] = { cameraKernel[2],cameraKernel[5] };

	int indexCheck;

	double radiusRatio;
	double sceneDiagnoalLen = sqrt(objectCenter[0] * objectCenter[0] + objectCenter[1] * objectCenter[1] + objectCenter[2] * objectCenter[2]);
	
	for (int i = 0; i < datasetSize; i++)
	{
		fscanf(cameraFiles, "%d:\n", &indexCheck);
		if (indexCheck != i)
		{
			printf("Error occured, file has been corruptted!\n");
			free(cameras);
			exit(0);
		}
		fscanf(cameraFiles, "%lf\n%lf\n%lf\n", &translateMat[0], &translateMat[1], &translateMat[2]);
		
		{
			fscanf(cameraFiles, "%lf %lf %lf %lf\n",
				&cameras[i].w2cExternalMat[0],
				&cameras[i].w2cExternalMat[1],
				&cameras[i].w2cExternalMat[2],
				&cameras[i].w2cExternalMat[3]);
			fscanf(cameraFiles, "%lf %lf %lf %lf\n",
				&cameras[i].w2cExternalMat[4],
				&cameras[i].w2cExternalMat[5],
				&cameras[i].w2cExternalMat[6],
				&cameras[i].w2cExternalMat[7]);
			fscanf(cameraFiles, "%lf %lf %lf %lf\n",
				&cameras[i].w2cExternalMat[8],
				&cameras[i].w2cExternalMat[9],
				&cameras[i].w2cExternalMat[10],
				&cameras[i].w2cExternalMat[11]);
			fscanf(cameraFiles, "%lf %lf %lf %lf\n",
				&cameras[i].w2cExternalMat[12],
				&cameras[i].w2cExternalMat[13],
				&cameras[i].w2cExternalMat[14],
				&cameras[i].w2cExternalMat[15]);
		}

		radiusRatio = sqrtf(translateMat[0] * translateMat[0] + translateMat[1] * translateMat[1] + translateMat[2] * translateMat[2]);

		if (needReScale && radiusRatio < 10)
		{
			for (int axis = 0; axis < 3; axis++)
			{
				//To make full use of Voxel...
				//resize first
				translateMat[axis] *= (objectCenter[axis] / cameraDistance);
				//then translate
				translateMat[axis] += objectCenter[axis];
			}
		}
		cameras[i].zReversed = needReverseZ;
		cameras[i].setCameraModelParams(alpha, principle, translateMat, translateMat, objectCenter);
		cameras[i].formulateProjectionMat();
		cameras[i].transferSelf2GPU();
		//printf("%d:\n", i);
		//cameras[i].printSelfInfo();
	}

	fclose(cameraFiles);
	return cameras;
}

Camera* Camera::readFromDTUDataset(const char* cameraFileName, int* targetImageSize, const int cameraNums)
{
	FILE* cameraFiles = fopen(cameraFileName, "r");
	if (!cameraFiles)
	{
		printf("Can not find %s!\n", cameraFileName);
		exit(-1);
	}
	int datasetSize;
	fscanf(cameraFiles, "total %d\n", &datasetSize);
	if (cameraNums != 0)
	{
		datasetSize = cameraNums;
	}
	Camera* cameras = (Camera*)malloc(sizeof(Camera) * datasetSize);

	double objectCenter[3] = { 320.,240.,256 };
	double scaleTransform[9];
	setMatIdentity(scaleTransform, 3, 3);

	double radiusRatio;
	int indexCheck;
	for (int i = 0; i < datasetSize; i++)
	{
		fscanf(cameraFiles, "%d:\n", &indexCheck);
		if (indexCheck != i)
		{
			printf("Error occured, file has been corruptted!\n");
			free(cameras);
			exit(0);
		}

		fscanf(cameraFiles, "IntrinsicMat:%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n",
			&cameras[i].intrinsicMat[0],
			&cameras[i].intrinsicMat[1],
			&cameras[i].intrinsicMat[2],
			&cameras[i].intrinsicMat[3],
			&cameras[i].intrinsicMat[4],
			&cameras[i].intrinsicMat[5],
			&cameras[i].intrinsicMat[6],
			&cameras[i].intrinsicMat[7],
			&cameras[i].intrinsicMat[8]
		);

		if (fabs(cameras[i].intrinsicMat[2] * 2 - targetImageSize[0]) > 1.e-8)
		{
			scaleTransform[0] = targetImageSize[0] / cameras[i].intrinsicMat[2] / 2;
		}

		if (fabs(cameras[i].intrinsicMat[5] * 2 - targetImageSize[1]) > 1.e-8)
		{
			scaleTransform[4] = targetImageSize[1] / cameras[i].intrinsicMat[5] / 2;
		}

		double temp[9];
		tinyMatMul(scaleTransform, cameras[i].intrinsicMat, temp, 3, 3, 3, 3);
		embedCopy(cameras[i].intrinsicMat, temp, 3, 3, 3, 3);

		cameras[i].skew = cameras[i].intrinsicMat[1];

		fscanf(cameraFiles, "ExternalMat:\n%lf %lf %lf %lf\n",
			&cameras[i].w2cExternalMat[0],
			&cameras[i].w2cExternalMat[1],
			&cameras[i].w2cExternalMat[2],
			&cameras[i].w2cExternalMat[3]);
		fscanf(cameraFiles, "%lf %lf %lf %lf\n",
			&cameras[i].w2cExternalMat[4],
			&cameras[i].w2cExternalMat[5],
			&cameras[i].w2cExternalMat[6],
			&cameras[i].w2cExternalMat[7]);
		fscanf(cameraFiles, "%lf %lf %lf %lf\n",
			&cameras[i].w2cExternalMat[8],
			&cameras[i].w2cExternalMat[9],
			&cameras[i].w2cExternalMat[10],
			&cameras[i].w2cExternalMat[11]);
		fscanf(cameraFiles, "%lf %lf %lf %lf\n",
			&cameras[i].w2cExternalMat[12],
			&cameras[i].w2cExternalMat[13],
			&cameras[i].w2cExternalMat[14],
			&cameras[i].w2cExternalMat[15]
		);
		fscanf(cameraFiles, "cameraCenter:\n%lf %lf %lf\n",
			&cameras[i].translation[0],
			&cameras[i].translation[1],
			&cameras[i].translation[2]
		);
		fscanf(cameraFiles, "RotationMat:\n%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n",
			&cameras[i].rotationMat[0],
			&cameras[i].rotationMat[1],
			&cameras[i].rotationMat[2],
			&cameras[i].rotationMat[3],
			&cameras[i].rotationMat[4],
			&cameras[i].rotationMat[5],
			&cameras[i].rotationMat[6],
			&cameras[i].rotationMat[7],
			&cameras[i].rotationMat[8]
		);
		double alpha[2] = { cameras[i].intrinsicMat[0],cameras[i].intrinsicMat[4] };
		double principle[2] = { cameras[i].intrinsicMat[2],cameras[i].intrinsicMat[5] };
		double translateMat[3] = {
			cameras[i].translation[0],
			cameras[i].translation[1],
			cameras[i].translation[2]
		};
		radiusRatio = sqrtf(translateMat[0] * translateMat[0] + translateMat[1] * translateMat[1] + translateMat[2] * translateMat[2]);

		if (radiusRatio < 10)
		{
			for (int axis = 0; axis < 3; axis++)
			{
				translateMat[axis] *= radiusRatio * 130.;
				translateMat[axis] += objectCenter[axis];
			}
		}
		cameras[i].setCameraModelParams(
			alpha,
			principle,
			translateMat,
			translateMat,
			0,
			cameras[i].intrinsicMat[1],
			cameras[i].rotationMat
		);
		cameras[i].zReversed = false;
		cameras[i].formulateProjectionMat();
		cameras[i].transferSelf2GPU();

		printf("%d:\n", i);
		cameras[i].printSelfInfo();
	}

	fclose(cameraFiles);
	return cameras;
}

void Camera::sortCameraByZPos(Camera* cameras, int* resIdx, const int cameraNums)
{
	Camera temp;
	double minZ;
	int tempIdx,minIdx;


	for (int i = 0; i < cameraNums; i++)
	{
		resIdx[i] = i;
	}
	// return;
	for (int turnsIdx = 0; turnsIdx < cameraNums - 1; turnsIdx++)
	{
		// minZ = cameras[turnsIdx].translation[2];
		minZ = atan2(cameras[turnsIdx].translation[1],cameras[turnsIdx].translation[0]);

		temp = cameras[turnsIdx];

		tempIdx = resIdx[turnsIdx];

		for (int cameraIdx = turnsIdx + 1; cameraIdx < cameraNums; cameraIdx++)
		{
			if (minZ > atan2(cameras[cameraIdx].translation[1],cameras[cameraIdx].translation[0]))
			// if(minZ>cameras[cameraIdx].translation[2])
			{
				minZ = atan2(cameras[cameraIdx].translation[1],cameras[cameraIdx].translation[0]);
				// minZ = cameras[cameraIdx].translation[2];
				minIdx = cameraIdx;
			}
			
		}
		
		cameras[turnsIdx] = cameras[minIdx];
		cameras[minIdx] = temp;

		resIdx[turnsIdx] = resIdx[minIdx];
		resIdx[minIdx] = tempIdx;
	}
}

void Camera::sortCameras(Camera* cameras, int* idx, const int cameraNums)
{
	Camera temp;
	for(int i = 0;i<cameraNums;i++)
	{
		temp = cameras[i];
		cameras[i] = cameras[idx[i]];
		cameras[idx[i]] = temp;

	}

}

void Camera::transferSelf2GPU()
{
	cudaMalloc((void**)&alphaDevice, sizeof(double) * 2);
	cudaMemcpy(alphaDevice, alpha, sizeof(double) * 2, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&principleDevice, sizeof(double) * 2);
	cudaMemcpy(principleDevice, principle, sizeof(double) * 2, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&projectionMatDevice, sizeof(double) * 12);
	cudaMemcpy(projectionMatDevice, projectionMatHost, sizeof(double) * 12, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&c2wExternalMatDevice, sizeof(double) * 16);
	cudaMemcpy(c2wExternalMatDevice, c2wExternalMat, sizeof(double) * 16, cudaMemcpyHostToDevice);

	embedCopy(M, projectionMatHost, 3, 3, 3, 4);
	getMatInvert(M, invertMHost, 3);
	embedCopy(p_4Host, projectionMatHost, 3, 1, 3, 4, 0, 3);

	cudaMalloc((void**)&invertMDevice, sizeof(double) * 9);
	cudaMemcpy(invertMDevice, invertMHost, sizeof(double) * 9, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&p_4Device, sizeof(double) * 3);
	cudaMemcpy(p_4Device, p_4Host, sizeof(double) * 3, cudaMemcpyHostToDevice);

	tinyMatMul(invertMHost, p_4Host, cameraCenterHost, 3, 3, 3, 1);
	cameraCenterHost[0] *= -1.;
	cameraCenterHost[1] *= -1.;
	cameraCenterHost[2] *= -1.;
	cameraCenterHost[3] = 1.0;
	cudaMalloc((void**)&cameraCenterDevice, sizeof(double) * 4);
	cudaMemcpy(cameraCenterDevice, cameraCenterHost, sizeof(double) * 4, cudaMemcpyHostToDevice);
}

void Camera::freeSelf()
{
	cudaFree(c2wExternalMatDevice);
	cudaFree(alphaDevice);
	cudaFree(principleDevice);
	cudaFree(projectionMatDevice);
	cudaFree(invertMDevice);
	cudaFree(p_4Device);
	cudaFree(cameraCenterDevice);
	checkGPUStatus("freeCamera");
}

void Camera::printSelfInfo()
{
	//printArray(intrinsicMat, 3, 3, "intrinsic params");
	//printArray(translation, 3, 1, "translations");
	//printArray(rotation, 3, 1, "rotations");
	//printArray(rotationMat, 3, 3, "rotation matrix");
	//printArray(c2wExternalMat, 4, 4, "external matrix invert");
	printArray(w2cExternalMat, 4, 4, "w2c external matrix");
	//printArray(projectionMatHost, 3, 4, "projection matrix");
	//printArray(M, 3, 3, "M");
	//printArray(invertMHost, 3, 3, "M invert");
	//printArray(p_4Host, 3, 1, "p_4Host");
}
