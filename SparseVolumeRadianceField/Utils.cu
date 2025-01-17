#include "Utils.cuh"

__device__ __host__ void printArray(double* arrays, const int rows, const int cols, const char* description)
{
	printf("%s:\n", description);
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			printf("%.6f\t", arrays[row * cols + col]);
		}
		printf("\n");
	}
}

template <typename T>
void __device__ __host__ embedCopy(T* dest, T* src, int destRow, int destCol, int srcRow, int srcCol, int rowOffset, int colOffset)
{
	if (destCol < srcCol || destRow < srcRow)
	{
		for (int row = 0; row < destRow; row++)
		{
			for (int col = 0; col < destCol; col++)
			{
				dest[row * destCol + col] = src[(row + rowOffset) * srcCol + colOffset + col];
			}
		}
		return;
	}

	for (int row = 0; row < srcRow; row++)
	{
		for (int col = 0; col < srcCol; col++)
		{
			dest[(row + rowOffset) * destCol + col + colOffset] = src[row * srcCol + col];
		}
	}
}

template <typename T>
void __host__ arrayAppend(T*& matrix, T* appendArray, int matRow, int matCol, int axis, int appendPosi)
{
	int newMatrixSize = axis == 0 ? matCol * (matRow + 1) : (matRow * (matCol + 1));
	T* newMatrix = (T*)malloc(sizeof(T) * newMatrixSize);
	if (axis == 0)
	{
		embedCopy(newMatrix, matrix, matRow + 1, matCol, matRow, matCol);
		embedCopy(newMatrix, appendArray, matRow + 1, matCol, 1, matCol, 3, 0);
	}
	else
	{
		embedCopy(newMatrix, matrix, matRow, matCol + 1, matRow, matCol);
		for (int row = 0; row < matRow; row++)
		{
			newMatrix[row * (matCol + 1) + matCol] = appendArray[row];
		}
	}
	free(matrix);
	matrix = newMatrix;
}

template
void __host__ arrayAppend(double*& matrix, double* appendArray, int matRow, int matCol, int axis, int appendPosi);

template<typename T>
void __device__ __host__ flatternMat(T** matrix, T* dest, int matRow, int matCol)
{
	for (int row = 0; row < matRow; row++)
	{
		for (int col = 0; col < matCol; col++)
		{
			dest[row * matCol + col] = matrix[row][col];
		}
	}
}

template<typename T>
void arrayMulScalar(T* src, int rows, int cols, T defaultScalar)
{
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			src[row * cols + col] *= defaultScalar;
		}
	}
}

template<typename T, typename T2>
void stackArray(T* src, T2* dest, int rows, int cols, int depth)
{
	for (int i = 0; i < depth; i++)
	{
		for (int row = 0; row < rows; row++)
		{
			for (int col = 0; col < cols; col++)
			{
				dest[i * rows * cols + row * cols + col] = src[row * cols + col];
			}
		}
	}
}

template
void stackArray(uchar* src, uchar* dest, int rows, int cols, int depth);

template
void stackArray(float* src, float* dest, int rows, int cols, int depth);

template
void stackArray(uchar* src, float* dest, int rows, int cols, int depth);


template<typename T>
void setArrayValue(T* src, int rows, int cols, T defaultValue)
{
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			src[row * cols + col] = defaultValue;
		}
	}
}

template
void setArrayValue(uchar* src, int rows, int cols, uchar defaultValue);

template
void setArrayValue(float* src, int rows, int cols, float defaultValue);

void __device__ __host__ swapRowForMat(double* mat, int row1, int row2, int rows, int cols)
{
	double temp;
	for (int col = 0; col < cols; col++)
	{
		temp = mat[row1 * cols + col];
		mat[row1 * cols + col] = mat[row2 * cols + col];
		mat[row2 * cols + col] = temp;
	}
}

void __device__ __host__ setMatIdentity(double* src, int rows, int cols)
{
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			src[row * cols + col] = row == col ? 1.0 : 0.;
		}
	}
}

bool __device__ __host__ tinyMatMul(double* left, double* right, double* res, int leftRows, int leftCols, int rightRows, int rightCols)
{
	if (leftCols != rightRows)
	{
		printf("matmul Faild\n");
		return false;
	}

	for (int row = 0; row < leftRows; row++)
	{
		for (int col = 0; col < rightCols; col++)
		{
			res[row * rightCols + col] = 0.;
		}
	}

	for (int row = 0; row < leftRows; row++)
	{
		for (int col = 0; col < rightCols; col++)
		{
			for (int index = 0; index < leftCols; index++)
			{
				//res[row][col] = left[row][index] * right[index][col];
				res[row * rightCols + col] += left[row * leftCols + index] * right[index * rightCols + col];
			}
		}
	}
}

void __device__ __host__ eulerAngle2RotationMat(double angles[3], double* rotationMat)
{
	double radianAngle[3];
	for (int i = 0; i < 3; i++)
	{
		radianAngle[i] = angles[i] * PI / 180.;
	}

	double RX[3][3] =
	{
		{1., 0., 0.,},
		{0., cos(radianAngle[0]), -sin(radianAngle[0])},
		{0., sin(radianAngle[0]), cos(radianAngle[0])}
	};

	double RY[3][3] =
	{
		{cos(radianAngle[1]), 0., sin(radianAngle[1])},
		{0., 1., 0.},
		{-sin(radianAngle[1]), 0., cos(radianAngle[1])}
	};

	double RZ[3][3] =
	{
		{cos(radianAngle[2]), -sin(radianAngle[2]), 0.},
		{sin(radianAngle[2]), cos(radianAngle[2]), 0.},
		{0., 0., 1.}
	};

	double RXY[3][3];
	double* xPtr[3], * yPtr[3], * zPtr[3], * xyPtr[3], * finalPtr[3];

	for (int i = 0; i < 3; i++)
	{
		xPtr[i] = RX[i];
		yPtr[i] = RY[i];
		zPtr[i] = RZ[i];
		xyPtr[i] = RXY[i];
	}

	double Rx[9], Ry[9], Rz[9], Rxy[9];
	flatternMat(xPtr, Rx, 3, 3);
	flatternMat(yPtr, Ry, 3, 3);
	flatternMat(zPtr, Rz, 3, 3);
	flatternMat(xyPtr, Rxy, 3, 3);

	double c2wRotateMat[9];
	tinyMatMul(Rx, Ry, Rxy, 3, 3, 3, 3);
	tinyMatMul(Rxy, Rz, c2wRotateMat, 3, 3, 3, 3);
	getMatInvert(c2wRotateMat, rotationMat, 3);
}

bool __host__ getMatInvert(double* src, double* dest, int rows)
{
	double* srcCopy = (double*)malloc(sizeof(double) * rows * rows * 2);
	embedCopy(srcCopy, src, rows, rows * 2, rows, rows);
	setMatIdentity(dest, rows, rows);
	embedCopy(srcCopy, dest, rows, rows * 2, rows, rows, 0, rows);
	//printArray(srcCopy, rows, rows * 2, "stacked");

	double pivotElem;
	for (int row = 0; row < rows; row++)
	{
		//1. divide per row's pivot elem;
		pivotElem = srcCopy[row * rows * 2 + row];
		int changeTimes = 0;
		while (fabs(pivotElem - 0.) < 1e-8)
		{
			++changeTimes;
			if (changeTimes > rows - row - 1)
			{
				printf("Singular mat detected!");
				free(srcCopy);
				return false;
			}
			swapRowForMat(srcCopy, row, row + changeTimes, rows, rows * 2);

			pivotElem = srcCopy[row * rows * 2 + row];
		}

		for (int col = row; col < rows * 2; col++)
		{
			srcCopy[row * rows * 2 + col] /= pivotElem;
		}
		//printArray(srcCopy, rows, rows * 2, "divide pivot");

		//2. Make elements before pivot elem zero
		double sub;
		for (int subRow = row + 1; subRow < rows; subRow++)
		{
			sub = srcCopy[subRow * rows * 2 + row];
			for (int col = 0; col < rows * 2; col++)
			{
				srcCopy[subRow * rows * 2 + col] -= sub * srcCopy[row * rows * 2 + col];
			}
		}
		//printArray(srcCopy, rows, rows * 2, "make elems before pivot zero");
	}
	//printArray(srcCopy, rows, rows * 2, "Up triangled");

	for (int row = rows - 2; row >= 0; row--)
	{
		for (int subRow = rows - 1; subRow > row; subRow--)
		{
			double sub = srcCopy[row * rows * 2 + subRow];
			for (int col = row + 1; col < rows * 2; col++)
			{
				srcCopy[row * rows * 2 + col] -= sub * srcCopy[subRow * rows * 2 + col];
			}
			//printArray(srcCopy, rows, rows * 2, "sub row");
		}
	}

	embedCopy(dest, srcCopy, rows, rows, rows, rows * 2, 0, rows);
	free(srcCopy);
	return true;
}

double __device__ __host__ getEuclidDistance(double* data1, double* data2, const int dimension)
{
	double res = 0.;
	for (int i = 0; i < dimension; i++)
	{
		res += (data2[i] - data1[i]) * (data2[i] - data1[i]);
	}
	return sqrt(res);
}

double __device__ __host__ getMSEError(double* losses, int len)
{
	assert(len!=0);
	double res = 0.0;
	for (int i = 0; i < len; i++)
	{
		res += losses[i] * losses[i];
	}
	return res / len;
}

void __host__ GetRotatedPointAroundAxis
(
	double* point, 
	double* res, 
	double* axisLineFirstPoint, 
	double* axisLineSecondPoint, 
	double rotateAngle
)
{
	double distance = getEuclidDistance(axisLineFirstPoint, axisLineSecondPoint, 3);

	double u = (axisLineFirstPoint[0] - axisLineSecondPoint[0]) / distance;
	double v = (axisLineFirstPoint[1] - axisLineSecondPoint[1]) / distance;
	double w = (axisLineFirstPoint[2] - axisLineSecondPoint[2]) / distance;

	double SinA = sin(rotateAngle * PI / 180.);
	double CosA = cos(rotateAngle * PI / 180.);

	double uu = u * u;
	double vv = v * v;
	double ww = w * w;
	double uv = u * v;
	double uw = u * w;
	double vw = v * w;

	double t00 = uu + (vv + ww) * CosA;
	double t10 = uv * (1 - CosA) + w * SinA;
	double t20 = uw * (1 - CosA) - v * SinA;

	double t01 = uv * (1 - CosA) - w * SinA;
	double t11 = vv + (uu + ww) * CosA;
	double t21 = vw * (1 - CosA) + u * SinA;

	double t02 = uw * (1 - CosA) + v * SinA;
	double t12 = vw * (1 - CosA) - u * SinA;
	double t22 = ww + (uu + vv) * CosA;

	double a0 = axisLineSecondPoint[0];
	double b0 = axisLineSecondPoint[1];
	double c0 = axisLineSecondPoint[2];

	double t03 = (a0 * (vv + ww) - u * (b0 * v + c0 * w)) * (1 - CosA) + (b0 * w - c0 * v) * SinA;
	double t13 = (b0 * (uu + ww) - v * (a0 * u + c0 * w)) * (1 - CosA) + (c0 * u - a0 * w) * SinA;
	double t23 = (c0 * (uu + vv) - w * (a0 * u + b0 * v)) * (1 - CosA) + (a0 * v - b0 * u) * SinA;


	res[0] = t00 * point[0] + t01 * point[1] + t02 * point[2] + t03;
	res[1] = t10 * point[0] + t11 * point[1] + t12 * point[2] + t13;
	res[2] = t20 * point[0] + t21 * point[1] + t22 * point[2] + t23;
}

void __device__ __host__ normalize(double* arrays, int length)
{
	double len = 0.0;
	for (int i = 0; i < length; i++)
	{
		len += arrays[i] * arrays[i];
	}
	len = sqrt(len);
	for (int i = 0; i < length; i++)
	{
		arrays[i] /= len;
	}
}

void __device__ __host__ crossProduct(double* src1, double* src2, double* dest)
{
	dest[0] = src1[1] * src2[2] - src1[2] * src2[1];
	dest[1] = src1[2] * src2[0] - src1[0] * src2[2];
	dest[2] = src1[0] * src2[1] - src1[1] * src2[0];
}

void __host__ setCameraLookAtXOY(double* cameraPoint, double* objectPoint, double* rotateMat)
{
	double newZAxis[3] = {
			objectPoint[0] - cameraPoint[0],
			objectPoint[1] - cameraPoint[1],
			objectPoint[2] - cameraPoint[2]
	};
	arrayMulScalar(newZAxis, 3, 1, -1.);

	if (fabs(newZAxis[0] - 0.) < 1.e-8 && fabs(newZAxis[2] - 0.) < 1.e-8)
	{
		double angles[3] = { (newZAxis[1] < 0. ? 90. : -90.),0.,(newZAxis[1] < 0. ? 0. : 180.) };

		eulerAngle2RotationMat(angles, rotateMat);
		return;
	}
	normalize(newZAxis, 3);
	double temp[3] = { 0., 1., 0. };
	double newXAxis[3];
	double newYAxis[3];
	crossProduct(temp, newZAxis, newXAxis);
	normalize(newXAxis, 3);
	crossProduct(newZAxis, newXAxis, newYAxis);

	embedCopy(rotateMat, newXAxis, 3, 3, 1, 3);
	embedCopy(rotateMat, newYAxis, 3, 3, 1, 3, 1, 0);
	embedCopy(rotateMat, newZAxis, 3, 3, 1, 3, 2, 0);

	//template printArray<double>(rotateMat, 3, 3, "w2c");
	double c2wRotateMat[9];
	getMatInvert(rotateMat, c2wRotateMat, 3);
	//rotate z 90:
	double tempRotation[9];
	embedCopy(tempRotation, c2wRotateMat, 3, 3, 3, 3);
	double flipover[9], angle[3] = { 0., 0., 90. };
	eulerAngle2RotationMat(angle, flipover);
	tinyMatMul(tempRotation, flipover, c2wRotateMat, 3, 3, 3, 3);

	double yTemp[3] = { 0.,1.,0., };
	double y[3];
	tinyMatMul(c2wRotateMat, yTemp, y, 3, 3, 3, 1);

	if (y[2] < 0)
	{
		embedCopy(tempRotation, c2wRotateMat, 3, 3, 3, 3);
		angle[2] = 180.;
		eulerAngle2RotationMat(angle, flipover);
		tinyMatMul(tempRotation, flipover, c2wRotateMat, 3, 3, 3, 3);
	}

	getMatInvert(c2wRotateMat, rotateMat, 3);

}

void setCameraLookAt(double* cameraPos, double* objectPos, double* rotateMat)
{
	double A, B, C, D;
	A = cameraPos[0] - objectPos[0];
	B = cameraPos[1] - objectPos[1];
	C = cameraPos[2] - objectPos[2];

	if (fabs(C) < 1.e-8)
	{
		setCameraLookAtXOY(cameraPos, objectPos, rotateMat);
		return;
	}
	D = A * cameraPos[0] + B * cameraPos[1] + C * cameraPos[2];

	double newXAxis[3], newYAxis[3], newZAxis[3];

	double a = 1.;
	double sintCoefficient = -a * B / sqrt(B * B + C * C);
	double costCoefficient = -a * A * C / sqrt(A * A + B * B + C * C) / sqrt(B * B + C * C);
	double phi = atan2(costCoefficient,sintCoefficient);
	double t = PI/2-phi;

	newYAxis[0] = a * sqrt((B * B + C * C) / (A * A + B * B + C * C)) * cos(t);
	newYAxis[1] = a / sqrt(B * B + C * C) * (C * sin(t) - A * B / sqrt(A * A + B * B + C * C) * cos(t));
	newYAxis[2] = sintCoefficient * sin(t) + costCoefficient * cos(t);

	normalize(newYAxis, 3);

	newZAxis[0] = A;
	newZAxis[1] = B;
	newZAxis[2] = C;
	normalize(newZAxis, 3);

	crossProduct(newYAxis, newZAxis, newXAxis);

	embedCopy(rotateMat, newXAxis, 3, 3, 1, 3);
	embedCopy(rotateMat, newYAxis, 3, 3, 1, 3, 1, 0);
	embedCopy(rotateMat, newZAxis, 3, 3, 1, 3, 2, 0);
}

void ShuffleRank(int* dest, int maxIndex, bool regenerate, int minIndex, const bool shuffle)
{
	if (regenerate)
	{
		for (int i = minIndex; i < maxIndex; i++)
		{
			dest[i] = i;
		}
	}
	if(!shuffle)return;
	int temp;
	for (int i = maxIndex - 1; i > minIndex; i--)
	{
		int changeIndex = rand() % i + minIndex;
		temp = dest[i];
		dest[i] = dest[changeIndex];
		dest[changeIndex] = temp;
	}
}

void readMask(uchar*& arrays, int rows, int cols, const char* fileName, int* bounds)
{
	FILE* file = fopen(fileName, "r");
	if (!file)
	{
		printf("File: %s can not be found.\n", fileName);
		exit(-1);
	}
	int temp;
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			fscanf(file, "%d ", &temp);
			// arrays[row * cols + col] = temp!=0?255:0;
			arrays[row * cols + col] = temp;

			if (temp)
			{
				if (row < bounds[0])
				{
					bounds[0] = row;
				}

				if (row > bounds[1])
				{
					bounds[1] = row;
				}

				if (col < bounds[2])
				{
					bounds[2] = col;
				}

				if (col > bounds[3])
				{
					bounds[3] = col;
				}
			}
			
		}
		fscanf(file, "\n");
	}
	fclose(file);
}

void readUcharFromFile(uchar*& arrays, int rows, int cols, const char* fileName)
{
	FILE* file = fopen(fileName, "r");
	if (!file)
	{
		printf("File: %s can not be found.\n", fileName);
		exit(-1);
	}
	int temp;
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			fscanf(file, "%d ", &temp);
			arrays[row * cols + col] = (unsigned char)temp;
		}
		fscanf(file, "\n");
	}
	fclose(file);
}

void saveUchar2File(uchar* arrays, int rows, int cols, const char* fileName)
{
	FILE* savingFile = fopen(fileName, "w");
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			fprintf(savingFile, "%d ", (int)arrays[row * cols + col]);
		}
		fprintf(savingFile, "\n");
	}
	fclose(savingFile);
}

void readIntFromFile(unsigned int*& arrays, int rows, int cols, const char* fileName)
{
	FILE* file = fopen(fileName, "r");
	if (!file)
	{
		printf("File: %s can not be found.\n", fileName);
		exit(-1);
	}
	int temp;
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			fscanf(file, "%d ", &temp);
			arrays[row * cols + col] = temp;
		}
		fscanf(file, "\n");
	}
	fclose(file);
}

void saveInt2File(unsigned int* arrays, int rows, int cols, const char* fileName)
{
	FILE* savingFile = fopen(fileName, "w");
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			fprintf(savingFile, "%d ", arrays[row * cols + col]);
		}
		fprintf(savingFile, "\n");
	}
	fclose(savingFile);
}

void readFloatersFromFile(float*& arrays, int rows, int cols, const char* fileName)
{
	FILE* file = fopen(fileName, "r");
	if (!file)
	{
		printf("File: %s can not be found.\n", fileName);
		exit(-1);
	}
	float temp;
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{

			fscanf(file, "%f ", &temp);
			arrays[row * cols + col] = temp;
		}
		fscanf(file, "\n");
		printf("%3d/%3d\r", row + 1,rows);
		fflush(stdout);
	}
	fclose(file);
}

void saveFloaters2File(float* arrays, int rows, int cols, const char* fileName, bool append, bool needProgress)
{
	FILE* savingFile;
	if (append)
	{
		savingFile = fopen(fileName, "a+");
	}
	else
	{
		savingFile = fopen(fileName, "w");
	}

	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			fprintf(savingFile, "%.6lf ", arrays[row * cols + col]);
		}
		fprintf(savingFile, "\n");

		if (needProgress)
		{
			printf("%3d/%3d\r", row + 1, rows);
			fflush(stdout);
		}	
	}
	fclose(savingFile);
}

void saveDouble2File(double* arrays, int rows, int cols, const char* fileName, bool append)
{
	FILE* savingFile;
	if (append)
	{
		savingFile = fopen(fileName, "a+");
	}
	else
	{
		savingFile = fopen(fileName, "w");
	}

	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			fprintf(savingFile, "%.10lf ", arrays[row * cols + col]);
		}
		fprintf(savingFile, "\n");
	}
	fclose(savingFile);
}

void checkGPUStatus(const char* whereCalled, bool printSuccessInfo )
{
	/*waits until all of the previous calls to the device have finished.*/
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error)
	{
		printf("\033[1;31mError occured in %s:%s\033[m\n", whereCalled, cudaGetErrorString(error));
		exit(-1);
	}
	else 
	{
		if(printSuccessInfo)
		{
			printf("\033[1;32m%s successed.\033[m\n", whereCalled, cudaGetErrorString(error));
		}
	}
}

void printGPUMemoryInfo()
{
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0)
	{
		printf("Specified GPU not Found!\n");
	}

	size_t gpu_total_size;
	size_t gpu_free_size;

	cudaError_t cuda_status = cudaMemGetInfo(&gpu_free_size, &gpu_total_size);

	if (cudaSuccess != cuda_status)
	{
		std::cout << "Error: cudaMemGetInfo fails : " << cudaGetErrorString(cuda_status) << std::endl;
		exit(1);
	}

	double total_memory = double(gpu_total_size) / (1024.0 * 1024.0);
	double free_memory = double(gpu_free_size) / (1024.0 * 1024.0);
	double used_memory = total_memory - free_memory;

	printf("||          \033[1;35mTotal\033[m||           \033[1;31mUsed\033[m||     \033[1;32m Available\033[m||\n");
	printf("||   \033[1;35m%12.lf\033[m||   \033[1;31m%12.lf\033[m||   \033[1;32m%12.lf\033[m||\n", total_memory, used_memory, free_memory);
}

void getCurrentTime(char* currentTime)
{
	struct tm* local;

	time_t t;
	t = time(NULL);
	local = localtime(&t);
	sprintf(currentTime, "%d_%d_%d_%d_%d_%d",
		local->tm_year + 1900,
		local->tm_mon + 1,
		local->tm_mday,
		local->tm_hour,
		local->tm_min,
		local->tm_sec
	);
}


Configuration* readConfig(const char* configFile)
{
	FILE* config = fopen(configFile, "r");
	if (!config)
	{
		printf("can not read config file:%s\n", configFile);
		exit(-1);
	}
	Configuration* expConfig = (Configuration*)malloc(sizeof(Configuration));

	fscanf(config, "SceneSize:%d %d %d\n", &expConfig->width, &expConfig->height, &expConfig->depth);
	fscanf(config, "RenderSize:%d %d\n", &expConfig->renderWidth, &expConfig->renderHeight);
	int graylized;
	fscanf(config, "Graylized:%d\n", &graylized);
	expConfig->graylized = graylized;
	fscanf(config, "SphericalHarmonicsBand:%d\n",&expConfig->sphericalHarmonicsBand);
	int hierarchyBandStrategy;
	fscanf(config, "HierarchyBandStrategy:%d\n", &hierarchyBandStrategy);
	expConfig->hierarchyBandStrategy = hierarchyBandStrategy;

	int shsForTargetResolutionOnly;
	fscanf(config,"SHsForTargetResolutionOnly:%d\n",&shsForTargetResolutionOnly);
	expConfig->SHsForTargetResoOnly = shsForTargetResolutionOnly;

	fscanf(config, "Hierarchy:%d\n", &expConfig->hierarchy);
	fscanf(config, "MinimumHierarchy:%d\n", &expConfig->minimumHierarchy);
	fscanf(config, "ToSparseHierarchy:%d\n", &expConfig->toSparseHierarchy);

	int silhouetteBound;
	fscanf(config, "SilhouetteBound:%d\n", &silhouetteBound);
	expConfig->silhouetteBound = silhouetteBound;

	int silhouetteInitial;
	fscanf(config, "SilhouetteInitial:%d\n", &silhouetteInitial);
	expConfig->silhouetteInitial = silhouetteInitial;
	fscanf(config, "SilhouetteRadius:%d\n", &expConfig->silhouetteRadius);

	int MaskAreaOnly;
	fscanf(config, "MaskAreaOnly:%d\n", &MaskAreaOnly);
	expConfig->MaskAreaOnly = MaskAreaOnly;

	int initialFromFile;
	fscanf(config, "InitialFromFile:%d\n", &initialFromFile);
	expConfig->initialFromFile = initialFromFile;

	fscanf(config, "InitialFileDirBase:%s\n", &expConfig->initialFileDirBase);
	fscanf(config, "AvailableSHBands:%d\n", &expConfig->availableSHBand);
	int renderOnly;
	fscanf(config, "RenderOnly:%d\n", &renderOnly);
	expConfig->renderOnly = renderOnly;

	fscanf(config, "ViewDirectory:%s\n", &expConfig->viewsDirectory);
	fscanf(config, "CameraPoseFile:%s\n", &expConfig->cameraPoseFile);
	fscanf(config, "ValidationPoseFile:%s\n", &expConfig->validationPoseFile);
	fscanf(config, "NovelViewPoseFile:%s\n", &expConfig->novelViewPoseFile);
	fscanf(config, "CameraDistance:%lf\n", &expConfig->cameraDistance);
	fscanf(config, "ViewCount:%d\n", &expConfig->viewCount);
	fscanf(config, "ValidationCount:%d\n", &expConfig->validationCount);
	fscanf(config, "TestCount:%d\n", &expConfig->testCount);
	fscanf(config, "TrainProcessViewIndex:%d\n", &expConfig->trainProcessViewIndex);
	int whiteBG;
	fscanf(config, "WhiteBG:%d\n", &whiteBG);
	expConfig->whiteBG = whiteBG;
	fscanf(config, "TestSkip:%d\n", &expConfig->testSkip);
	fscanf(config, "RepeatLayers:%d\n", &expConfig->repeatLayers);
	fscanf(config, "RandomSeed:%d\n", &expConfig->randomSeed);
	srand(expConfig->randomSeed);
	fscanf(config, "LearningRate:%lf\n", &expConfig->learningRate);
	fscanf(config, "SHLearningRate:%lf\n", &expConfig->shLearningRate);
	fscanf(config, "DensityOptimizer:%d\n", &expConfig->densityOptimizer);
	fscanf(config, "ColorChannelOptimizer:%d\n", &expConfig->colorChannelOptimizer);
	fscanf(config, "Momentum:%lf\n", &expConfig->momentumCoefficient);
	fscanf(config, "Epochs:%d\n", &expConfig->epochs);
	fscanf(config, "ValidationInterval:%d\n", &expConfig->validationInterval);
	fscanf(config, "BatchSize:%d\n", &expConfig->batchSize);
	int fixBatchSize = 0;
	fscanf(config, "FixBatchSize:%d\n", &fixBatchSize);
	expConfig->fixBatchSize = fixBatchSize != 0;

	fscanf(config, "BaseDir:%s\n", &expConfig->resBaseDirectory);
	char command[1024];
	sprintf(command, "mkdir %s", expConfig->resBaseDirectory);
	system(command);

	sprintf(expConfig->optimizedResDirectory, "%s%soptimizedRes", expConfig->resBaseDirectory, PATH_CONCATENATOR);
	sprintf(command, "mkdir %s", expConfig->optimizedResDirectory);
	system(command);

#if defined(__linux__)
	sprintf(command, "cp %s %s%s", configFile, expConfig->resBaseDirectory, PATH_CONCATENATOR);
	system(command);
#else
	sprintf(command, "copy %s %s%s", configFile, expConfig->resBaseDirectory, PATH_CONCATENATOR);
	system(command);
#endif

	fscanf(config, "SaveIntervals:%d\n", &expConfig->saveIntervals);
	fscanf(config, "HalfLR:%d\n", &expConfig->halfLRIntervals);
	fscanf(config, "LowestLR:%lf\n", &expConfig->lowestLR);

	fscanf(config, "PruneEpoch:%d\n", &expConfig->pruneEpoch);
	fscanf(config, "PruneThreshold:%f\n", &expConfig->pruneThreshold);
	fscanf(config, "Skips:%d\n", &expConfig->skips);
	
	expConfig->skipIndex=(int*)malloc(sizeof(int)*expConfig->skips);
	
	fscanf(config,"SkipIndex:%d ",&expConfig->skipIndex[0]);
	if(expConfig->skips!=0)
	{
		printf("Skipping index:%d",expConfig->skipIndex[0]);
		for(int i = 1;i<expConfig->skips;i++)
		{
			fscanf(config,"%d ",&expConfig->skipIndex[i]);
			printf("'\t%d",expConfig->skipIndex[i]);
		}
		printf("\n");
	}

	fscanf(config,"\n");
	

	fscanf(config, "StepSize:%f\n", &expConfig->stepSize);
	int solid;
	fscanf(config, "SolidStrategy:%d\n", &solid);
	expConfig->solidStrategy = solid;

	fscanf(config, "InitialDensity:%f\n", &expConfig->initialDensity);
	fscanf(config, "InitialColor:%f\n", &expConfig->initialColor);

	fscanf(config, "VarianceLossInterval:%d\n", &expConfig->varianceLossInterval);
	fscanf(config, "VarianceCoefficient:%lf\n", &expConfig->varianceCoefficient);
	fscanf(config, "ColorDisparityCoefficient:%lf\n", &expConfig->shCoefficient);

	fscanf(config, "MaskRayCoefficient:%lf\n", &expConfig->maskLossCoefficient);

	fscanf(config, "VarianceType:%d\n", &expConfig->vairanceType);
	fscanf(config, "NeighboorNumsForTVLoss:%d\n", &expConfig->neighboorNums);
	fscanf(config, "DiffuseRatio:%lf\n", &expConfig->diffuseRatio);

	int exportModel;
	fscanf(config, "ExportModel:%d\n", &exportModel);
	expConfig->exportModel = exportModel == 1;

	int shuffleRays;
	fscanf(config, "ShuffleRays:%d\n",&shuffleRays);
	expConfig->shuffleRays = shuffleRays == 1;

	int justModel;
	fscanf(config,"JustModel:%d\n",&justModel);
	expConfig->justModel = justModel==1;
	if(justModel)
	{
		// expConfig->maskLossCoefficient = 1.0;
		// expConfig->shuffleRays = true;
		
		printf("Just model, set maskLossCoefficient to 1.0 automatically...\n");
	}

	for (int i = expConfig->minimumHierarchy; i <= (justModel?expConfig->minimumHierarchy:expConfig->hierarchy); i++)
	{
		sprintf(command, "mkdir %s%strainProcess_hierarchy%d", expConfig->resBaseDirectory, PATH_CONCATENATOR, i);
		system(command);
	}

	fscanf(config, "HuberDelta:%f\n", &expConfig->huberDelta);
	fclose(config);
	return expConfig;
}

__global__ void initialize_generator
(
	curandStateScrambledSobol64_t* states,
	curandDirectionVectors64_t* devVectors,
	unsigned int* devScrambleConstants,
	int randomOffset,
	size_t size
)
{
	size_t Idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (Idx >= size) return;

	curand_init(devVectors[Idx], devScrambleConstants[Idx], Idx + randomOffset, &states[Idx]);
}

__global__ void uniformDistribution(
	curandStateScrambledSobol64_t* states,
	double* dest,
	double radius,
	double center,
	int length
)
{
	int dataIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (dataIndex >= length)return;

	dest[dataIndex] = (curand_uniform_double(&states[dataIndex % 20000]) * 2 - 1.0) * radius + center;
}

__global__ void normalDistribution(
	curandStateScrambledSobol64_t* states,
	double* dest,
	double radius,
	double center,
	int length
)
{
	int dataIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (dataIndex >= length)return;
	//3 sigma principle, here divide 4
	dest[dataIndex] = curand_normal_double(&states[dataIndex % 20000]) / 4. * radius + center;
	{
		// printf("%.10lf\n",dest[dataIndex]);
		/*
			__device__​ float curand_log_normal ( gen_type* state, float  mean, float  stddev );
			//Return a log-normally distributed float

			__device__​ double curand_log_normal_double ( gen_type* state, double  mean, double  stddev );
			//Return a log-normally distributed double

			__device__​ float curand_normal ( gen_type* state );
			//Return a normally distributed float

			__device__​ double curand_normal_double ( gen_type* state );
			//Return a normally distributed double

			__device__​ unsigned int curand_poisson ( gen_type* state, double lambda );
			//Return a Poisson-distributed unsigned int

			__device__​ float curand_uniform ( gen_type* state );
			//Return a uniformly distributed float

			__device__​ double curand_uniform_double ( gen_type* state );
			//Return a uniformly distributed double
		*/
	}
}

void randGPU(double* sampleIndexGPUPtr, const int len, const float sigma,const int randType)
{
	//host端变量
	curandDirectionVectors64_t* hostVectors;
	unsigned long long int* hostScrambleConstants;
	//device端变量
	curandStateScrambledSobol64_t* states;
	curandDirectionVectors64_t* devVectors;
	unsigned int* devScrambleConstants;
	const size_t size = len >= 20000 ? 20000 : len;
	//获取方向向量和扰动量
	assert(CURAND_STATUS_SUCCESS == curandGetDirectionVectors64(&hostVectors, CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));
	assert(CURAND_STATUS_SUCCESS == curandGetScrambleConstants64(&hostScrambleConstants));
	//申请device端内存
	assert(cudaSuccess == cudaMalloc(&states, size * sizeof(curandStateScrambledSobol64_t)));
	assert(cudaSuccess == cudaMalloc(&devVectors, size * sizeof(curandDirectionVectors64_t)));
	assert(cudaSuccess == cudaMalloc(&devScrambleConstants, size * sizeof(unsigned int)));
	//将获取到的方向向量和扰动量拷贝到device端
	assert(cudaSuccess == cudaMemcpy(devVectors, hostVectors, size * sizeof(curandDirectionVectors64_t), cudaMemcpyHostToDevice));
	assert(cudaSuccess == cudaMemcpy(devScrambleConstants, hostScrambleConstants, size * sizeof(unsigned int), cudaMemcpyHostToDevice));

	int square = sqrt(len * 1.0) + 0.5;
	dim3 grid((square + 31) / 32, (square + 31) / 32);
	dim3 block(32, 32);
	initialize_generator << <grid, block >> > (states, devVectors, devScrambleConstants, rand(), size);
	cudaDeviceSynchronize();

	if (randType == 0)
	{
		uniformDistribution << <grid, block >> > (states, sampleIndexGPUPtr, .5, .5, len);
	}
	else if (randType == 1)
	{
		normalDistribution << <grid, block >> > (states, sampleIndexGPUPtr, sigma, 0., len);
	}
	
	//free(hostVectors);
	//free(hostScrambleConstants);
	cudaFree(states);
	cudaFree(devVectors);
	cudaFree(devScrambleConstants);
	checkGPUStatus("UniformRand");
}

bool isSquareNumber(int i)
{
	int temp = sqrt(i);
	return temp * temp == i;
}
