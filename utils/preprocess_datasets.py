import cv2
import numpy as np
import imageio
import os
import argparse
from tqdm import tqdm


def image2txt(imageFileName, txtFileName, targetWidth, targetHeight, grayOnly=False,needMaskInfo=False):
    image = imageio.imread(imageFileName)
    if needMaskInfo:
        mask = image[..., -1].astype(np.uint8)
        np.savetxt(txtFileName + "_mask.txt", X=mask, fmt='%d')
    image = cv2.resize(image, (targetWidth, targetHeight))
    if not grayOnly:
        r, g, b, m = cv2.split(image)
        np.savetxt(fname=txtFileName + "_red.txt", X=r, fmt='%d')
        np.savetxt(fname=txtFileName + "_green.txt", X=g, fmt='%d')
        np.savetxt(fname=txtFileName + "_blue.txt", X=b, fmt='%d')

    graylizedImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    np.savetxt(txtFileName + ".txt", graylizedImage, fmt='%d')


def isValidDatasetPath(dataDir):
    assert os.path.exists(dataDir), "Can not find Specified dataset directory"
    train_dir = os.path.join(dataDir,"train")
    assert os.path.exists(train_dir)
    return train_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True,
                            help='directory of dataset, for example:/root/data/NeRFDatasets/lego')
    
    dataDir = parser.parse_args().dataset_dir
    trainDir = isValidDatasetPath(dataDir=dataDir)

    outputDir = os.path.join(dataDir,"hierarchy0","train")
    os.mkdir(outputDir)

    trainingImageNums = 100
    for idx in tqdm(range(trainingImageNums),total=trainingImageNums,desc="transfer images to txt"):
        assert os.path.exists(os.path.join(trainDir,"r_{}.png".format(idx))),"Cannot find: " +  os.path.join(trainDir,"r_{}.png".format(idx))
        
        image2txt(
            imageFileName=os.path.join(trainDir,"r_{}.png".format(idx)),
            txtFileName=os.path.join(outputDir,"_{}".format(idx)),
            targetWidth=800,targetHeight=800,
            grayOnly=False,needMaskInfo=True         
        )