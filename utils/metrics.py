import cv2
import numpy as np
import imageio
import os
import argparse
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import lpips

global loss_fn
global loss_fn2


def txt2Image(txtFileName, graylized=True, save=False, needSubfix=True, floaters=False,removeTxt=False):
    image = 0
    if graylized:
        if not floaters:
            image = np.loadtxt(txtFileName + (".txt" if needSubfix else ""), dtype=int)
            if removeTxt:
                os.remove(txtFileName + (".txt" if needSubfix else ""))
        else:
            image = np.loadtxt(txtFileName + ("_Floaters.txt" if needSubfix else ""))
            if removeTxt:
                os.remove(txtFileName + ("_Floaters.txt" if needSubfix else ""))
    else:
        if not floaters:
            r = np.loadtxt(txtFileName + ("_red.txt" if needSubfix else ""), dtype=int)
            g = np.loadtxt(txtFileName + ("_green.txt" if needSubfix else ""), dtype=int)
            b = np.loadtxt(txtFileName + ("_blue.txt" if needSubfix else ""), dtype=int)
            if removeTxt:
                os.remove(txtFileName + ("_red.txt" if needSubfix else ""))
                os.remove(txtFileName + ("_green.txt" if needSubfix else ""))
                os.remove(txtFileName + ("_blue.txt" if needSubfix else ""))
        else:
            r = np.loadtxt(txtFileName + ("_red_Floaters.txt" if needSubfix else ""))
            g = np.loadtxt(txtFileName + ("_green_Floaters.txt" if needSubfix else ""))
            b = np.loadtxt(txtFileName + ("_blue_Floaters.txt" if needSubfix else ""))
            if removeTxt:
                os.remove(txtFileName + ("_red_Floaters.txt" if needSubfix else ""))
                os.remove(txtFileName + ("_green_Floaters.txt" if needSubfix else ""))
                os.remove(txtFileName + ("_blue_Floaters.txt" if needSubfix else ""))
        image = cv2.merge([b, g, r])

    if save:
        if floaters:
            temp = np.clip(image,0,1. if floaters else 255)
            temp *=255.0
            temp = temp.astype(np.uint8) 
        cv2.imwrite((txtFileName[:-4] if not needSubfix else txtFileName) + ("_whiteBG" if not floaters else "") +".png", image if not floaters else temp)

    image = np.clip(image,0,1. if floaters else 255)
    if not floaters:
        image = image.astype(np.uint8)
    return image


def depthtxt2image(filePath,removetxt=False,channels=1,save=True):
    raw = np.loadtxt(filePath)
    if removetxt:
        os.remove(filePath)
    
    raw = raw / np.max(raw) * 255.   
    raw = raw.astype(np.uint8)
    
    cv2.equalizeHist(raw, raw)
    if channels==3:
        raw = cv2.applyColorMap(raw, cv2.COLORMAP_TURBO)
    
    if save:
        cv2.imwrite(filePath[:-4]+".png",raw)
    return raw
    

def mergeAlphaChannel(alphaFilePath,image,remove=False):
    alpha = np.loadtxt(alphaFilePath,dtype=np.uint8)
    if remove:
        os.remove(alphaFilePath)
    image *= 255.
    b,g,r = cv2.split(image.astype(np.uint8))
    res=cv2.merge([r,g,b,alpha])
    imageio.imwrite(alphaFilePath[:-4]+".png",res,"png")


def qualityMeasure(img1, img2, multiChannel=False,needLpips=False,normalized=True):
    # mse = mean_squared_error(img1, img2)
    psnr = peak_signal_noise_ratio(img1, img2)

    # for multichannel,multichannel=True
    if not normalized:
        ssim = structural_similarity(img1, img2, multichannel=multiChannel)  
    else:
        ssim = structural_similarity(img1, img2, multichannel=multiChannel,data_range = 1.)  
    
    if not needLpips:  
        return psnr, ssim
    
    exp = img1.copy()
    gt = img2.copy()

    if normalized:
        exp *= 255.
        exp = exp.astype(np.uint8)
        gt *= 255.
        gt = gt.astype(np.uint8)

    exp = lpips.im2tensor(exp)
    gt = lpips.im2tensor(gt)
    
    # by default this 'normalize' is False meaning that the input is expected to be in the
    # [-1,1]range, if set to True will instead expect input to be in the [0,1] range
    lpipsVGG = loss_fn.forward(exp,gt,normalize=True).detach().numpy().reshape(-1)[0]
    lpipsAlex = loss_fn2.forward(exp,gt,normalize=True).detach().numpy().reshape(-1)[0]
    # lpipsMetric = lpips.
    # print({"PSNR:"})
    return psnr, ssim,lpipsVGG, lpipsAlex


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True,
                            help='directory of training output dir, for example:/norton/Clear-Plenoxels/output/lego_')  
    parser.add_argument("--dataset_dir", type=str, required=True,
                            help='directory of dataset, for example:/root/data/NeRFDatasets/lego') 
    
    baseDir = parser.parse_args().exp_dir
    gtDir = parser.parse_args().dataset_dir

    epochs = [7,8,9,10]
    hierarchys = [3,2,1,0]

    shForTargetResolutionOnly = True
    Band = 0

    width = 800
    height = 800
    removetxt= False
    temporalSkip = 5

    for (epoch,hierarchy) in zip(epochs,hierarchys):
        #1. Initial state test images
        dir = os.path.join(baseDir,"optimizedRes","hierachy{}InitialState_band{}".format(hierarchy,Band),"test")
        for i in tqdm(range(0,200,temporalSkip),desc="initial states"):
            #RGB
            image = txt2Image(os.path.join(dir,"_"+str(i)),graylized=False,save=True,floaters=True,removeTxt=removetxt)
            #Depth
            depthtxt2image(os.path.join(dir,"_"+str(i)+"_disparity.txt"),removetxt=removetxt,channels=3)
            #Mask
            mergeAlphaChannel(os.path.join(dir,"_"+str(i)+"_mask.txt"),image,remove=removetxt)

        #2. trainingProcess images
        dir = os.path.join(baseDir,"trainProcess_hierarchy{}".format(hierarchy))
        for i in tqdm(range(0,epoch+1),desc="training process"):
            #RGB
            image = txt2Image(os.path.join(dir,"_band{}_{}".format(Band,i)),graylized=False,save=True,floaters=True,removeTxt=removetxt)
            #Depth
            depthtxt2image(os.path.join(dir,"_band{}_{}_disparity.txt".format(Band,i)),removetxt=removetxt,channels=3)
            #Mask
            mergeAlphaChannel(os.path.join(dir,"_band{}_{}_mask.txt".format(Band,i)),image,remove=removetxt)

        
        #3. temporal test images
        dir = os.path.join(baseDir,"optimizedRes","hierachy{}Band_{}".format(hierarchy,Band),"test")
        for i in tqdm(range(0,200,temporalSkip),desc="training results"):
            #RGB
            image = txt2Image(os.path.join(dir,"_"+str(i)),graylized=False,save=True,floaters=True,removeTxt=removetxt)
            #Depth
            depthtxt2image(os.path.join(dir,"_"+str(i)+"_disparity.txt"),removetxt=removetxt,channels=3)
            #Mask
            mergeAlphaChannel(os.path.join(dir,"_"+str(i)+"_mask.txt"),image,remove=removetxt)
    
    #4. final training result
    #4.1 training dataset
    dir = os.path.join(baseDir,"optimizedRes","hierachy0final","fittingRes")
    for i in tqdm(range(100),desc="training txts to images",total=100):
        #RGB
        image = txt2Image(os.path.join(dir,"_"+str(i)),graylized=False,save=True,floaters=True,removeTxt=removetxt)
        #Depth
        depthtxt2image(os.path.join(dir,"_"+str(i)+"_disparity.txt"),removetxt=removetxt,channels=3)
        #Mask
        mergeAlphaChannel(os.path.join(dir,"_"+str(i)+"_mask.txt"),image,remove=removetxt)

    #4.2 validation dataset
    dir = os.path.join(baseDir,"optimizedRes","hierachy0final","validation")
    for i in tqdm(range(100),desc="validation txts to images",total=100):
        #RGB
        image = txt2Image(os.path.join(dir,"_"+str(i)),graylized=False,save=True,floaters=True,removeTxt=removetxt)
        #Depth
        depthtxt2image(os.path.join(dir,"_"+str(i)+"_disparity.txt"),removetxt=removetxt,channels=3)
        #Mask
        mergeAlphaChannel(os.path.join(dir,"_"+str(i)+"_mask.txt"),image,remove=removetxt)

    #4.3 test dataset
    dir = os.path.join(baseDir,"optimizedRes","hierachy0final","test")
    for i in tqdm(range(200),desc="validation txts to images",total=200):
        #RGB
        image = txt2Image(os.path.join(dir,"_"+str(i)),graylized=False,save=True,floaters=True,removeTxt=removetxt)
        #Depth
        depthtxt2image(os.path.join(dir,"_"+str(i)+"_disparity.txt"),removetxt=removetxt,channels=3)
        #Mask
        mergeAlphaChannel(os.path.join(dir,"_"+str(i)+"_mask.txt"),image,remove=removetxt)
    
