import cv2
from skimage import metrics
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import csv
import os

import glob
"""MSE and SSIM are traditional computer vision and image processing methods to compare images. 
They tend to work best when images are near-perfectly aligned (otherwise, the pixel locations and values would not match up,
 throwing off the similarity score)."""
def create_ext(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def Compare_Funct(image_scan, image_reel):
    image_scan=cv2.imread(image_scan)
    image_reel=cv2.imread(image_reel)
    hist_img1 = cv2.calcHist([image_scan], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_img1[255, 255, 255] = 0 #ignore all white pixels
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_img2 = cv2.calcHist([image_reel], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_img2[255, 255, 255] = 0  #ignore all white pixels
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Find the metric value
    #Ï   https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html#gga994f53817d621e2e4228fc646342d386aa88d6751fb2bb79e07aa8c8717fda881
    Correlation_hist = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
    Correlation_chisqr = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CHISQR)
    Correlation_Intersection = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_INTERSECT )
    Correlation_Bhattacharyya  = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA )
    Correlation_HELLINGER = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_HELLINGER )
    Correlation_CHISQR_ALT = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CHISQR_ALT )
    Correlation_KL_DIV = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_KL_DIV )

    """
    0 = Correlation_hist 
    1 = Correlation_chisqr 
    2 = Correlation_Intersection
    3 = Correlation_Bhattacharyya  
    4 = Correlation_HELLINGER 
    5 = Correlation_CHISQR_ALT 
    6 = Correlation_KL_DIV 
    """

    return(Correlation_hist,Correlation_chisqr,Correlation_Intersection,Correlation_Bhattacharyya,Correlation_HELLINGER,Correlation_CHISQR_ALT,Correlation_KL_DIV)

def SSIM_Scoring(Image_Scan, Image_Reel):
    image1 = cv2.imread(Image_Scan)
    image2 = cv2.imread(Image_Reel)
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation = cv2.INTER_AREA)
    print(image1.shape, image2.shape)
    # Convert images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Calculate SSIM
    ssim_score = metrics.structural_similarity(image1_gray, image2_gray, full=True)
    
    return ssim_score

def mse(Image_Scan, Image_Reel):
    imageA = cv2.imread(Image_Scan)
    imageB = cv2.imread(Image_Reel)
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    """First we convert the images from unsigned 8-bit integers to floating point, 
    that way we don’t run into any problems with modulus operations “wrapping around”. 
    We then take the difference between the images by subtracting the pixel intensities. 
    Next up, we square these difference (hence mean squared error, and finally sum them up."""
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(Image_Scan, Image_Reel,title):
    imageA = Image_Scan
    imageB = Image_Reel
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    imageA = cv2.imread(Image_Scan)
    imageB = cv2.imread(Image_Reel)

    win_size = min(imageA.shape[0], imageA.shape[1]) // 100
    print(win_size)
    
    s = ssim(imageA, imageB, channel_axis=-1)
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")
    # show the images
    plt.show()
     

path_file= create_ext('D:\Tache\CVH\Tache Camera\Camera_Image_Analysis\KPI.csv')

kpi_file = open(path_file, mode='w')
kpi_result = csv.writer(kpi_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
kpi_result.writerow(['Scenario','MSE', 'SSIM', 'Correlation', 'Chi-Square', 'Intersection','Bhattacharyya distance', 'Alternative Chi-Square', 'Kullback-Leibler divergence'])
print('file created')

filenames = glob.glob(f"D:\Tache\CVH\Tache Camera\Test/*.png")
image_reel = ("D:\Tache\CVH\Tache Camera\Test1\Reel_MP9s.png")
for i in filenames:
    MSE = mse(i,image_reel)
    SSIM = SSIM_Scoring(i,image_reel)
    Correlation = Compare_Funct(i,image_reel)
    
    file_name = os.path.basename(i)
    print('Processing : ', file_name)
    kpi_result.writerow([file_name,round(MSE, 2),round(SSIM[0], 2),round(Correlation[0], 2),round(Correlation[1], 2),round(Correlation[2], 2),round(Correlation[3], 2),round(Correlation[5], 2),round(Correlation[6], 2)])
image_scaner = ("D:\Tache\CVH\Tache Camera\SceenShotProject\sct-mon1_75x2120_3450x1800.png")

# image_reel = ("D:\Tache\CVH\Tache Camera\SceenShotProject\sct-mon1_75x2120_3450x1800.png")



Correlation = Compare_Funct(image_scaner,image_reel)
SSIM = SSIM_Scoring(image_scaner,image_reel)
MSE = mse(image_scaner,image_reel)
compare_images(image_scaner,image_reel,'Compare')

print(f"Correlation: ", round(Correlation[0], 4))
print(f"Chi-Square: ", round(Correlation[1], 4))
print(f"Intersection: ", round(Correlation[2], 4))
print(f"Bhattacharyya distance: ", round(Correlation[3], 4))
print(f"Alternative Chi-Square: ", round(Correlation[5], 4))
print(f"Kullback-Leibler divergence: ", round(Correlation[6], 4))



print(f"Structural Similarity Measure Score: ", round(SSIM[0], 2))

print(f"Mean Squared error: ", round(MSE, 2))