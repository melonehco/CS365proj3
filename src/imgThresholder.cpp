/* imgThresholder.cpp
 * Thresholds an image
 * 
 * to run:
 * <path-to-bin>/bin/task1 <# results to show> <optional db directory>
 * 
 * Zena Abulhab and Melody Mao
 * CS365 Spring 2019
 * Project 3
 */

#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <ctype.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

/**
 * Reads in images from the given directory and returns them in a Mat vector 
 */
vector<Mat> readInImageDir( const char *dirname )
{
	DIR *dirp;
	struct dirent *dp;
	printf("Accessing directory %s\n\n", dirname);

	// open the directory
	dirp = opendir( dirname );
	if( dirp == NULL ) {
		printf("Cannot open directory %s\n", dirname);
		exit(-1);
	}

	// loop over the contents of the directory, looking for images
    vector<Mat> images;
	while( (dp = readdir(dirp)) != NULL ) {
		if( strstr(dp->d_name, ".jpg") ||
			strstr(dp->d_name, ".png") ||
			strstr(dp->d_name, ".ppm") ||
			strstr(dp->d_name, ".tif") ) {

			//printf("reading in image file: %s\n", dp->d_name);

            // read the image
            string filename = string( dirname ) + "/" + string( dp->d_name );
            Mat src;
            src = imread( filename );

            // test if the read was successful
            if(src.data == NULL) {
                cout << "Unable to read image" << filename << "\n";
                exit(-1);
            }
            images.push_back( src );
		}
	}

	// close the directory
	closedir(dirp);

    return images;
}

/**
 * Displays both the query image and all of the result images in the same window
 */
void displayImgsInSameWindow(vector<Mat> images)
{
	int numRows, numCols;
	int imgsSqrt = (int)sqrt(images.size());
	// assume each image in the set has same dimensions
	int indivImgHeight = images[0].rows;
	int indivImgWidth = images[0].cols;

	numCols = imgsSqrt;
	// account for extra row(s) for remaining image(s), and round up
	numRows = ceil((float)images.size()/(float)numCols);  

	//                     width                 height         color channel & initial color vals
	Mat dstMat(Size(numCols*indivImgWidth, numRows*indivImgHeight), CV_8UC3, Scalar(0, 0, 0));

	int curImgIdx = 0;
	for (int i = 0; i < numRows; i++)
	{
		for (int j = 0; j < numCols; j++)
		{
			if (curImgIdx == images.size())
			{
				break;
			}
			images[curImgIdx].copyTo(dstMat(Rect(j*indivImgWidth, i*indivImgHeight, indivImgWidth, indivImgHeight)));
			curImgIdx ++;
		}
	}

	// Shrink the collective image down after all smaller images are in it
	float scaledWidth = 600;
	float scale, scaledHeight;
	scale = scaledWidth / dstMat.cols;
	scaledHeight = dstMat.rows * scale;

	resize(dstMat,dstMat, Size(scaledWidth, scaledHeight));

	string window_name = "match";
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	imshow(window_name, dstMat);
}

/**
 * Displays both the query image and all of the result
 * images in different windows, side by side
 */
void displayImgsInSeparateWindows(vector<pair <Mat,Mat> > imagePairs)
{
	float scaledWidth = 500;
	float scale, scaledHeight;

	for (int i = 0; i < imagePairs.size(); i++)
	{
        scale = scaledWidth / imagePairs[i].first.cols;
        scaledHeight = imagePairs[i].first.rows * scale;

        // resize both images
        resize(imagePairs[i].first, imagePairs[i].first, Size(scaledWidth, scaledHeight));
        resize(imagePairs[i].second, imagePairs[i].second, Size(scaledWidth, scaledHeight));

        // put both images into one window

        // destination window
        Mat dstMat(Size(2*scaledWidth, scaledHeight), CV_8UC3, Scalar(0, 0, 0));

        string window_name = "match " + to_string(i);

        // vector<Mat> imgsVector;
        // imgsVector.push_back(imagePairs[i].first);
        // imgsVector.push_back(imagePairs[i].second);

        imagePairs[i].first.copyTo(dstMat(Rect(0, 0, scaledWidth, scaledHeight)));
        imagePairs[i].second.copyTo(dstMat(Rect(scaledWidth, 0, scaledWidth, scaledHeight)));
        
        namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	    imshow(window_name, dstMat);

	}
}

/**
 * Returns the thresholded version of an image
 */
Mat thresholdImg(Mat originalImg)
{
    cout << (float) mean(originalImg).val[0] << "\n";

    Mat thresholdedVer;
    thresholdedVer.create(originalImg.size(), originalImg.type());

    Mat grayVer;
    grayVer.create(originalImg.size(), originalImg.type());

    // Select initial threshold value, typically the mean 8-bit value of the original image.
    cvtColor(originalImg, grayVer, CV_BGR2GRAY);
    float thresholdVal = (float) mean(grayVer).val[0];
    cout << mean(grayVer)<< "\n";
    bool isDone = false;

    while (!isDone)
    {
        int sumFG = 0;
        int sumBG = 0;
        int countFG = 0;
        int countBG = 0;
        // Divide the original image into black and white using the grayscale versiony
        for (int i = 0; i< grayVer.rows; i++)
        {
            for (int j = 0; j< grayVer.cols; j++)
            {
                // make pixel white if less than threshold val
                if (grayVer.at<Vec3b>(i,j)[0] < thresholdVal)
                {
                    thresholdedVer.at<Vec3b>(i,j)[0] = 255; // background
                    sumBG += grayVer.at<Vec3b>(i,j)[0];
                    countBG++;
                }
                else // make pixel black
                {
                    thresholdedVer.at<Vec3b>(i,j)[0] = 0; // foreground
                    sumFG += grayVer.at<Vec3b>(i,j)[0];
                    countFG++;
                }
            }
        }
        // Find the average mean values of the foreground and background regions
        float meanFG = (float)sumFG/(float)countFG;
        float meanBG = (float)sumBG/(float)countBG;

        // Calculate the new threshold by averaging the two means.
        float newThreshold = (meanFG+meanBG)/2.0f;

        //cout << fabs(newThreshold-thresholdVal) << "\n";
        if (fabs(newThreshold-thresholdVal) < 50) // if within a certain limit, done thresholding
        {
            isDone = true;
        }
        else // keep trying for a better thresholding
        {
            thresholdVal = newThreshold;
        }
    }

    return thresholdedVer;
}

/**
 * Returns a vector of ordered pairs of the original images and their thresholded versions
 */
vector<pair <Mat, Mat> > thresholdImageDB(vector<Mat> images)
{
    vector< pair < Mat, Mat > > thresholdedImgs;
    for (int i = 0; i < images.size(); i++)
    {
        cout << "thresholding image " << i << "\n";
        Mat thresholdedImg = thresholdImg(images[i]);
        thresholdedImgs.push_back(make_pair(images[i],thresholdedImg));
    }
    return thresholdedImgs;
}

int main( int argc, char *argv[] ) {
    char dirName[256];

	// If user didn't give directory name
	if(argc < 2) 
	{
		cout << "Usage: |directory name|\n";
		exit(-1);
	}
	strcpy(dirName, argv[1]);


    vector<Mat> images = readInImageDir( dirName );



	cout << "Thresholding images...\n\n";
	vector< pair< Mat, Mat> > sortedImages = thresholdImageDB( images );
	
	displayImgsInSeparateWindows(sortedImages);
	//displayImgsInSameWindow(sortedImages);

	waitKey(0);
		
	printf("\nTerminating\n");

	return(0);

}