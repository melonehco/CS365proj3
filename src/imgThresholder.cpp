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

struct FeatureVector {
    double fillRatio;
    double bboxDimRatio;
};

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

        // destination window
        Mat dstMat(Size(2*scaledWidth, scaledHeight), CV_8UC3, Scalar(0, 0, 0));

        string window_name = "match " + to_string(i);

        // put both images into the window
        imagePairs[i].first.copyTo(dstMat(Rect(0, 0, scaledWidth, scaledHeight)));
        imagePairs[i].second.copyTo(dstMat(Rect(scaledWidth, 0, scaledWidth, scaledHeight)));


        namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	    imshow(window_name, dstMat);

	}
}

/**
 * Returns the thresholded version of an image
 */
// TODO: Make this return a 3-channel image instead??
Mat thresholdImg(Mat originalImg)
{
    Mat thresholdedVer;
    thresholdedVer.create(originalImg.size(), CV_8UC1);

    Mat grayVer;
    grayVer.create(originalImg.size(), CV_8UC1);

    // Select initial threshold value, typically the mean 8-bit value of the original image.
    cvtColor(originalImg, grayVer, CV_BGR2GRAY);
    float thresholdVal = (float) mean(grayVer).val[0];
    
    bool isDone = false;
    while (!isDone)
    {
        int sumFG = 0;
        int sumBG = 0;
        int countFG = 0;
        int countBG = 0;
        // Divide the original image into black and white using the grayscale version
        for (int i = 0; i < grayVer.rows; i++)
        {
            for (int j = 0; j < grayVer.cols; j++)
            {
                //cout << grayVer.at<Vec3b>(i,j) << "\n";
                // make pixel white if less than threshold val
                if (grayVer.at<unsigned char>(i,j) < thresholdVal)
                {
                    thresholdedVer.at<unsigned char>(i,j) = 255; // background
                    //thresholdedVer.at<Vec3b>(i,j)[1] = 255;
                    //thresholdedVer.at<Vec3b>(i,j)[2] = 255; 
                    sumBG += grayVer.at<unsigned char>(i,j);
                    countBG++;
                }
                else // make pixel black
                {
                    thresholdedVer.at<unsigned char>(i,j) = 0; // foreground
                    //thresholdedVer.at<Vec3b>(i,j)[1] = 0;
                    //thresholdedVer.at<Vec3b>(i,j)[2] = 0;
                    sumFG += grayVer.at<unsigned char>(i,j);
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
        if (fabs(newThreshold-thresholdVal) < 1) // if within a certain limit, done thresholding
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
        cout << "-> thresholding image " << i << "\n";
        Mat thresholdedImg = thresholdImg(images[i]);
        thresholdedImgs.push_back(make_pair(images[i],thresholdedImg));
    }
    return thresholdedImgs;
}

/**
 * Returns a vector of pairs of originals and their connectedComponent visualizations.
 * Takes in the original-thresholded pairs image vector.
 * Note that the built-in connectedComponents function must take in a 
 * one-channel image, but our display function requires both to be 3-channel
 */     
// TODO: Make this display regions in different colors
vector< pair<Mat,Mat> > getConnectedComponentsVector(vector< pair <Mat, Mat> > thresholdedImages)
{
    cout << "\nAnalyzing connected components...\n";
    vector< pair< Mat, Mat> > labelMatsAndOriginals;
    int testInt = 0;
    for (int i = 0; i < thresholdedImages.size(); i++)
    {
        // a one-channel temp to use to do connectedComponents, since
        // connectedComponents can only output to a 1 channel mat
        Mat tempOneChannelMat(Size(thresholdedImages[i].second.cols, thresholdedImages[i].second.rows), thresholdedImages[i].second.type());
        // make one-channel versions of the thresholded images and store them in the temp
        cvtColor(thresholdedImages[i].second, tempOneChannelMat, CV_BGR2GRAY);
        
        testInt = connectedComponents(tempOneChannelMat, tempOneChannelMat); //8-connectedness by default

        // Normalize the regions to fit over 0 to 255 so they can be visualized

        normalize(tempOneChannelMat, tempOneChannelMat, 0, 255, NORM_MINMAX, CV_8U);

        // Store the connectedComponents output in a RGB version, so 
        // we can store this in the vector that will be displayed
        Mat outputMat(Size(tempOneChannelMat.cols, tempOneChannelMat.rows), CV_8UC3, Scalar(0, 0, 0));
        cvtColor(tempOneChannelMat,outputMat,CV_GRAY2BGR);

        

        labelMatsAndOriginals.push_back(make_pair(thresholdedImages[i].second,outputMat));

        cout << "number of regions in image " << i << " : " << testInt << "\n";
    }

    return labelMatsAndOriginals;
}

/*
 * Returns a feature vector describing the specified region in the given region map
 */
FeatureVector calcFeatureVector(Mat &regionMap, int regionID, 
                                vector<vector<Point>> &contoursOut, RotatedRect &bboxOut)
{
    FeatureVector features;

    cout << "HERE?\n";
    //create mask for selected region
    //from: https://stackoverflow.com/questions/37745274/opencv-find-perimeter-of-a-connected-component
    Mat1b mask_region = regionMap == regionID;
    findContours(mask_region, contoursOut, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    cout << "HERE??\n";
    //obtain rotated bounding box
    RotatedRect bbox = minAreaRect(contoursOut[0]);
    cout << "HERE???\n";
    bboxOut.angle = bbox.angle;
    cout << "HERE????\n";
    bboxOut.center = bbox.center;
    cout << "HERE?????\n";
    bboxOut.size = bbox.size;
 
    //calculate bounding box fill ratio
    double objArea = contourArea(contoursOut[0]);
    double bboxArea = bbox.size.width * bbox.size.height;
    double fillRatio = objArea / bboxArea;
    features.fillRatio = fillRatio;
    //calculate ratio of bbox dims
    double bboxDimRatio = bbox.size.width / bbox.size.height;
    if (bboxDimRatio > 1)
    {
        bboxDimRatio = 1.0 / bboxDimRatio;
    }
    features.bboxDimRatio = bboxDimRatio;
    
    Moments m = moments(contoursOut[0], true);

    return features;
}

/* Returns a version of the input image with 3 color channels to allow for display
 * Assumes the input image has a single channel of type unsigned char
 */
Mat makeImgMultChannels(Mat &orig)
{
    Mat result(orig.size(), CV_8UC3);

    for (int i = 0; i < orig.rows; i++)
    {
        for (int j = 0; j < orig.cols; j++)
        {
            result.at<Vec3b>(i, j)[0] = orig.at<unsigned char>(i, j);
            result.at<Vec3b>(i, j)[1] = orig.at<unsigned char>(i, j);
            result.at<Vec3b>(i, j)[2] = orig.at<unsigned char>(i, j);
        }
    }

    return result;
}

/* displays original images side by side with thresholded versions,
 * with bounding boxes drawn on
 */
void displayBoundingBoxes(vector<pair<Mat, Mat>> imagePairs, vector<RotatedRect> boxes, vector<vector<Point>> contours)
{
    float scaledWidth = 500;
	float scale, scaledHeight;

	for (int i = 0; i < imagePairs.size(); i++)
	{
        //make multi-channel version of threshold image
        Mat thresholdedCopy = makeImgMultChannels( imagePairs[i].second );

        //draw in contour
        Scalar color1 = Scalar(150, 50, 255);
        drawContours( thresholdedCopy, contours, i, color1, 4); //thickness 4

        //draw in bounding box
        Point2f rect_points[4];
        boxes[i].points( rect_points ); //copy points into array
        Scalar color2 = Scalar(200, 255, 100);
        for ( int j = 0; j < 4; j++ ) //draw a line between each pair of points
        {
            line( thresholdedCopy, rect_points[j], rect_points[(j+1)%4], color2, 4); //thickness 4
        }

        // resize both images
        scale = scaledWidth / imagePairs[i].first.cols;
        scaledHeight = imagePairs[i].first.rows * scale;
        resize(imagePairs[i].first, imagePairs[i].first, Size(scaledWidth, scaledHeight));
        resize(thresholdedCopy, thresholdedCopy, Size(scaledWidth, scaledHeight));

        // put both images into one window

        // destination window
        Mat dstMat(Size(2*scaledWidth, scaledHeight), CV_8UC3, Scalar(0, 0, 0));

        string window_name = "result " + to_string(i);

        imagePairs[i].first.copyTo(dstMat(Rect(0, 0, scaledWidth, scaledHeight)));
        thresholdedCopy.copyTo(dstMat(Rect(scaledWidth, 0, scaledWidth, scaledHeight)));

        namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	    imshow(window_name, dstMat);
	}
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

	cout << "\nThresholding images...\n";
	vector< pair< Mat, Mat> > threshImages = thresholdImageDB( images );
    
    // Display the pairs of originals next to their thresholded versions
	//displayImgsInSeparateWindows(threshImages);

    /**
    // Get a vector with pairs of the originals next to the CC visualizations
    vector<pair<Mat, Mat> > labelImages = getConnectedComponentsVector(threshImages);
    // Display the pairs
    displayImgsInSeparateWindows(labelImages);
    */

    cout << "\nAnalyzing connected components...\n";
    vector<Mat> labelImages;
    for (int i = 0; i < threshImages.size(); i++)
    {
        cout << "A\n";
        Mat labelImage(threshImages[i].second.size(), threshImages[i].second.type());
        cout << "AA\n";
        connectedComponents(threshImages[i].second, labelImage); //8-connectedness by default
        cout << "AAA\n";
        labelImages.push_back(labelImage);
    }

    cout << "\nComputing features...\n";
    vector<FeatureVector> features;
    vector<vector<Point>> contours;
    vector<RotatedRect> bboxes;
    for (int i = 0; i < labelImages.size(); i++)
    {
        vector<vector<Point>> contoursOut;
        RotatedRect bboxOut;

        FeatureVector ft = calcFeatureVector(labelImages[i], 1, contoursOut, bboxOut);
        features.push_back(ft);
        contours.push_back(contoursOut[0]);
        bboxes.push_back(bboxOut);

        //just outputting to check
        cout << i << ": fill ratio " << ft.fillRatio << "\n";
        cout << i << ": bbox dim ratio " << ft.bboxDimRatio << "\n";
    }

    displayBoundingBoxes(threshImages, bboxes, contours);

	waitKey(0);
		
	printf("\nTerminating\n");

	return(0);

}