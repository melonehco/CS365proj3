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
#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

struct FeatureVector {
    double fillRatio;
    double bboxDimRatio;
};

struct ImgInfo {
    Mat orig;
    Mat thresholded;
    Mat regionMap;
    vector<vector<Point>> contours;
    RotatedRect bbox;
    FeatureVector features;
    string label;
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

    //create mask for selected region
    //from: https://stackoverflow.com/questions/37745274/opencv-find-perimeter-of-a-connected-component
    Mat1b mask_region = regionMap == regionID;
    findContours(mask_region, contoursOut, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    //obtain rotated bounding box
    RotatedRect bbox = minAreaRect(contoursOut[0]);
    bboxOut.angle = bbox.angle;
    bboxOut.center = bbox.center;
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

/* creates ImgInfo for the given input image
 * see ImgInfo struct at top for fields
 */
ImgInfo calcImgInfo (Mat &orig)
{
    ImgInfo result;
    result.orig = orig;
    result.thresholded = thresholdImg(orig);
    result.regionMap.create(result.thresholded.size(), result.thresholded.type());
    connectedComponents(result.thresholded, result.regionMap); //8-connectedness by default

    //currently hardcoded to use region 1
    result.features = calcFeatureVector(result.regionMap, 1, result.contours, result.bbox);

    return result;
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

/* displays original image side by side with thresholded versions\,
 * with bounding box and contour drawn on
 */
void displayBBoxContour(string winName, ImgInfo &imgData)
{
    float scaledWidth = 500;
	float scale, scaledHeight;

    //make multi-channel version of threshold image
    Mat thresholdedCopy = makeImgMultChannels( imgData.thresholded );

    //draw in contour
    Scalar color1 = Scalar(150, 50, 255);
    drawContours( thresholdedCopy, imgData.contours, 0, color1, 4); //thickness 4

    //draw in bounding box
    Point2f rect_points[4];
    imgData.bbox.points( rect_points ); //copy points into array
    Scalar color2 = Scalar(200, 255, 100);
    for ( int j = 0; j < 4; j++ ) //draw a line between each pair of points
    {
        line( thresholdedCopy, rect_points[j], rect_points[(j+1)%4], color2, 4); //thickness 4
    }

    // resize both images
    scale = scaledWidth / imgData.orig.cols;
    scaledHeight = imgData.orig.rows * scale;
    resize(imgData.orig, imgData.orig, Size(scaledWidth, scaledHeight));
    resize(thresholdedCopy, thresholdedCopy, Size(scaledWidth, scaledHeight));

    // put both images into one window

    // destination window
    Mat dstMat(Size(2*scaledWidth, scaledHeight), CV_8UC3, Scalar(0, 0, 0));

    imgData.orig.copyTo(dstMat(Rect(0, 0, scaledWidth, scaledHeight)));
    thresholdedCopy.copyTo(dstMat(Rect(scaledWidth, 0, scaledWidth, scaledHeight)));

    namedWindow(winName, CV_WINDOW_AUTOSIZE);
    imshow(winName, dstMat);
}

/* displays each original image/thresholded image pair
 * and requests a label from the user through the terminal
 * returns the vector of label strings received
 */
vector<string> displayAndRequestLabels(vector<ImgInfo> &imagesData)
{
    vector<string> labels;
    for (int i = 0; i < imagesData.size(); i++)
    {
        string window_name = "result " + to_string(i);
        displayBBoxContour(window_name, imagesData[i]);
        waitKey(500); //make imshow go through before using cin

        //string label;
        cout << "Please enter object label: ";
        cin >> imagesData[i].label;
        labels.push_back(imagesData[i].label);
        destroyWindow(window_name);
    }
    return labels;
}

/* writes the given labels and their associated feature vectors out to a file
 */
void writeFeaturesToFile(vector<ImgInfo> &imagesData)
{
    ofstream outfile;
    outfile.open ("featureDB.txt");

    for (int i = 0; i < imagesData.size(); i++)
    {
        FeatureVector fv = imagesData[i].features;
        string ftStr = to_string(fv.fillRatio) + " " + to_string(fv.bboxDimRatio);
        outfile << imagesData[i].label << " " << ftStr << "\n";
    }

    outfile.close();
}

/* returns a label string for the given FeatureVector based on
 * the given feature database
 */
string classifyFeatureVector(FeatureVector &input, map<string, vector<FeatureVector>> &db)
{

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

    //compile image info and map of labels w/ associated feature vectors, for classifying
    cout << "\nProcessing training images...\n";
    vector<ImgInfo> imagesData;
    map<string, vector<FeatureVector>> knownObjectDB; //for classification stage
    for (int i = 0; i < images.size(); i++)
    {
        imagesData.push_back( calcImgInfo(images[i]) );
        knownObjectDB[imagesData[i].label].push_back(imagesData[i].features);
    }

	//get object labels and write out to file
    displayAndRequestLabels(imagesData);
    cout << "\nWriting out to file...\n";
    writeFeaturesToFile(imagesData);

    //ask for an image to classify
    // string imgName;
    // cout << "Please give an image filename for classification: ";
    // cin >> imgName;
    // string result = classifyFeatureVector(imgFt, knownObjectDB);

	waitKey(0);
		
	printf("\nTerminating\n");

	return(0);

}