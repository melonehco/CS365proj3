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
#include <fstream> //for writing out to file
#include <iomanip> //for string formatting via a stream
#include <cstring> //for strtok
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

// A grouping of features used to store object info in the image feature database
struct FeatureVector 
{
    double fillRatio;
    double bboxDimRatio;
    double axisRatio;
};

// A grouping of info about a given image. Can hold multiple objects at a time
struct ImgInfo 
{
    Mat orig;
    Mat thresholded;
    Mat regionMap;
    vector< vector<Point> > contours;
    RotatedRect bbox; // should also be vectors
    FeatureVector features; // should also be vectors
    string label;
};

// Define a function pointer type for using different classifiers
typedef string (*classifyFuncPtr)(FeatureVector&, map<string, vector<FeatureVector> >&, FeatureVector&);

/**
 * Reads in images from the given directory and returns them in a Mat vector 
 */
vector<Mat> readInImageDir( const char *dirname, vector<string> &labelsOut )
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

            //get label from filename
            char *label = strtok ( dp->d_name, "." );
            labelsOut.push_back( string(label) );
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
 * NOTE: The foreground is white and the background is black,
 * because that's the way it needs to be for the built-in drawContours method
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
                // make pixel white if less than threshold val (eg closer to 0)
                if (grayVer.at<unsigned char>(i,j) > thresholdVal)
                {
                    thresholdedVer.at<unsigned char>(i,j) = 0; // background
                    sumBG += grayVer.at<unsigned char>(i,j);
                    countBG++;
                }
                else // make pixel black
                {
                    thresholdedVer.at<unsigned char>(i,j) = 255; // foreground
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
 * Returns a denoised version of a given image matrix
 * Uses dilation first, followed by erosion
 */
Mat getDenoisedImg(Mat img)
{
    //                                           shape          kernel size           starting point
    // Defaults:                               MORPH_RECT           3x3                    0,0                                  
    Mat dilationSpecs = getStructuringElement( MORPH_RECT,       Size( 5,5 ),          Point(0,0));

    Mat erosionSpecs = getStructuringElement( MORPH_RECT,       Size( 5,5 ),          Point(0,0));
    Mat newImg(img.size(), img.type());
    dilate( img, newImg, dilationSpecs );
    erode( newImg, newImg, erosionSpecs );

    return newImg;
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

    // De-noise the images first
    for (int i = 0; i < thresholdedImages.size(); i++)
    {
        thresholdedImages[i].second = getDenoisedImg(thresholdedImages[i].second);
    }

    vector< pair< Mat, Mat> > origMatsAndColoredSections;
    for (int i = 0; i < thresholdedImages.size(); i++)
    {
        // temp mat to store connected components version of image
        // before we pass it into the RGB mat to return for display
        Mat connCompMat(Size(thresholdedImages[i].second.cols, thresholdedImages[i].second.rows), thresholdedImages[i].second.type());

        // TODO: Make use of testInt to know how many regions there are
        int numSections = connectedComponents(thresholdedImages[i].second, connCompMat); //8-connectedness by default
        
        // Store the connectedComponents output in a RGB version, so 
        // we can store this in the vector that will be displayed
        Mat outputRGBMat(thresholdedImages[i].second.size(), CV_8UC3, Scalar(0,0,0));

        cvtColor(thresholdedImages[i].second,outputRGBMat,CV_GRAY2BGR);   

        // Vector of random colors to color each section in the image
        vector< vector<int> > randColorValsForSections;
        for (int j = 0; j < numSections; j++)
        {
            vector<int> colorVect;
            // get # between 0 and 255 for all 3 channels
            colorVect.push_back(rand() % 255);
            colorVect.push_back(rand() % 255);
            colorVect.push_back(rand() % 255);

            randColorValsForSections.push_back(colorVect);
        }
        cout << "number of regions in image " << i << ": " << numSections << "\n";

        // For each pixel in the cc mat, check which value it has and color accordingly
        for (int row = 0; row< connCompMat.rows; row++)
        {
            for (int col = 0; col < connCompMat.cols; col++)
            {                    
                for (int k = 0; k < numSections; k++)
                {   
                    // segfault here!!!! Whyyyyy
                    //cout << "original col,row val: " << (int)connCompMat.at<int>(Point(row, col)) << "\n";
                    if (connCompMat.at<int>(row, col) == k)
                    {
                        outputRGBMat.at<Vec3b>(row, col)[0] = randColorValsForSections[k][0];
                        outputRGBMat.at<Vec3b>(row, col)[1] = randColorValsForSections[k][1];
                        outputRGBMat.at<Vec3b>(row, col)[2] = randColorValsForSections[k][2];
                        break; // look at next pixel after we've colored this one
                    }
                }
            }
        }

        // store in the output vector
        origMatsAndColoredSections.push_back(make_pair(thresholdedImages[i].first,outputRGBMat));
    }

    return origMatsAndColoredSections;
}

/**
 * Returns a feature vector describing the specified region in the given region map.
 * Currently, we specify the region to be the one with the most points on its contour, 
 * eg the biggest non-background region
 */
FeatureVector calcFeatureVectors(Mat &regionMap, int regionCount, vector< vector<Point> > &contoursOut, RotatedRect &bboxOut)
{
    FeatureVector featureVec;
    
    vector< vector<Point> > allContours;
    //create mask for entire region and find contours with it
    //from: https://stackoverflow.com/questions/37745274/opencv-find-perimeter-of-a-connected-component
    Mat1b mask_region = regionMap > 0; // exclude bg (region val of 0)
    // "RETR_EXTERNAL" excludes inner contours; we only want top-level contours
    findContours(mask_region, allContours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    

    // choose largest contour
    int largestContourIdx = 0; // default to first (top-left-most) contour
    for (int i = 0; i < allContours.size(); i++)
    {
        if (allContours[i].size() > allContours[largestContourIdx].size())
        {
            largestContourIdx = i;
        }
    }

    for (vector<Point> contour : allContours)
    {
        contoursOut.push_back(contour);
    }

    // obtain rotated bounding box
    // NOTE: contoursOut[0] gets outermost contour level, eg not contours contained within other contours

    RotatedRect bbox = minAreaRect(contoursOut[largestContourIdx]); 
    bboxOut.angle = bbox.angle;
    bboxOut.center = bbox.center;
    bboxOut.size = bbox.size;

    //calculate bounding box fill ratio
    double objArea = contourArea(contoursOut[largestContourIdx]);
    double bboxArea = bbox.size.width * bbox.size.height;
    double fillRatio = objArea / bboxArea;
    featureVec.fillRatio = fillRatio;

    //calculate ratio of bbox dims
    double bboxDimRatio = bbox.size.width / bbox.size.height;
    if (bboxDimRatio > 1)
    {
        bboxDimRatio = 1.0 / bboxDimRatio;
    }
    featureVec.bboxDimRatio = bboxDimRatio;

    // calculate ratio of major to minor axis
    // (x,y),(MA,ma),angle 
    //cout << contoursOut[i] << "\n";
    RotatedRect rect = fitEllipse(contoursOut[largestContourIdx]);
    // (the axes might need to be floats)
    double majorAxisLength = rect.size.width  / 2;    // this would be the longer one
    double minorAxisLength = rect.size.height / 2;
    double axesRatio = majorAxisLength/minorAxisLength;
    if (axesRatio > 1)
    {
        axesRatio = 1.0 / axesRatio;
    }
    featureVec.axisRatio = axesRatio;

    Moments m = moments(contoursOut[largestContourIdx], true);
    // m.mu20,m.mu02, m.mu22

    return featureVec;
}

/** 
 * Creates ImgInfo for the given input image
 * see ImgInfo struct at top for fields
 */
ImgInfo calcImgInfo (Mat &orig)
{
    ImgInfo result;
    result.orig = orig;
    result.thresholded = thresholdImg(orig);
    result.thresholded = getDenoisedImg(result.thresholded);
    result.regionMap.create(result.thresholded.size(), result.thresholded.type());
    int numRegions = connectedComponents(result.thresholded, result.regionMap); //8-connectedness by default

    //currently hardcoded to use region 1
    result.features = calcFeatureVectors(result.regionMap, numRegions, result.contours, result.bbox);
    return result;
}

/** 
 * Returns a version of the input image with 3 color channels to allow for display
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

/** 
 * Displays original image side by side with thresholded versions,
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

/**
 * Displays each original image/thresholded image pair
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

/* *
 * Writes the given labels and their associated feature vectors out to a file
 */
void writeFeaturesToFile(vector<ImgInfo> &imagesData)
{
    ofstream outfile;
    outfile.open ("featureDB.txt");

    for (int i = 0; i < imagesData.size(); i++)
    {
        FeatureVector fv = imagesData[i].features;
        string ftStr = to_string(fv.fillRatio) + " " + to_string(fv.bboxDimRatio) + " " + to_string(fv.axisRatio);
        outfile << imagesData[i].label << " " << ftStr << "\n";
    }

    outfile.close();
}

/**
 * Returns the standard deviation of the given vector
 */
double stdDev( vector<double> numbers )
{
    int n = numbers.size();
    double sum = 0.0, mean, sqDiffSum = 0.0;
    sum = accumulate(numbers.begin(), numbers.end(), 0.0);
    mean = sum/n;

    for(int i = 0; i < n; ++i)
    {
        sqDiffSum += (numbers[i] - mean) * (numbers[i] - mean);
    }

    return sqrt(sqDiffSum / n);
}

/**
 * Returns a FeatureVector containing the standard deviation of each feature
 * in the given image data
 */
FeatureVector calcFeatureStdDevVector(vector<ImgInfo> imagesData)
{
    //collect features into vectors for std dev function
    vector<double> fillRatioVector;
    vector<double> bboxDimRatioVector;
    vector<double> axisRatioVector;
    for (ImgInfo ii : imagesData)
    {
        fillRatioVector.push_back( ii.features.fillRatio );
        bboxDimRatioVector.push_back( ii.features.bboxDimRatio );
        axisRatioVector.push_back( ii.features.axisRatio );
    }

    FeatureVector stdDevVector;
    stdDevVector.fillRatio = stdDev( fillRatioVector );
    stdDevVector.bboxDimRatio = stdDev( bboxDimRatioVector );
    stdDevVector.axisRatio = stdDev ( axisRatioVector );

    return stdDevVector;
}

/**
 * Returns the scaled Euclidian distance between the given feature vectors
 * using the given standard deviation feature vector
 */
double scaledEuclidianDist(FeatureVector &fv1, FeatureVector &fv2, FeatureVector &stdDevVector)
{
    const int numFeaturesPerObj = 3;

    double dist = 0;
    dist += fabs(fv1.fillRatio - fv2.fillRatio) / stdDevVector.fillRatio;
    dist += fabs(fv1.bboxDimRatio - fv2.bboxDimRatio) / stdDevVector.bboxDimRatio;
    dist += fabs(fv1.axisRatio - fv2.axisRatio) / stdDevVector.axisRatio;

    dist /= (double) numFeaturesPerObj;

    return dist;
}

/**
 * Returns the best label match for the input feature vector
 * in the database of known objects based on scaled euclidian distance,
 * using the given feature standard deviation vector
 */
string classifyByEuclidianDist(FeatureVector &input, map<string, vector<FeatureVector> > &db, FeatureVector &stdDevVector)
{
    //compare each known object type to the input
    string result = "unknown";

    // The minimum distance found between the input feature vector and a label database entry
    double minDist = 0.65; //this value is just from empirical observation

    // for each label in the database
    for(map< string, vector<FeatureVector> >::value_type& objectType : db)
    {
        //cout << "looking at entry: " << objectType.first << "\n";

        double dist = 0;
        //for each feature vector stored with this label
        for ( FeatureVector features : objectType.second )
        {
            //metric: (x_1 - x_2) / stdev_x
            dist += scaledEuclidianDist(input, features, stdDevVector);
        }
        //average distance across number of entries for this object type/label
        dist /= ((double) objectType.second.size());

        //cout << "   distance: " << dist << "\n";

        if ( dist < minDist ) //if we find a closer object match, save it
        {
            result = objectType.first;
            minDist = dist;
        }
    }
    
    return result;
}

/**
 * Returns true if the second value in the first pair is less than 
 * the second value in the second pair. Used to sort distances for classifyByKNN.
 */
bool sortBySecondVal(const pair<string, double> &pair1, const pair<string, double> &pair2)
{
	return (pair1.second < pair2.second);
}

/**
 * Returns the best label match for the given input feature vector,
 * by running KNN with the given database of known objects,
 * using the given standard deviation feature vector
 */
string classifyByKNN(FeatureVector &input, map<string, vector<FeatureVector> > &db, FeatureVector &stdDevVector)
{
    string result = "unknown";

    //calculate and store distances from all feature vectors in database
    vector<pair <string, double> > labelDistPairs;
    // for each label in the database
    for(map< string, vector<FeatureVector> >::value_type& objectType : db)
    {
        //cout << "looking at entry: " << objectType.first << "\n";

        //for each feature vector stored with this label
        for ( FeatureVector features : objectType.second )
        {
            double dist = scaledEuclidianDist(input, features, stdDevVector);
            labelDistPairs.push_back( make_pair(objectType.first, dist) );
        }
    }

    //sort by distance
    sort(labelDistPairs.begin(), labelDistPairs.end(), sortBySecondVal);

    //find most common label among top n matches
    int n = 10;
    int end = labelDistPairs.size() > n ? n : labelDistPairs.size(); //index to stop at
    map<string, int> labelCounts; //count for each label in the top n
    string curLabel, maxLabel; //current label in loop, most common label
    int maxCount = 0;
    for (int i = 0; i < end; i++) //count instances of each label
    {
        curLabel = labelDistPairs[i].first;
        //update count map
        if ( labelCounts.find(curLabel) == labelCounts.end() ) //not already in map
        {
            //add entry
            labelCounts[curLabel] = 1;
        }
        else //found in map
        {
            labelCounts[curLabel] += 1;
        }

        //check if current label has been seen the most times
        if ( labelCounts[curLabel] > maxCount )
        {
            maxCount = labelCounts[curLabel];
            maxLabel = curLabel;
        }
    }
    //TODO: keep as unknown if top matches aren't good matches
    //set result to maxLabel if matches are good enough
    result = maxLabel;

    return result;
}

/**
 * Classifies objects on a live video feed
 */
int openVideoInput( map<string, vector< FeatureVector> > knownObjectDB, FeatureVector stdDevVector, classifyFuncPtr classifyFuncToUse )
{
    cv::VideoCapture *capdev;

	// open the video device
	capdev = new cv::VideoCapture(0);
	if( !capdev->isOpened() ) {
		printf("Unable to open video device\n");
		return(-1);
	}

	cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
		       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

	printf("Expected size: %d %d\n", refS.width, refS.height);

	cv::namedWindow("Input Video", 1);
	cv::Mat frame;

    cv::namedWindow("Thresholded", 1);
    cv::Mat thresholdedFrame;
    ostringstream strStream;

	for(;;) {
		*capdev >> frame; // get a new frame from the camera, treat as a stream

        //do the single-image stuff but with this frame instead
        thresholdedFrame = thresholdImg(frame);
        ImgInfo info = calcImgInfo(frame);
        string result = classifyFuncToUse( info.features, knownObjectDB, stdDevVector);

        //draw in contour
        Scalar color1 = Scalar(150, 50, 255);
        drawContours( thresholdedFrame, info.contours, 0, color1, 4); //thickness 4

        //draw in bounding box
        Point2f rect_points[4];
        info.bbox.points( rect_points ); //copy points into array
        Scalar color2 = Scalar(200, 255, 100);
        for ( int j = 0; j < 4; j++ ) //draw a line between each pair of points
        {
            line( thresholdedFrame, rect_points[j], rect_points[(j+1)%4], color2, 4); //thickness 4
        }

        //show label and feature values on display
        putText(thresholdedFrame, result, Point(10,50),
                FONT_HERSHEY_COMPLEX, 0.8, Scalar(255,255,255));
        
        strStream << "fill ratio: " << std::fixed << std::setprecision(2) << info.features.fillRatio;
        string fillFt = strStream.str();
        strStream.str(std::string()); //clear stream
        putText(thresholdedFrame, fillFt, Point(10,70),
                FONT_HERSHEY_COMPLEX, 0.8, Scalar(255,255,255));

        strStream << "bbox dim ratio: " << std::fixed << std::setprecision(2) << info.features.bboxDimRatio;
        string bboxFt = strStream.str();
        strStream.str(std::string()); //clear stream
        putText(thresholdedFrame, bboxFt, Point(10,90),
                FONT_HERSHEY_COMPLEX, 0.8, Scalar(255,255,255));

        strStream << "major-minor axis ratio: " << std::fixed << std::setprecision(2) << info.features.axisRatio;
        string axisFt = strStream.str();
        strStream.str(std::string()); //clear stream
        putText(thresholdedFrame, axisFt, Point(10,110),
                FONT_HERSHEY_COMPLEX, 0.8, Scalar(255,255,255));

		if( frame.empty() ) {
		  printf("frame is empty\n");
		  break;
		}
		
		cv::imshow("Video", frame);
        cv::imshow("Threshold", thresholdedFrame);

		if(cv::waitKey(10) == 'q') {
		    break;
		}

	}

	// terminate the video capture
	delete capdev;
    return (0);
}

int main( int argc, char *argv[] ) {
    char dirName[256];

    classifyFuncPtr classifyFuncToUse; 
    map<string, classifyFuncPtr> stringToFuncMap;
    stringToFuncMap["KNN"] = &classifyByKNN;
	stringToFuncMap["EUC"] = &classifyByEuclidianDist;
    classifyFuncToUse = stringToFuncMap["EUC"];

	// If user didn't give directory name
	if(argc < 2) 
	{
		cout << "Usage: |directory name|\n";
		exit(-1);
	}

	strcpy(dirName, argv[1]);

    //read in training images
    vector<string> labels;
    vector<Mat> images = readInImageDir( dirName, labels );

    //compile image info
    cout << "\nProcessing training images...\n";
    vector<ImgInfo> imagesData;
    for (int i = 0; i < images.size(); i++)
    {
        cout << "Image #" << i << " is up now!\n";
        imagesData.push_back( calcImgInfo(images[i]) );
        imagesData[i].label = labels[i];
    }

    //calculate standard deviations of features
    FeatureVector stdDevVector = calcFeatureStdDevVector( imagesData );

	//get object labels and write out to file
    cout << "\nWriting out to file...\n";
    //writeFeaturesToFile(imagesData);

    //compile map of labels w/ associated feature vectors, for classifying
    map< string, vector<FeatureVector> > knownObjectDB; //for classification stage
    for (int i = 0; i < images.size(); i++)
    {
        knownObjectDB[imagesData[i].label].push_back(imagesData[i].features);
    }

    // //Uncomment to display connected components visualizations
    // vector<pair <Mat,Mat> > thresholdedImgs = thresholdImageDB(images);
    // displayImgsInSeparateWindows(getConnectedComponentsVector(thresholdedImgs));
    

    cout << "\nOpening live video..\n";
    openVideoInput( knownObjectDB, stdDevVector, classifyFuncToUse );

	waitKey(0);
		
	printf("\nTerminating\n");

	return(0);

}