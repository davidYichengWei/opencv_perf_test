#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp> // Include for GPU-based KMeans
#include <iostream>
#include <string>
#include <chrono> // For timing

using namespace cv;
using namespace std;
using namespace std::chrono;

void applyCUDAGaussianBlur(const std::string& inputImagePath, const std::string& outputImagePath, int ksize = 9) {
    // Read the input image
    Mat image = imread(inputImagePath);
    if (image.empty()) {
        cerr << "Error: Could not open or find the image!" << endl;
        return;
    }

    // Upload image to GPU
    cuda::GpuMat d_image;
    d_image.upload(image);

    // Create Gaussian filter
    Ptr<cuda::Filter> gaussianFilter = cuda::createGaussianFilter(d_image.type(), d_image.type(), Size(ksize, ksize), 0);

    // Apply Gaussian blur on GPU
    cuda::GpuMat d_blurred;
    gaussianFilter->apply(d_image, d_blurred);

    // Download result back to CPU
    Mat blurred;
    d_blurred.download(blurred);

    // Save the blurred image to the output path
    imwrite(outputImagePath, blurred);

    cout << "CUDA Gaussian blur completed and saved to " << outputImagePath << endl;
}

void applyCUDAImageInversion(const std::string& inputImagePath, const std::string& outputImagePath) {
    // Read the input image
    Mat image = imread(inputImagePath);
    if (image.empty()) {
        cerr << "Error: Could not open or find the image!" << endl;
        return;
    }

    // Upload image to GPU
    cuda::GpuMat d_image;
    d_image.upload(image);

    // Invert the image on GPU
    cuda::GpuMat d_inverted;
    cuda::bitwise_not(d_image, d_inverted);

    // Download result back to CPU
    Mat inverted;
    d_inverted.download(inverted);

    // Save the inverted image to the output path
    imwrite(outputImagePath, inverted);

    cout << "CUDA image inversion completed and saved to " << outputImagePath << endl;
}

void applyCPUKMeansSegmentation(const std::string& inputImagePath, const std::string& outputImagePath, int k) {
    // Read the input image
    Mat image = imread(inputImagePath);
    if (image.empty()) {
        cerr << "Error: Could not open or find the image!" << endl;
        return;
    }

    // Convert the image into a 2D array of pixels (each pixel is a vector of RGB values)
    Mat reshapedImage = image.reshape(1, image.rows * image.cols);
    reshapedImage.convertTo(reshapedImage, CV_32F);

    // Start timing
    auto start = high_resolution_clock::now();

    // Apply K-Means clustering (CPU version)
    Mat labels, centers;
    double compactness = kmeans(reshapedImage, k, labels,
                                TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
                                3, KMEANS_PP_CENTERS, centers);

    // End timing
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "CPU K-Means completed in " << duration.count() << " ms with compactness: " << compactness << endl;

    // Convert back to 8-bit and reshape to the original image
    centers.convertTo(centers, CV_8U);
    Mat segmentedImage(image.size(), image.type());

    for (int i = 0; i < image.rows * image.cols; ++i) {
        int clusterIdx = labels.at<int>(i);
        segmentedImage.at<Vec3b>(i / image.cols, i % image.cols) = centers.at<Vec3b>(clusterIdx);
    }

    // Save the segmented image to the output path
    imwrite(outputImagePath, segmentedImage);

    cout << "CPU-based image segmentation saved to " << outputImagePath << endl;
}

void applyGPUKMeansSegmentation(const std::string& inputImagePath, const std::string& outputImagePath, int k) {
    // Read the input image
    Mat image = imread(inputImagePath);
    if (image.empty()) {
        cerr << "Error: Could not open or find the image!" << endl;
        return;
    }

    // Convert the image into a 2D array of pixels
    Mat reshapedImage = image.reshape(1, image.rows * image.cols);
    reshapedImage.convertTo(reshapedImage, CV_32F);

    // Upload data to GPU
    cuda::GpuMat d_data;
    d_data.upload(reshapedImage);

    // Create GPU KMeans object
    Ptr<cuda::KMeans> gpu_kmeans = cuda::createKMeans();

    // Prepare labels and centers
    cuda::GpuMat d_labels;
    cuda::GpuMat d_centers;

    // Start timing
    auto start = high_resolution_clock::now();

    // Apply K-Means clustering (GPU version)
    double compactness = gpu_kmeans->cluster(d_data, k, d_labels,
                                             TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
                                             d_centers);

    // End timing
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "GPU K-Means completed in " << duration.count() << " ms with compactness: " << compactness << endl;

    // Download labels and centers
    Mat labels, centers;
    d_labels.download(labels);
    d_centers.download(centers);

    // Convert centers back to 8-bit
    centers.convertTo(centers, CV_8U);

    // Create segmented image
    Mat segmentedImage(image.size(), image.type());

    for (int i = 0; i < image.rows * image.cols; ++i) {
        int clusterIdx = labels.at<int>(i);
        segmentedImage.at<Vec3b>(i / image.cols, i % image.cols) = centers.at<Vec3b>(clusterIdx);
    }

    // Save the segmented image to the output path
    imwrite(outputImagePath, segmentedImage);

    cout << "GPU-based image segmentation saved to " << outputImagePath << endl;
}

int main(int argc, char** argv) {
    // Ensure proper usage
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <image_name>" << endl;
        return 1;
    }

    int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    if (deviceCount == 0) {
        cerr << "No CUDA devices found. Exiting." << endl;
        return -1;
    }

    // Select the first CUDA device
    cv::cuda::setDevice(0);

    // File paths
    string inputDir = "../image/";
    string outputDir = "../output/";
    string imageName = argv[1];
    string inputImagePath = inputDir + imageName;

    // Number of clusters for K-Means
    int k = 10;  // You can adjust this

    // Perform CPU K-Means segmentation
    string cpuSegmentedImagePath = outputDir + "cpu_segmented_" + imageName;
    applyCPUKMeansSegmentation(inputImagePath, cpuSegmentedImagePath, k);

    // Perform GPU K-Means segmentation
    string gpuSegmentedImagePath = outputDir + "gpu_segmented_" + imageName;
    applyGPUKMeansSegmentation(inputImagePath, gpuSegmentedImagePath, k);

    // // Perform CUDA image inversion
    // string invertedImagePath = outputDir + "cuda_inverted_" + imageName;
    // applyCUDAImageInversion(inputImagePath, invertedImagePath);

    // // Perform CUDA Gaussian blur
    // string blurredImagePath = outputDir + "cuda_blurred_" + imageName;
    // int blurKernelSize = 9;  // You can adjust the kernel size
    // applyCUDAGaussianBlur(inputImagePath, blurredImagePath, blurKernelSize);

    return 0;
}