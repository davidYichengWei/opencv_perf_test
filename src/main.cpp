#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp> // Include for GPU-based KMeans
#include <iostream>
#include <string>
#include <chrono> // For timing
#include <vector>
#include <fstream>

using namespace cv;
using namespace std;
using namespace std::chrono;

void resizeToFullHD(Mat& image) {
    resize(image, image, Size(1920, 1080), 0, 0, INTER_LINEAR);
}

void applyCPUKMeansSegmentation(const std::string& inputImagePath, const std::string& outputImagePath, int k, milliseconds& duration, bool resizeToHD = false) {
    // Read the input image
    Mat image = imread(inputImagePath);
    if (image.empty()) {
        cerr << "Error: Could not open or find the image!" << endl;
        return;
    }

    // Resize to Full HD if requested
    if (resizeToHD) {
        resizeToFullHD(image);
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
    duration = duration_cast<milliseconds>(end - start);

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

    // cout << "CPU-based image segmentation saved to " << outputImagePath << endl;
}

void applyGPUKMeansSegmentation(const std::string& inputImagePath, const std::string& outputImagePath, int k, milliseconds& duration, bool resizeToHD = false) {
    // Read the input image
    Mat image = imread(inputImagePath);
    if (image.empty()) {
        cerr << "Error: Could not open or find the image!" << endl;
        return;
    }

    // Resize to Full HD if requested
    if (resizeToHD) {
        resizeToFullHD(image);
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
    duration = duration_cast<milliseconds>(end - start);

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

    // cout << "GPU-based image segmentation saved to " << outputImagePath << endl;
}

struct BenchmarkResult {
    int k;
    double avg_speedup;
    double avg_psnr;
};

std::vector<BenchmarkResult> benchmarkKMeans(const vector<string>& imageNames, int k_min, int k_max, bool resizeToHD = false) {
    std::vector<BenchmarkResult> results;
    string inputDir = "../image/";
    string outputDir = "../output/";

    // For each k value
    for (int k = k_min; k <= k_max; ++k) {
        double total_speedup = 0.0;
        double total_psnr = 0.0;
        int valid_comparisons = 0;

        // For each image
        for (const auto& imageName : imageNames) {
            string inputImagePath = inputDir + imageName;
            string cpuOutputPath = outputDir + "cpu_k" + to_string(k) + "_" + 
                                 (resizeToHD ? "hd_" : "") + imageName;
            string gpuOutputPath = outputDir + "gpu_k" + to_string(k) + "_" + 
                                 (resizeToHD ? "hd_" : "") + imageName;

            // Get CPU duration
            milliseconds cpu_duration;
            applyCPUKMeansSegmentation(inputImagePath, cpuOutputPath, k, cpu_duration, resizeToHD);

            // Get GPU duration
            milliseconds gpu_duration;
            applyGPUKMeansSegmentation(inputImagePath, gpuOutputPath, k, gpu_duration, resizeToHD);

            // Calculate speedup
            double speedup = static_cast<double>(cpu_duration.count()) / gpu_duration.count();
            total_speedup += speedup;

            // Calculate PSNR
            Mat cpuResult = imread(cpuOutputPath);
            Mat gpuResult = imread(gpuOutputPath);
            if (!cpuResult.empty() && !gpuResult.empty()) {
                double psnr = PSNR(cpuResult, gpuResult);
                total_psnr += psnr;
                valid_comparisons++;
            }

            cout << "k=" << k << ", image=" << imageName 
                 << ", speedup=" << speedup 
                 << "x, PSNR=" << (total_psnr/valid_comparisons) << " dB" << endl;
        }

        // Store average results for this k
        BenchmarkResult result;
        result.k = k;
        result.avg_speedup = total_speedup / imageNames.size();
        result.avg_psnr = total_psnr / valid_comparisons;
        results.push_back(result);
    }

    return results;
}

int main(int argc, char** argv) {
    // Select the first CUDA device
    cv::cuda::setDevice(0);

    // List of images to process
    vector<string> imageNames = {
        "4k_1.jpeg", "4k_2.jpeg", "4k_3.jpeg", "4k_4.jpeg",
        "4k_5.jpeg", "4k_6.jpeg", "4k_7.jpeg", "4k_8.jpeg"
    };

    // Run benchmarks for both 4K and Full HD
    // cout << "Running 4K benchmarks..." << endl;
    // auto results_4k = benchmarkKMeans(imageNames, 2, 10, false);
    
    cout << "\nRunning Full HD benchmarks..." << endl;
    auto results_hd = benchmarkKMeans(imageNames, 2, 10, true);

    // Save results to separate CSV files
    // ofstream outFile4k("benchmark_results_4k.csv");
    // outFile4k << "k,speedup,psnr\n";
    // for (const auto& result : results_4k) {
    //     outFile4k << result.k << "," 
    //              << result.avg_speedup << "," 
    //              << result.avg_psnr << "\n";
    // }
    // outFile4k.close();

    ofstream outFileHD("benchmark_results_hd.csv");
    outFileHD << "k,speedup,psnr\n";
    for (const auto& result : results_hd) {
        outFileHD << result.k << "," 
                 << result.avg_speedup << "," 
                 << result.avg_psnr << "\n";
    }
    outFileHD.close();

    return 0;
}