#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <numeric>


cv::Mat read_image(const std::string &path, bool rgb) {
    cv::Mat img, img_vec;
    int flag;

    if (rgb)
        flag = CV_LOAD_IMAGE_COLOR;
    else
        flag = CV_LOAD_IMAGE_GRAYSCALE;

    img = cv::imread(path, flag);

    return img;
}

/*
 * Task 1. Find sum of all pixels
*/
void accum(std::atomic<size_t>& sum, const cv::Mat &img, size_t start, size_t end) {
    for (auto it = start; it < end; ++it) {
        sum.fetch_add(img.data[it], std::memory_order_relaxed);
    }
}

/*
 * Task 2. Find min value
*/
void minimum(std::atomic<size_t>& min, const cv::Mat &img, size_t start, size_t end) {
    size_t curr_min = 10000;

    for (auto it = start; it < end; ++it) {
        if (img.data[it] < curr_min)
            curr_min = img.data[it];
    }

    if (curr_min < min) {
        min.store(curr_min);
    }
}

/*
 * Task 3. Apply conv filter
 */

void apply_filter(const cv::Mat& img, const cv::Mat& kernel) {
    cv::Mat dst;

    cv::filter2D(img, dst, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
}

int main(int argc, char **argv)
{
    cv::Mat img;

    img = read_image("../images/pepe_frog.jpg", true);

    unsigned int num_threads = 4;
    unsigned int size = img.cols * img.rows;
    unsigned int chunk_start, chunk_end;
    unsigned int step = size_t(size / num_threads);

    std::cout << "Task 1. Sum of all pixels" << std::endl;
    auto start_cpu = std::chrono::system_clock::now();
    size_t cpu_sum = 0;
    for (unsigned int i = 0; i < size; ++i) {
        cpu_sum += img.data[i];
    }
    std::chrono::duration<double> dur_cpu = std::chrono::system_clock::now() - start_cpu;
    std::cout << "1 Thread Time for addition " << dur_cpu.count() << " seconds" << std::endl;
    std::cout << "1 Thread Result: " << cpu_sum << std::endl;

    std::vector<std::thread> t_pool;
    std::atomic<size_t> sum{0};

    auto start_threads = std::chrono::system_clock::now();
    for (unsigned int i = 0; i < num_threads; ++i) {
        chunk_start = i * step;
        chunk_end = (i + 1) * step;

        std::thread t(accum, std::ref(sum), std::ref(img), chunk_start, chunk_end);
        t_pool.push_back(std::move(t));
    }
    for (unsigned int i = 0; i < t_pool.size(); ++i) {
        t_pool[i].join();
    }
    std::chrono::duration<double> dur_th = std::chrono::system_clock::now() - start_threads;
    std::cout << num_threads << " Threads Time for addition " << dur_th.count() << " seconds" << std::endl;
    std::cout << num_threads << " Threads Result: " << sum << std::endl;



    std::cout << std::endl << "Task 2. Min value" << std::endl;
    auto start_min_single = std::chrono::system_clock::now();
    size_t min_value = 1000000;

    for (size_t i = 0; i < size; ++i) {
        if (img.data[i] < min_value)
            min_value = img.data[i];
    }
    auto end_min_single = std::chrono::system_clock::now() - start_min_single;
    std::cout << "1 Thread Time for min " << dur_cpu.count() << " seconds" << std::endl;
    std::cout << "1 Thread Result: " << min_value << std::endl;

    std::vector<std::thread> t_pool_min;
    std::atomic<size_t> t_min{0};

    auto start_threads_min = std::chrono::system_clock::now();
    for (unsigned int i = 0; i < num_threads; ++i) {
        chunk_start = i * step;
        chunk_end = (i + 1) * step;

        std::thread t(minimum, std::ref(t_min), std::ref(img), chunk_start, chunk_end);
        t_pool_min.push_back(std::move(t));
    }
    for (unsigned int i = 0; i < t_pool_min.size(); ++i) {
        t_pool_min[i].join();
    }
    std::chrono::duration<double> dur_th_min = std::chrono::system_clock::now() - start_threads_min;
    std::cout << num_threads << " Threads Time for addition " << dur_th_min.count() << " seconds" << std::endl;
    std::cout << num_threads << " Threads Result: " << t_min << std::endl;


    std::cout << std::endl << "Task 3. Convolution" << std::endl;

    cv::Mat dst;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 1, 0, -1, 1, 0, -1, 1, 0, -1);
    std::vector<cv::Mat> images;

    images.push_back(read_image("../images/city.jpg", true));
    images.push_back(read_image("../images/pepe_frog.jpg", true));
    images.push_back(read_image("../images/pepe_frog_2.jpg", true));
    images.push_back(read_image("../images/pepe_frog_3.jpg", true));
    images.push_back(read_image("../images/space.jpg", true));


    auto start_conv_single = std::chrono::system_clock::now();
    for (auto i: images) {
        cv::filter2D(i, dst, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    }
    std::chrono::duration<double> dur_conv_single = std::chrono::system_clock::now() - start_conv_single;
    std::cout << "1 Thread Time for min " << dur_conv_single.count() << " seconds" << std::endl;

    std::vector<std::thread> t_pool_conv;
    auto start_conv_t = std::chrono::system_clock::now();
    for (unsigned int i = 0; i < num_threads; ++i) {
        chunk_start = i * step;
        chunk_end = (i + 1) * step;

        std::thread t(apply_filter, std::ref(img), std::ref(kernel));
        t_pool_conv.push_back(std::move(t));
    }
    for (unsigned int i = 0; i < t_pool_conv.size(); ++i) {
        t_pool_conv[i].join();
    }
    std::chrono::duration<double> dur_conv_t = std::chrono::system_clock::now() - start_conv_t;
    std::cout << num_threads << " Threads Time for convolution " << dur_conv_t.count() << " seconds" << std::endl;

    return 0;

}