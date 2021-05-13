#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <vector>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <omp.h>

#include "AC/PybindTranslator.h"
namespace py = pybind11;

#define NUM_STATISTICS 3

struct QPConfig
{
	int numProcess = 8;
	int subImgW = 104;
	int subImgH = 104;

	int sugImgPadSize = 20;
	int interpMethod = 0;

	float darkThreshold = 35;
}; 

void loadConfig(const pybind11::object& configPy, QPConfig& configCpp) {
	//if (pybind11::hasattr(configPy, "maxReprojErr"))
	//{
	//	auto attr = configPy.attr("maxReprojErr");
	//	configCpp.maxReprojErr = pybind11::cast<float>(*attr);
	//	std::cout << "maxReprojErr set to:" << configCpp.maxReprojErr << std::endl;
	//}

	//if (pybind11::hasattr(configPy, "reprojErrFile"))
	//{
	//	auto attr = configPy.attr("reprojErrFile");
	//	configCpp.outputReprojErrFile = pybind11::cast<std::string>(*attr);
	//	std::cout << "outputReprojErrFile set to:" << configCpp.outputReprojErrFile << std::endl;
	//}

	//if (pybind11::hasattr(configPy, "doUndist"))
	//{
	//	auto attr = configPy.attr("doUndist");
	//	configCpp.doUndist = pybind11::cast<bool>(*attr);
	//	std::cout << "doUndist set to:" << configCpp.doUndist << std::endl;
	//}
	AC::PyTranslator pyTran(configPy);
	pyTran
	("numProcess", configCpp.numProcess)
		("subImgW", configCpp.subImgW)
		("subImgH", configCpp.subImgH)
		("sugImgPadSize", configCpp.sugImgPadSize)
		("interpMethod", configCpp.interpMethod)
		("darkThreshold", configCpp.darkThreshold)
		;
}

//py::array_t<uint8_t> make_array(const py::ssize_t size) {
//	uint8_t* ret = new uint8_t[size];
//	return py::array(size, ret);
//}



py::list genQP(std::string fileName, const std::vector<std::vector<std::vector<float>>>& qvs, const pybind11::object& configPy) {
	QPConfig config;
	loadConfig(configPy, config);

	size_t imgSize = config.subImgW * config.subImgH;
	size_t numSubImg = qvs.size();
	size_t bufSize = numSubImg * imgSize;
	cv::Mat img = cv::imread(fileName, cv::IMREAD_GRAYSCALE);

	//uint8_t* ret = new uint8_t[bufSize];
	//std::fill_n(ret, bufSize, 255);
	py::array_t<uint8_t> arr({numSubImg, size_t(config.subImgH), size_t(config.subImgW)});
	py::array_t<float> imgStatistics({ numSubImg, size_t(NUM_STATISTICS) });

#pragma omp parallel for num_threads(config.numProcess)
	for (int i = 0; i < qvs.size(); i++)
	{
		uint8_t* bufHead = arr.mutable_data(i);
		
		cv::Mat subImg(config.subImgH, config.subImgW, CV_8UC1, bufHead);
		if (qvs[i].size() != 4) {
			std::cout << "Quad " << i << " has " << qvs[i].size() << " vertices, not 4.\n";
			continue;
		}
		std::vector<cv::Point2f> pts1, pts2;

		pts1.push_back(cv::Point2f(qvs[i][0][0], qvs[i][0][1]));
		pts1.push_back(cv::Point2f(qvs[i][1][0], qvs[i][1][1]));
		pts1.push_back(cv::Point2f(qvs[i][2][0], qvs[i][2][1]));
		pts1.push_back(cv::Point2f(qvs[i][3][0], qvs[i][3][1]));
		pts2.push_back(cv::Point2f(config.sugImgPadSize, config.sugImgPadSize));
		pts2.push_back(cv::Point2f(config.subImgW - config.sugImgPadSize, config.sugImgPadSize));
		pts2.push_back(cv::Point2f(config.subImgW - config.sugImgPadSize, config.subImgH - config.sugImgPadSize));
		pts2.push_back(cv::Point2f(config.sugImgPadSize, config.subImgH - config.sugImgPadSize));

		cv::Mat hChart = cv::findHomography(pts1, pts2, 0);

		//warpPerspective(wPDetector.getBinaryImg(), code, hChart, code.size());
		if (hChart.rows == 3 && hChart.cols == 3)
		{
			switch (config.interpMethod)
			{
			case 0:
				cv::warpPerspective(img, subImg, hChart, subImg.size(), cv::INTER_LINEAR);
				break;
			case 1:
				cv::warpPerspective(img, subImg, hChart, subImg.size(), cv::INTER_NEAREST);

			default:
				break;
			}
		}
		else
		{
			subImg = cv::Mat::zeros(config.subImgW, config.subImgH, CV_8UC1);
		}
		cv::Scalar m, stdv;
		cv::Rect centerRegion(config.sugImgPadSize, config.sugImgPadSize, config.subImgW - 2 * config.sugImgPadSize, config.subImgH - 2 * config.sugImgPadSize);
		cv::Mat subImgCenter = subImg(centerRegion);

		meanStdDev(subImgCenter, m, stdv);

		imgStatistics.mutable_at(i, 0) = m[0];
		imgStatistics.mutable_at(i, 1) = stdv[0];

		cv::Mat darkMask = subImgCenter < config.darkThreshold;
		int numDarks = cv::countNonZero(darkMask);
		imgStatistics.mutable_at(i, 2) = numDarks;
	}

	py::list returnList;
	returnList.append(arr);
	returnList.append(imgStatistics);

	return returnList;
}


PYBIND11_MODULE(QuadGeneratorCPP, m) {
	m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";


	m.def("genQP", &genQP, R"pbdoc(
	    Print a set of strings.
	
	)pbdoc", py::arg("imgFile"), py::arg("qvList"), py::arg("config"), py::return_value_policy::take_ownership);

	

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}