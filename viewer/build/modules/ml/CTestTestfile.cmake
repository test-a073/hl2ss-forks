# CMake generated Testfile for 
# Source directory: /home/smart/Desktop/sasika/hl2ss-forks/viewer/opencv-4.x/modules/ml
# Build directory: /home/smart/Desktop/sasika/hl2ss-forks/viewer/build/modules/ml
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_ml "/home/smart/Desktop/sasika/hl2ss-forks/viewer/build/bin/opencv_test_ml" "--gtest_output=xml:opencv_test_ml.xml")
set_tests_properties(opencv_test_ml PROPERTIES  LABELS "Main;opencv_ml;Accuracy" WORKING_DIRECTORY "/home/smart/Desktop/sasika/hl2ss-forks/viewer/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/smart/Desktop/sasika/hl2ss-forks/viewer/opencv-4.x/cmake/OpenCVUtils.cmake;1799;add_test;/home/smart/Desktop/sasika/hl2ss-forks/viewer/opencv-4.x/cmake/OpenCVModule.cmake;1365;ocv_add_test_from_target;/home/smart/Desktop/sasika/hl2ss-forks/viewer/opencv-4.x/cmake/OpenCVModule.cmake;1123;ocv_add_accuracy_tests;/home/smart/Desktop/sasika/hl2ss-forks/viewer/opencv-4.x/modules/ml/CMakeLists.txt;2;ocv_define_module;/home/smart/Desktop/sasika/hl2ss-forks/viewer/opencv-4.x/modules/ml/CMakeLists.txt;0;")