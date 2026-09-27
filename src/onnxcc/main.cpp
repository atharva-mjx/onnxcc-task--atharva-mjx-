#include <iostream>
#include "onnxcc/version.h"
#include "onnxcc/cli/parser.h"

int main(int argc,char** argv) {
    std::cout << "Welcome to ONNXCC " << onnxcc::get_version() << " (" << onnxcc::get_version_codename() << ")!" << std::endl;
    return onnxcc::cli::run(argc,argv);
    return 0;
}