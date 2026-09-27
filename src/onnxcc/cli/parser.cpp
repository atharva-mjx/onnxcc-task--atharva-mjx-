#include "onnxcc/cli/parser.h"
#include <onnxcc/third_party/cxxopts.hpp>
#include <iostream>
#include <string>

namespace onnxcc::cli {
namespace {

// Centralized usage output to ensure consistency between --help and error states.
void print_top_level_usage(std::ostream& out) {
    out << "Usage: onnxcc <command> [options]\n\n";
    out << "Commands:\n";
    out << "  dump   Inspect an ONNX model\n\n";
    out << "Run 'onnxcc <command> --help' for command-specific options.\n";
}

// Executes the `dump` subcommand. Currently restricted to argument validation.
int run_dump(int argc, char** argv) {
    cxxopts::Options options("onnxcc dump", "Inspect an ONNX model");

    options.add_options()
        ("model", "Path to the ONNX model file", cxxopts::value<std::string>())
        ("show-graph", "Print the model's graph representation")
        ("verbose", "Enable verbose output")
        ("h,help", "Print usage information");

    // cxxopts throws on parsing failures
    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // Process --help immediately to bypass required argument validation.
    if (result.count("help")) {
        std::cout << options.help() << "\n";
        return 0;
    }

    if (!result.count("model")) {
        std::cerr << "Error: --model is required\n";
        return 1;
    }

    std::string model_path = result["model"].as<std::string>();
    bool show_graph = result.count("show-graph") > 0;
    bool verbose = result.count("verbose") > 0;

    // Wire this to the ONNX parser to output the actual model graph.
    std::cout << "Model: " << model_path << "\n";
    std::cout << "show-graph: " << (show_graph ? "true" : "false") << "\n";
    std::cout << "verbose: " << (verbose ? "true" : "false") << "\n";

    return 0;
}

}

int run(int argc, char** argv) {
    // Require at least one subcommand.
    if (argc < 2) {
        std::cerr << "Error: no command provided\n\n";
        print_top_level_usage(std::cerr);
        return 1;
    }

    std::string command = argv[1];

    if (command == "--help" || command == "-h") {
        print_top_level_usage(std::cout);
        return 0;
    }

    if (command == "dump") {
        // Shift arguments left so cxxopts processes "dump" as argv[0].
        return run_dump(argc - 1, argv + 1);
    }

    // Handle unknown subcommands.
    std::cerr << "Error: unknown command '" << command << "'\n\n";
    print_top_level_usage(std::cerr);
    return 1;
}

} 