#include <gtest/gtest.h>
#include "onnxcc/cli/parser.h"
#include <string>
#include <vector>

namespace {

int call_run(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& a : args) {
        argv.push_back(a.data());
    }
    return onnxcc::cli::run(static_cast<int>(argv.size()), argv.data());
}

} 

// onnxcc dump --model <path> -> exit 0
TEST(CliContract, DumpWithModelSucceeds) {
    testing::internal::CaptureStdout();
    int exit_code = call_run({"onnxcc", "dump", "--model", "foo.onnx"});
    std::string out = testing::internal::GetCapturedStdout();

    EXPECT_EQ(exit_code, 0);
    EXPECT_NE(out.find("foo.onnx"), std::string::npos);
}

// onnxcc dump --model <path> --show-graph -> exit 0
TEST(CliContract, DumpShowGraphFlag) {
    testing::internal::CaptureStdout();
    int exit_code = call_run({"onnxcc", "dump", "--model", "foo.onnx", "--show-graph"});
    std::string out = testing::internal::GetCapturedStdout();

    EXPECT_EQ(exit_code, 0);
    EXPECT_NE(out.find("show-graph: true"), std::string::npos);
}

// onnxcc dump --model <path> --verbose -> exit 0
TEST(CliContract, DumpVerboseFlag) {
    testing::internal::CaptureStdout();
    int exit_code = call_run({"onnxcc", "dump", "--model", "foo.onnx", "--verbose"});
    std::string out = testing::internal::GetCapturedStdout();

    EXPECT_EQ(exit_code, 0);
    EXPECT_NE(out.find("verbose: true"), std::string::npos);
}

// onnxcc dump --help -> exit 0, usage printed to stdout
TEST(CliContract, DumpHelpPrintsUsage) {
    testing::internal::CaptureStdout();
    int exit_code = call_run({"onnxcc", "dump", "--help"});
    std::string out = testing::internal::GetCapturedStdout();

    EXPECT_EQ(exit_code, 0);
    EXPECT_NE(out.find("--model"), std::string::npos);
    EXPECT_NE(out.find("--show-graph"), std::string::npos);
    EXPECT_NE(out.find("--verbose"), std::string::npos);
}

// onnxcc --help -> exit 0, top-level usage printed to stdout
TEST(CliContract, TopLevelHelpPrintsUsage) {
    testing::internal::CaptureStdout();
    int exit_code = call_run({"onnxcc", "--help"});
    std::string out = testing::internal::GetCapturedStdout();

    EXPECT_EQ(exit_code, 0);
    EXPECT_NE(out.find("dump"), std::string::npos);
}

// onnxcc dump (no --model) -> non-zero exit, error on stderr
TEST(CliContract, DumpWithoutModelFails) {
    testing::internal::CaptureStderr();
    int exit_code = call_run({"onnxcc", "dump"});
    std::string err = testing::internal::GetCapturedStderr();

    EXPECT_NE(exit_code, 0);
    EXPECT_NE(err.find("--model"), std::string::npos);
}

// onnxcc bogus -> non-zero exit, error on stderr naming the bad subcommand
TEST(CliContract, UnknownSubcommandFails) {
    testing::internal::CaptureStderr();
    int exit_code = call_run({"onnxcc", "bogus"});
    std::string err = testing::internal::GetCapturedStderr();

    EXPECT_NE(exit_code, 0);
    EXPECT_NE(err.find("bogus"), std::string::npos);
}

// onnxcc (no arguments) -> non-zero exit, usage on stderr
TEST(CliContract, NoArgumentsFails) {
    testing::internal::CaptureStderr();
    int exit_code = call_run({"onnxcc"});
    std::string err = testing::internal::GetCapturedStderr();

    EXPECT_NE(exit_code, 0);
    EXPECT_NE(err.find("Usage"), std::string::npos);
}