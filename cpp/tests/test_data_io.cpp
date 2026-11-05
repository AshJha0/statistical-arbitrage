/// \file test_data_io.cpp
/// \brief Loader validation: day-column monotonicity, non-finite cells,
/// ragged/non-numeric rows, duplicate headers, config scalars.
#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include "statarb/data_io.hpp"

namespace {

struct TempFile {
    std::string path;
    explicit TempFile(const std::string& text) {
        char name[] = "/tmp/statarb_test_XXXXXX";
        const int fd = mkstemp(name);
        if (fd >= 0) close(fd);
        path = name;
        std::ofstream(path) << text;
    }
    ~TempFile() { std::remove(path.c_str()); }
};

void expect_error_containing(const std::string& text, const std::string& needle) {
    const TempFile f(text);
    try {
        statarb::load_csv(f.path);
        FAIL() << "expected an error containing '" << needle << "'";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
    }
}

TEST(DataIo, CsvDayColumnMustBeMonotone) {
    // Proposed test 16 (MAJ-6).
    expect_error_containing("day,a\n0,1\n1,2\n2,3\n2,4\n3,5\n",
                            "day column must be strictly increasing");
    expect_error_containing("day,a\n0,1\n2,2\n1,3\n3,4\n",
                            "day column must be strictly increasing");
    // Gaps are allowed (strictly increasing); a file without `day` is not checked.
    const TempFile gap("day,a\n0,1\n1,2\n3,3\n4,4\n");
    const auto d = statarb::load_csv(gap.path);
    ASSERT_EQ(d.at("day").size(), 4u);
    EXPECT_EQ(d.at("day")[2], 3.0);
    const TempFile nod("b,a\n5,1\n2,2\n");
    EXPECT_EQ(statarb::load_csv(nod.path).at("b")[0], 5.0);
}

TEST(DataIo, CsvRejectsNonFiniteRaggedNonNumeric) {
    expect_error_containing("day,a\n0,1\n1,nan\n", "non-numeric");
    expect_error_containing("day,a\n0,1\n1,inf\n", "non-numeric");
    expect_error_containing("day,a\n0,1\n1,1e999\n", "non-finite");
    expect_error_containing("day,a\n0,1\n1\n", "ragged");
    expect_error_containing("day,a\n0,1\n\n1,2\n", "ragged");  // interior blank line
    expect_error_containing("day,a\n0,1\n1,abc\n", "non-numeric");
    expect_error_containing("day,a\n", "empty");
    expect_error_containing("day,day\n0,1\n", "duplicate column");
    EXPECT_THROW(statarb::load_csv("/nonexistent/nope.csv"), std::invalid_argument);
    // Trailing blank lines and CRLF endings are tolerated.
    const TempFile ok("day,a\r\n0,1\r\n1,2\r\n\r\n");
    EXPECT_EQ(statarb::load_csv(ok.path).at("a")[1], 2.0);
}

TEST(DataIo, BundledFilesPassValidation) {
    for (const char* name : {"pair_equity.csv", "fx_triangle.csv", "random_walks.csv"}) {
        const auto d = statarb::load_csv(std::string("../data/") + name);
        ASSERT_EQ(d.at("day").size(), 1500u);
        for (std::size_t i = 0; i < 1500; ++i) ASSERT_EQ(d.at("day")[i], static_cast<double>(i));
    }
}

TEST(DataIo, ConfigRejectsNonFiniteAndNonScalar) {
    const TempFile nan("{\"cost_bps\": NaN, \"window\": 60}");
    EXPECT_THROW(statarb::load_config(nan.path), std::invalid_argument);
    const TempFile inf("{\"cost_bps\": Infinity}");
    EXPECT_THROW(statarb::load_config(inf.path), std::invalid_argument);
    const TempFile boolean("{\"cost_bps\": true}");
    EXPECT_THROW(statarb::load_config(boolean.path), std::invalid_argument);
    const TempFile arr("[1, 2]");
    EXPECT_THROW(statarb::load_config(arr.path), std::invalid_argument);
    EXPECT_THROW(statarb::load_config("/nonexistent/nope.json"), std::invalid_argument);
    const TempFile ok("{\"a\": 1.5, \"b\": -2e3}");
    const auto cfg = statarb::load_config(ok.path);
    EXPECT_EQ(cfg.at("a"), 1.5);
    EXPECT_EQ(cfg.at("b"), -2000.0);
}

}  // namespace
