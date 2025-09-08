#pragma once

#include <string>
#include <optional>
#include <iostream>
#include <cstdlib>

struct CLIOptions {
    std::string mode;
    std::optional<size_t> amountOfImages;
};


static bool isPositiveInteger(const char* s) {
    char* end;
    long val = std::strtol(s, &end, 10);
    return *end == '\0' && val > 0;
}

CLIOptions parseArguments(int argc, char* argv[]) {
    if (argc < 2 or argc > 3) {
        throw std::runtime_error("Usage: " + std::string(argv[0]) + " <train|reconstruct|generate> [amountOfImages]");
    }

    CLIOptions opts;
    opts.mode = argv[1];

    if (opts.mode == "train") {
        if (argc > 2)  throw std::runtime_error("'train' does not take additional arguments.");
    } 
    
    else if (opts.mode == "reconstruct") {
        if (argc > 2)  throw std::runtime_error("'reconstruct' does not take additional arguments.");
    }

    else if (opts.mode == "generate"){
        if (argc > 2) {
            if (!isPositiveInteger(argv[2])) {
                throw std::runtime_error("the amount of images must be a positive integer.");
            }
            opts.amountOfImages = static_cast<size_t>(std::atoi(argv[2]));
        }
    }
    else {
        throw std::runtime_error("Unknown mode: " + opts.mode + ". Usage: " + std::string(argv[0]) + " <train|reconstruct|generate [amountOfImages]>");
    }

    return opts;
}

