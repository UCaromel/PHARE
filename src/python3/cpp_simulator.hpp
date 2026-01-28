#ifndef PHARE_PYTHON_CPP_SIMULATOR_HPP
#define PHARE_PYTHON_CPP_SIMULATOR_HPP

// https://stackoverflow.com/a/51061314/795574
// ASAN detects leaks by default, even in system/third party libraries
inline char const* __asan_default_options()
{
    return "detect_leaks=0";
}

#endif /*PHARE_PYTHON_CPP_SIMULATOR_H*/
