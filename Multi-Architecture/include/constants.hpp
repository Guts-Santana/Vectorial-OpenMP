/*
 * Multi-Architecture Farrar
 * Copyright (C) 2026 Gustavo Santana Lima
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <array>
#include <string>
#include <limits>
#include <cstdint>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/resource.h>

constexpr int MATCH = 1;
constexpr int MISMATCH = -3;
constexpr int GAP = -2;
constexpr int GAP_OPEN = -5;
constexpr int GAP_EXT = -2;

constexpr int16_t MINUS_INF = std::numeric_limits<int16_t>::min();

namespace fs = std::filesystem;


#endif // CONSTANTS_HPP
