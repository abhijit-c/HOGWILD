// This file is part of libigl, a simple c++ geometry processing library.
// 
// https://github.com/libigl/libigl
//
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef READ_CSV_H
#define READ_CSV_H

#include <Eigen/Core>

#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

// read a matrix from a csv file into a Eigen matrix
// Templates:
//   Scalar  type for the matrix
// Inputs:
//   str  path to .csv file
// Outputs:
//   M  eigen matrix 
template <typename Scalar>
inline bool readCSV(
  const std::string str, 
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& M);

template <typename Scalar>
inline bool readCSV(
  const std::string str, 
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& M)
{
  using namespace std;

  std::vector<std::vector<Scalar> > Mt;
  
  std::ifstream infile(str.c_str());
  std::string line;
  while (std::getline(infile, line))
  {
    std::istringstream iss(line);
    vector<Scalar> temp;
    Scalar a;
    char ch;
    while (iss >> a){
      temp.push_back(a);
      if(!(iss >> ch))
        break;
    }

    if (temp.size() != 0) // skip empty lines
      Mt.push_back(temp);
  }
  
  if (Mt.size() != 0)
  {
    // Verify that it is indeed a matrix
    for (unsigned i = 0; i<Mt.size(); ++i)
    {
      if (Mt[i].size() != Mt[0].size())
      {
        infile.close();
        return false;
      }
    }
    
    M.resize(Mt.size(),Mt[0].size());
    for (unsigned i = 0; i<Mt.size(); ++i)
      for (unsigned j = 0; j<Mt[i].size(); ++j)
        M(i,j) = Mt[i][j];
    
    infile.close();
    return true;
  }
  
  infile.close();
  return false;
}

#endif
