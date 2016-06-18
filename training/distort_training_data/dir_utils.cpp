//
//  dir_utils.cpp
//  caffe-test
//
//  Created by James Cox on 09/05/2016.
//  Copyright © 2016 James Cox-Morton. All rights reserved.
//

#include "dir_utils.hpp"

#include <glob.h>

using namespace std;

namespace DirUtils {
  vector<string> list_directory(const string &rect_folder) {
    vector<string> folders;
    glob_t glob_result;
    glob((rect_folder + "/*").c_str(), GLOB_TILDE, NULL, &glob_result);
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
      folders.push_back(string(glob_result.gl_pathv[i]));
    }
    return folders;
  }
  
  
  string extract_filename(const string &full_filename) {
    string::const_iterator last_slash = full_filename.begin();
    for(string::const_iterator iter = full_filename.begin(); iter != full_filename.end(); ++iter) {
      if (*iter == '/') {
        last_slash = iter;
      }
    }
    
    return string(++last_slash, full_filename.end());
  }

}