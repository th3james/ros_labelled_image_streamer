//
//  dir_utils.hpp
//  caffe-test
//
//  Created by James Cox on 09/05/2016.
//  Copyright © 2016 James Cox-Morton. All rights reserved.
//

#ifndef dir_utils_hpp
#define dir_utils_hpp

#include <vector>
#include <string>

namespace DirUtils {
  std::vector<std::string> list_directory(const std::string &directory);
  std::string extract_filename(const std::string &full_filename);
}

#endif /* dir_utils_hpp */
