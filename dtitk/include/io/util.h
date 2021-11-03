/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: util.h,v $
  Language:    C++
  Date:        $Date: 2011/12/21 20:39:20 $
  Version:     $Revision: 1.1.1.1 $

  Copyright (c) Gary Hui Zhang (garyhuizhang@gmail.com).
  All rights reserverd.

  DTI-TK is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  DTI-TK is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with DTI-TK.  If not, see <http://www.gnu.org/licenses/>.
============================================================================*/


#ifndef _io_util_H
#define _io_util_H

#include <vector>
#include <string>

namespace io {
	
	using namespace std;
	
	int getNoOfWords (char *in);
	int parseAsDoubles (char *in, double *out, int size);
	
	void parseFileList (const char *input, vector<string>& files);
	
	int parseNiftiFilenameByParts (const char *input, string& prefix, string& fsuffix, string& isuffix);
	int parseNiftiFilename (const char *input, string& fname, string& iname);
	
	void parseDTITKFilename (const char *input, string& prefix, string& suffix);
	
	string getFilenameForNiftiVolumeDerived (const char *input, const char *tag);
	string getFilenameForSymTensor3DVolumeDerived (const char *input, const char *tag);
	string getFilenameForSymTensor3DVolumeDerived (const char *input, const char *tag, const int index);

}

#endif

