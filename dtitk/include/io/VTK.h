/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: VTK.h,v $
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


#ifndef _io_VTK_H
#define _io_VTK_H

#include <iostream>
#include "Endian.h"
#include <string>
#include "zlib.h"

using namespace std;

namespace io {
	
	class VTK {
		public:
		enum PrimitiveType {SHT, INT, FLT, DBL};
		
		protected:
		//file stream
		gzFile fp;
		//header flag
		bool header;
		
		//vtk structured points attributes
		int dimensions[3];
		double spacing[3];
		double origin[3];
		
		//data format
		string format;

		//primitive data type
		PrimitiveType type;
		
		public:
		VTK (const char *filename, char mode);
		virtual ~VTK ();
		
	};
		
}

#endif

