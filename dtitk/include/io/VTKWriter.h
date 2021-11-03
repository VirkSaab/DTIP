/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: VTKWriter.h,v $
  Language:    C++
  Date:        $Date: 2012/02/17 08:58:18 $
  Version:     $Revision: 1.2 $

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


#ifndef _io_VTKWriter_H
#define _io_VTKWriter_H

#include "VTK.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>

namespace io {
	
	class VTK;
	
	template <class Object>
	class VTKWriter : public VTK {
		public:
		VTKWriter (const char *filename, PrimitiveType _type = FLT);
		~VTKWriter ();

		bool writeHeader ();
		
		void setDimensions (const int *size);
		void setSpacing (const double *vsize);
		void setOrigin (const double *_origin);
		
		virtual bool writeElement (const Object& element) = 0;
		
	};


	//implementation
	template <class Object>
	VTKWriter<Object>::VTKWriter (const char *filename, PrimitiveType _type)
		: VTK(filename, 'w') {
		switch (_type) {
			case SHT:
			case INT:
			case FLT:
			case DBL:
				type = _type;
				break;
			default:
				cerr << "unsupported type" << endl;
				exit(1);
		}
	}

	template <class Object>
	VTKWriter<Object>::~VTKWriter () {}

	template <class Object>
	void VTKWriter<Object>::setDimensions (const int *size) {
		for (int i = 0; i < 3; ++i) {
			dimensions[i] = size[i];
		}
	}
	
	template <class Object>
	void VTKWriter<Object>::setSpacing (const double *vsize) {
		for (int i = 0; i < 3; ++i) {
			spacing[i] = vsize[i];
		}
	}
	
	template <class Object>
	void VTKWriter<Object>::setOrigin (const double *_origin) {
		for (int i = 0; i < 3; ++i) {
			origin[i] = _origin[i];
		}
	}
	
	template <class Object>
	bool VTKWriter<Object>::writeHeader () {
		if (header) {
			cerr << "Header already written" << endl;
			return true;
		}
		
		cout << "Writing " << format << " ..." << endl;
		string line = "# vtk DataFile Version 2.0\n";
		gzputs(fp, line.c_str());
		line = "LITTLE ENDIAN\n";
		gzputs(fp, line.c_str());
		line = "BINARY\n";
		gzputs(fp, line.c_str());
		line = "DATASET STRUCTURED_POINTS\n";
		gzputs(fp, line.c_str());
		line = "DIMENSIONS ";
		char buf[256];
		sprintf(buf, "%d", dimensions[0]);
		line += buf;
		line += ' ';
		sprintf(buf, "%d", dimensions[1]);
		line += buf;
		line += ' ';
		sprintf(buf, "%d", dimensions[2]);
		line += buf;
		line += '\n';
		gzputs(fp, line.c_str());
		line = "SPACING ";
		sprintf(buf, "%f", spacing[0]);
		line += buf;
		line += ' ';
		sprintf(buf, "%f", spacing[1]);
		line += buf;
		line += ' ';
		sprintf(buf, "%f", spacing[2]);
		line += buf;
		line += '\n';
		gzputs(fp, line.c_str());
		line = "ORIGIN ";
		sprintf(buf, "%f", origin[0]);
		line += buf;
		line += ' ';
		sprintf(buf, "%f", origin[1]);
		line += buf;
		line += ' ';
		sprintf(buf, "%f", origin[2]);
		line += buf;
		line += '\n';
		gzputs(fp, line.c_str());
		line = "POINT_DATA ";
		sprintf(buf, "%d", dimensions[0] * dimensions[1] * dimensions[2]);
		line += buf;
		line += '\n';
		gzputs(fp, line.c_str());
		line = format;
		line += " data ";
		// type has been checked in the constructor to be valid
		if (type == SHT) {
			line += "short\n";
		} else if (type == INT) {
			line += "int\n";
		} else if (type == FLT) {
			line += "float\n";
		} else if (type == DBL) {
			line += "double\n";
		}
		gzputs(fp, line.c_str());
		
		// indicate header is processed
		header = true;
		
		return true;
	}
	
}

#endif

