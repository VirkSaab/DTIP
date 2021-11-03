/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: VTKReader.h,v $
  Language:    C++
  Date:        $Date: 2012/02/09 09:56:29 $
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


#ifndef _io_VTKReader_H
#define _io_VTKReader_H

#include "VTK.h"
#include "util.h"
#include <cstring>
#include <cstdio>

namespace io {
	
	template <class Object>
	class VTKReader : public VTK {
		public:
		VTKReader (const char *filename);
		~VTKReader ();
		
		bool readHeader ();
		
		void getDimensions (int *size) const;
		void getSpacing (double *vsize) const;
		void getOrigin (double *_origin) const;
		
		virtual bool readElement (Object& element) = 0;		
	};

	//implementation
	template <class Object>
	VTKReader<Object>::VTKReader (const char *filename)
		: VTK(filename, 'r') {}

	template <class Object>
	VTKReader<Object>::~VTKReader () {}

	template <class Object>
	void VTKReader<Object>::getDimensions (int *size) const {
		for (int i = 0; i < 3; ++i) {
			size[i] = dimensions[i];
		}
	}
	
	template <class Object>
	void VTKReader<Object>::getSpacing (double *vsize) const {
		for (int i = 0; i < 3; ++i) {
			vsize[i] = spacing[i];
		}
	}
	
	template <class Object>
	void VTKReader<Object>::getOrigin (double *_origin) const {
		for (int i = 0; i < 3; ++i) {
			_origin[i] = origin[i];
		}
	}
	
	template <class Object>
	bool VTKReader<Object>::readHeader () {
		if (header) {
			cerr << "Header already read" << endl;
			return true;
		}
		
		cout << "Reading " << format << " ..." << endl;
		
		const int bufSize = 256;
		char buf[bufSize];
		int counter = 0;
		
		//version line
		++counter;
		gzgets(fp, buf, bufSize);
		const char *line = "# vtk DataFile Version 2.0\n";
		if (strcmp(buf, line) != 0) {
			cerr << "Invalid VTK File Format." << endl;
			cerr << "Line " << counter << endl;
			return false;
		}
		
		++counter;
		//comment line. The type is put in here. skip
		gzgets(fp, buf, bufSize);
		
		//ASCII or BINARY line
		++counter;
		gzgets(fp, buf, bufSize);
		line = "BINARY\n";
		if (strcmp(buf, line) != 0) {
			cerr << "Invalid VTK File Format." << endl;
			cerr << "Line " << counter << endl;
			return false;
		}
		
		//Define data set as a structured points object (so it can be
		//read into VTK directly)
		++counter;
		gzgets(fp, buf, bufSize);
		line = "DATASET STRUCTURED_POINTS\n";
		if (strcmp(buf, line) != 0) {
			cerr << "Invalid VTK File Format." << endl;
			cerr << "Line " << counter << endl;
			return false;
		}

		char tmp[bufSize];
				
		//Array dimensions
		++counter;
		gzgets(fp, buf, bufSize);
		if (getNoOfWords(buf) != 4) {
			cerr << "Invalid VTK File Format." << endl;
			cerr << "Line " << counter << endl;
			return false;
		}
		
		sscanf(buf, "%s %d %d %d", tmp,
				dimensions, dimensions + 1, dimensions + 2);
				
		line = "DIMENSIONS";
		if (strcmp(tmp, line) != 0) {
			cerr << "Invalid VTK File Format." << endl;
			cerr << "Line " << counter << endl;
			return false;
		}
		
		//Voxel dimensions
		++counter;
		gzgets(fp, buf, bufSize);
		if (getNoOfWords(buf) != 4) {
			cerr << "Invalid VTK File Format." << endl;
			cerr << "Line " << counter << endl;
			return false;
		}
		
		sscanf(buf, "%s %lf %lf %lf", tmp, spacing, spacing + 1, spacing + 2);
		line = "SPACING";
		if (strcmp(tmp, line) != 0) {
			cerr << "Invalid VTK File Format." << endl;
			cerr << "Line " << counter << endl;
			return false;
		}
		
		//Origin
		++counter;
		gzgets(fp, buf, bufSize);
		if (getNoOfWords(buf) != 4) {
			cerr << "Invalid VTK File Format." << endl;
			cerr << "Line " << counter << endl;
			return false;
		}
		
		sscanf(buf, "%s %lf %lf %lf", tmp, origin, origin + 1, origin + 2);
		line = "ORIGIN";
		if (strcmp(tmp, line) != 0) {
			cerr << "Invalid VTK File Format." << endl;
			cerr << "Line " << counter << endl;
			return false;
		}

		//Finally the total number of points in the array.
		++counter;
		gzgets(fp, buf, bufSize);
		if (getNoOfWords(buf) != 2) {
			cerr << "Invalid VTK File Format." << endl;
			cerr << "Line " << counter << endl;
			return false;
		}
		
		long tmpTotal;
		sscanf(buf, "%s %ld", tmp, &tmpTotal);
		long total = dimensions[0] * dimensions[1] * dimensions[2];
		line = "POINT_DATA";
		if (strcmp(tmp, line) != 0) {
			cerr << "Invalid VTK File Format." << endl;
			cerr << "Line " << counter << endl;
			return false;
		}

		if (tmpTotal != total) {
			cerr << "Inconsistent VTK File." << endl;
			cerr << "Line " << counter << endl;
			return false;
		}
		
		++counter;
		gzgets(fp, buf, bufSize);
		if (getNoOfWords(buf) != 3) {
			cerr << "Invalid VTK File Format." << endl;
			cerr << "Line " << counter << endl;
			return false;
		}
		
		char tmp2[bufSize];
		char tmp3[bufSize];
		sscanf(buf, "%s %s %s", tmp, tmp2, tmp3);
		if (strcmp(tmp, format.c_str()) != 0) {
			cerr << "Not a " << format << " VTK file" << endl;
			return false;
		}
		
		line = "data";
		if (strcmp(tmp2, line) != 0) {
			cerr << "Invalid VTK File Format." << endl;
			cerr << "Line " << counter << endl;
			return false;
		}
		
		if (strcmp(tmp3, "short") == 0) {
			type = SHT;
		} else if (strcmp(tmp3, "int") == 0) {
			type = INT;
		} else if (strcmp(tmp3, "float") == 0) {
			type = FLT;
		} else if (strcmp(tmp3, "double") == 0) {
			type = DBL;
		} else {
			cerr << "Invalid VTK File Format." << endl;
			cerr << "Line " << counter << endl;
			return false;
		}
		
		//indicate header is processed
		header = true;
		
		return true;	
	}
	
}

#endif

