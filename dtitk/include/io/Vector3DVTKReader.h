/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Vector3DVTKReader.h,v $
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


#ifndef _io_Vector3DVTKReader_H
#define _io_Vector3DVTKReader_H

#include "VTKReader.h"
#include "../geometry/Vector3D.h"

namespace io {
	
	using geometry::Vector3D;
	
	class Vector3DVTKReader : public VTKReader<Vector3D> {
		public:
		Vector3DVTKReader (const char *filename);
		~Vector3DVTKReader ();
		
		bool readElement (Vector3D& element);
		
		template <class PixelType>
		bool readElementGeneric (Vector3D& element);
	};
	
}

#endif

