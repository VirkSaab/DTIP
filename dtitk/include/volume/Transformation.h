/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Transformation.h,v $
  Language:    C++
  Date:        $Date: 2011/12/21 20:39:21 $
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


#ifndef _volume_Transformation_H
#define _volume_Transformation_H

#include "../geometry/Matrix3D.h"
#include "../geometry/Rotation3D.h"

namespace volume {
	
	using namespace geometry;
	
	class Transformation {
		protected:
		mutable Matrix3D transM;
		mutable Matrix3D transMInv;
		mutable Rotation3D transQ;
		mutable Rotation3D transQInv;
		mutable Matrix3D transS;
		
		protected:
		Transformation();
		
	};
}

#endif

