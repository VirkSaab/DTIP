/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Scale3D.h,v $
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


#ifndef _geometry_Scale3D_H
#define _geometry_Scale3D_H

#include "Matrix3D.h"

namespace geometry {
	
	class Scale3D : public Matrix3D {
		public:
		// identity matrix
		Scale3D ();
		Scale3D (double xx, double yy, double zz);
		
		void set (double xx, double yy, double zz);
	
		Scale3D& inverseEqual ();
		Scale3D inverse () const;
		
		Scale3D& toXXDeriv ();
		Scale3D& toYYDeriv ();
		Scale3D& toZZDeriv ();
		
	};
	
}

#endif

