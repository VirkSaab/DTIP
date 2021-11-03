/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Translation3D.h,v $
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


#ifndef _geometry_Translation3D_H
#define _geometry_Translation3D_H

#include "Rigid3D.h"

namespace geometry {
	
	class Vector3D;
	
	class Translation3D : public Rigid3D {
		public:
		Translation3D ();
		Translation3D (const Vector3D& vec);
		Translation3D (const double x, const double y, const double z);
		
		Translation3D& operator= (const Vector3D& rhs);
		
		Translation3D& inverseEqual ();
		Translation3D inverse () const;
		
		void set (const double x, const double y, const double z);
		void set (const double *para);
		
	};
	
}

#endif

