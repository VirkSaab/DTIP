/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Rigid3D.h,v $
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


#ifndef _geometry_Rigid3D_H
#define _geometry_Rigid3D_H

#include "Affine3D.h"

namespace geometry {
	
	class Vector3D;
	class Rotation3D;
	
	class Rigid3D : public Affine3D {
		public:
		Rigid3D ();
		explicit Rigid3D (const Vector3D&);
		explicit Rigid3D (const Rotation3D&);
		Rigid3D (const Vector3D&, const Rotation3D&);
		Rigid3D (const Rigid3D&);
		
		Rigid3D& operator= (const Rotation3D&);
		Rigid3D& operator= (const Rigid3D&);
		
		Rigid3D operator* (const Rigid3D& rhs) const;
		Rigid3D& operator*= (const Rigid3D& rhs);
		Affine3D& operator*= (const Matrix3D& rhs);
		
		Rigid3D& inverseEqual ();
		Rigid3D inverse () const;
		
		// about the origin!!!
		// so it is different from constructor
		void set (const double para[6]);
		
		friend Vector3D operator* (const Rigid3D& lhs, const Vector3D& rhs);		
	};

}

#endif

