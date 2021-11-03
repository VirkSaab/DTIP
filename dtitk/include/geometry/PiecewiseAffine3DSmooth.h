/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: PiecewiseAffine3DSmooth.h,v $
  Language:    C++
  Date:        $Date: 2011/12/21 20:39:19 $
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


#ifndef _geometry_PiecewiseAffine3DSmooth_H
#define _geometry_PiecewiseAffine3DSmooth_H

#include "Affine3D.h"
#include "PiecewiseAffine3D.h"
#include <iostream>

namespace geometry {
	
	class Vector3D;
	class Matrix3D;
	
	class PiecewiseAffine3DSmooth : public PiecewiseAffine3D {
		protected:
		mutable Affine3D currentTrans;
		
		public:
		PiecewiseAffine3DSmooth () {};
		PiecewiseAffine3DSmooth (const char *filename) : PiecewiseAffine3D(filename) {};
		PiecewiseAffine3DSmooth (const PiecewiseAffine3D& rhs) : PiecewiseAffine3D(rhs) {};
		virtual ~PiecewiseAffine3DSmooth () {};
		
		void getCurrentJacobian (Matrix3D& jac) const;
		void writeAsDispField (const char *filename, const double out_vsize[3]) const;
		void writeAsJacobian (const char *filename, const double out_vsize[3]) const;
		friend Vector3D& operator*= (Vector3D& lhs, const PiecewiseAffine3DSmooth& rhs);
		
	};

}

#endif

