/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: UniformCubicBSpline3D.h,v $
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


#ifndef _geometry_UniformCubicBSpline3D_H
#define _geometry_UniformCubicBSpline3D_H

#include "Vector3D.h"
#include <iostream>

// we use (m+3) control points

namespace geometry {
	
	using namespace std;
	
	class Matrix3D;
	
	class UniformCubicBSpline3D {
		protected:
		/**
		 * state variable --- currently queried position
		 */
		mutable Vector3D current;
		
		// dimensions
		int m[3];
		
		// boundary size
		int size[3];
		
		// span
		mutable int span[3];
		
		// position within a span, its square, its cubic
		mutable double u[3];
		mutable double u2[3];
		mutable double u3[3];
		
		// basis function values
		mutable double B[3][4];
		
		// basis function 1st-order derivatives
		mutable double B1[3][4];
		
		// control points 3D array for both directions
		double ***V[3];
		
		void allocate ();
		void deallocate ();
		
		public:
		UniformCubicBSpline3D ();
		UniformCubicBSpline3D (const int m[3], const int size[3], double ***V[3]);
		UniformCubicBSpline3D (const UniformCubicBSpline3D& rhs);
		UniformCubicBSpline3D (const char *filename);
		virtual ~UniformCubicBSpline3D ();
		
		UniformCubicBSpline3D& operator= (const UniformCubicBSpline3D& rhs);
		
		/**
		 * change the transformation
		 */
		void setUniformCubicBSpline (const int m[3], const int size[3], double ***V[3]);
		
		/**
		 * jacobian at the current position
		 */
		void getCurrentJacobian (Matrix3D& jac) const;
		
		void writeAsDispField (const char *filename, const double out_vsize[3]);
		
		/**
		 * actual transformation constructon
		 */
		friend Vector3D operator* (UniformCubicBSpline3D& lhs, const Vector3D& rhs);
		friend Vector3D& operator*= (Vector3D& lhs, const UniformCubicBSpline3D& rhs);
		
		friend ostream& operator<< (ostream&, const UniformCubicBSpline3D&);
		
	};

}

#endif

