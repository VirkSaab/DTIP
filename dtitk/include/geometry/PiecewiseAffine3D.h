/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: PiecewiseAffine3D.h,v $
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


#ifndef _geometry_PiecewiseAffine3D_H
#define _geometry_PiecewiseAffine3D_H

#include "Piecewise.h"
#include <iostream>

namespace geometry {
	
	using namespace std;
	
	class Vector3D;
	class Matrix3D;
	class Affine3D;
	class UniformCubicBSpline3D;
	
	class PiecewiseAffine3D : public Piecewise {
		protected:
		Affine3D ***pr;
		
		void allocate ();
		void deallocate ();
		
		public:
		PiecewiseAffine3D ();
		PiecewiseAffine3D (const char *filename);
		PiecewiseAffine3D (const PiecewiseAffine3D& rhs);
		virtual ~PiecewiseAffine3D ();
		
		void configure (const int size[3], const double vsize[3]);
		
		// set up values through parameters
		void setQS (const double *para);
		void getQS (double *para) const;
		
		Affine3D getAffineAt (int i, int j, int k) const;
		void setAffineAt (int i, int j, int k, const Affine3D& in);
		
		// jacobian at the current position
		void getCurrentJacobian (Matrix3D& jac) const;
		
		// regularization measure
		// sampled at the same locations as the image voxels
		double computeDiscontinuityOfCell (const int cell[3], const double rt_vsize[3], const bool lowerHalfOnly = false) const;
		double computeDiscontinuityGradientOfCell (const int cell[3], const Matrix3D dTransM[9], const double rt_vsize[3], double xi[], const bool lowerHalfOnly = false) const;
		
		// assignment
		PiecewiseAffine3D& operator= (const Affine3D& rhs);
		PiecewiseAffine3D& operator= (UniformCubicBSpline3D& rhs);
		
		PiecewiseAffine3D& operator= (const PiecewiseAffine3D& rhs);
		
		void writeAsDispField (const char *filename, const double out_vsize[3]) const;
		void writeAsJacobian (const char *filename, const double out_vsize[3]) const;
		void load (const char *filename);
		void saveAs (const char *filename) const;
		
		friend Vector3D operator* (const PiecewiseAffine3D& lhs, const Vector3D& rhs);
		friend Vector3D& operator*= (Vector3D& lhs, const PiecewiseAffine3D& rhs);
		
	};

}

#endif

