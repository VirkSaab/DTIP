/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Affine3D.h,v $
  Language:    C++
  Date:        $Date: 2012/10/02 18:20:15 $
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


#ifndef _geometry_Affine3D_H
#define _geometry_Affine3D_H

#include "Vector3D.h"
#include "Matrix3D.h"
#include <iostream>

namespace geometry {
	
	using namespace std;
	
	class Affine3D {
		protected:
		Vector3D vec;
		Matrix3D mat;
		
		void multiplicationBy (const Affine3D& rhs);
		
		public:
		Affine3D ();
		Affine3D (const char *filename);
		explicit Affine3D (const Vector3D&);
		explicit Affine3D (const Matrix3D&);
		Affine3D (const Vector3D&, const Matrix3D&);
		Affine3D (const Affine3D& rhs);
		virtual ~Affine3D ();
		
		Affine3D& operator= (const Affine3D& rhs);
		Affine3D& operator= (const Matrix3D& rhs);
		
		// move origin of the transformation by some vector
		void moveOriginBy (const Vector3D& vec);
		
		void setMatrix (const Matrix3D& in);
		void setVector (const Vector3D& in);
		
		void getMatrix (Matrix3D& out) const;
		void getVector (Vector3D& out) const;
		
		// QS and QR parametrization
		// about the origin
		// different from the constructor
		void setQS (const double para[12]);
		void setQS (const double para[12], const Vector3D& center);
		void getQS (double para[12]) const;
		void getQS (double para[12], const Vector3D& center) const;
		void setQR (const double para[12], const bool upperTriangle = true);
		
		// identity
		Affine3D& toIdentity ();
		
		// inverse
		Affine3D& inverseEqual ();
		Affine3D inverse () const;
		
		// square root
		Affine3D& sqrtEqual ();
		Affine3D sqrt () const;
		
		// scalar multiplication
		Affine3D operator* (const double rhs) const;
		Affine3D& operator*= (const double rhs);
		friend Affine3D operator* (const double lhs, const Affine3D& rhs);
		
		// addition
		Affine3D& operator+= (const Affine3D& rhs);
		Affine3D operator+ (const Affine3D& rhs) const;
		
		// composition of affine transformations
		Affine3D operator* (const Affine3D& rhs) const;
		Affine3D& operator*= (const Affine3D& rhs);
		Affine3D& operator*= (const Matrix3D& rhs);
		
		// construct affine from the movements of four non-coplanar points
		Affine3D& buildAffine (const Vector3D in[4], const Vector3D out[4]);
		
		// IO
		bool load (const char *filename);
  		bool loadDTITKFormat (const char *filename);
  		bool loadITKFormat (const char *filename);
  		bool loadFSLFormat (const char *filename);
		void saveAs (const char *filename) const;
		void saveAsDTITKFormat (const char *filename) const;
		void saveAsITKFormat (const char *filename) const;
		void saveAsFSLFormat (const char *filename) const;
		void writeAsDispField (const char *filename, const int size[3], const double vsize[3]) const;
		
		friend Vector3D operator* (const Affine3D& lhs, const Vector3D& rhs);
		friend Vector3D& operator*= (Vector3D& lhs, const Affine3D& rhs);
		
		friend ostream& operator<< (ostream&, const Affine3D&);
		
		static Affine3D computeShapeAverage (const char *in);
		
		friend class PiecewiseAffine3D;
	};

}

#endif

