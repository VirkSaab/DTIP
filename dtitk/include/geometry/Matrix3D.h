/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Matrix3D.h,v $
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


#ifndef _geometry_Matrix3D_H
#define _geometry_Matrix3D_H

#include <iostream>

namespace geometry {
	
	using namespace std;
	
	typedef double (* array_2D)[3];
	typedef const double (* const_array_2D)[3];
	
	class Vector3D;
	class SymTensor3D;
	
	class Matrix3D {
		protected:
		static const int dim;
		double matrix[3][3];
		
		// (*this) * (rhs) = product
		void multiplication(const Matrix3D& rhs, Matrix3D& product) const;
		
		static void pseudoInverse (array_2D matrix , const_array_2D a);
		
		public:
		// default constructor with matrix filled with zeros
		Matrix3D ();
		
		// when id = false, the matrix is not initialized
		// when id = true, the matrix is initialized as an
		// identity matrix
		Matrix3D (bool id);
		
		// copy constructor
		Matrix3D (const Matrix3D&);
		
		// conversion from symmetric tensor
		explicit Matrix3D (const SymTensor3D&);
		
		// conversion from 2-D array
		Matrix3D (const double in[3][3]);
		
		// conversion from 3-D vectors
		Matrix3D (const Vector3D& v1, const Vector3D& v2, const Vector3D& v3);
		
		// destructor
		virtual ~Matrix3D();
		
		// get element
		double getElement (const int i, const int j) const;
		// modifier
		void setElement (const int i, const int j, const double value);
		void setMatrix (const double in[3][3]);
		// columnwise vector matrix construct
		void setMatrix (const Vector3D& v1, const Vector3D& v2, const Vector3D& v3);
		void setColumn (const int col, const Vector3D& v);
		
		// QR based matrix parametrization
		Matrix3D& setQR (const double *para, const bool upperTriangle = true);
		
		// QS based matrix parameterization
		Matrix3D& setQS (const double *para);
		void getQS (double *para) const;
		
		// cholesky based spd matrix parameterization
		Matrix3D& setCholesky (const double *para);
		void getCholesky (double *para) const;
		
		// set HouseHolder reflection matrix
		Matrix3D& setHouseHolder (const double theta, const double phi);
		Matrix3D& setHouseHolder (const double *para);
		
		// assignment operator
		Matrix3D& operator= (const Matrix3D&);
		
		// assignment conversion from symmetric tensor
		Matrix3D& operator= (const SymTensor3D&);
		
		// assignment conversion from double
		// creating diagonal matrix
		Matrix3D& operator= (double diag);
		
		// determinant
		double det () const;
		
		// make determinant equal to 1
		void factorOutDet ();
		
		// inverse
		Matrix3D& pseudoInverseEqual ();
		Matrix3D pseudoInverse () const;
		Matrix3D& inverseEqual ();
		Matrix3D inverse () const;
		
		// square root
		Matrix3D& sqrtEqual ();
		Matrix3D sqrt () const;
		
		// transpose
		Matrix3D& transposeEqual ();
		Matrix3D transpose () const;
		Matrix3D& addSelfTransposeEqual ();
		
		// similarity transformation
		Matrix3D& similarityTransformByEqual (const Matrix3D&);
		Matrix3D similarityTransformBy (const Matrix3D&) const;
		
		// transformation by left and right matrices
		Matrix3D& transformByEqual (const Matrix3D& lhs, const Matrix3D& rhs);
		Matrix3D transformBy (const Matrix3D& lhs, const Matrix3D& rhs) const;
		 
		// matrix multiplication
		Matrix3D operator* (const Matrix3D&) const;
		Matrix3D& operator*= (const Matrix3D&);
		
		// scalar multiplication
		Matrix3D& operator*= (const double rhs);
		Matrix3D operator* (const double rhs) const;
		friend Matrix3D operator* (const double lhs, const Matrix3D& rhs);
		
		// addition
		Matrix3D& operator+= (const Matrix3D& rhs);
		Matrix3D operator+ (const Matrix3D& rhs) const;
		
		// polar decomposition
		void toPolarDecomposition (Matrix3D& orthognal, Matrix3D& spd) const;
		void toLogSPD (const bool isRight = true);
		
		// frobenius distance
		double getDistanceTo (const Matrix3D& other) const;
		
		// matrix multiplies vector
		Vector3D operator* (const Vector3D&) const;
		friend Vector3D& operator*= (Vector3D&, const Matrix3D&);
		
		// to identity
		Matrix3D& toIdentity ();
		
		friend class Vector3D;
		friend class SymTensor3D;
		friend class SymMatrix3D;
		friend ostream& operator<< (ostream&, const Matrix3D&);
		
	};

}

#endif

