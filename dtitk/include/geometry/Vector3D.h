/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Vector3D.h,v $
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


#ifndef _geometry_Vector3D_H
#define _geometry_Vector3D_H

#include <iostream>
#include <cmath>
#include "Matrix3D.h"

namespace geometry {
	
	using namespace std;
	
	class Vector3D {
		public:
		static const int dim;
		static const Vector3D zero;
		
		private:
		double column[3];
		
		public:
		inline explicit Vector3D (double x = 0.0) {
			buildColumn(x, x, x);
		}
		
		inline Vector3D (double x0, double x1, double x2) {
			buildColumn(x0, x1, x2);
		}
		
		inline Vector3D (const double *rhs) {
			buildColumn(rhs[0], rhs[1], rhs[2]);
		}
		
		inline Vector3D (const int *rhs) {
			buildColumn(rhs[0], rhs[1], rhs[2]);
		}
		
		// copy constructor
		inline Vector3D (const Vector3D& rhs) {*this = rhs; }
		
		// assignment operator
		inline Vector3D& operator= (const Vector3D& rhs) {
			buildColumn(rhs[0], rhs[1], rhs[2]);
			return *this;
		}
		
		inline Vector3D& operator= (const int rhs[3]) {
			buildColumn(rhs[0], rhs[1], rhs[2]);
			return *this;
		}
		
		inline Vector3D& operator= (const double rhs[3]) {
			buildColumn(rhs[0], rhs[1], rhs[2]);
			return *this;
		}
		
		inline Vector3D& operator= (const double rhs) {
			buildColumn(rhs, rhs, rhs);
			return *this;
		}
		
		// array-like indexing
		inline double& operator[] (int index) {
			return column[index];
		}
		
		// array-like indexing for const Vector
		inline const double& operator[] (int index) const {
			return column[index];
		}
		
		inline void set (double x0, double x1, double x2) {
			buildColumn(x0, x1, x2);
		}
		
		inline void set (const double *para) {
			buildColumn(para[0], para[1], para[2]);
		}
		
		// dot product
		double dot (const Vector3D& other) const;
		
		// negation
		Vector3D& negateEqual ();
		Vector3D negate () const;
		
		Vector3D operator- (const Vector3D& rhs) const;
		Vector3D& operator-= (const Vector3D& rhs);
		Vector3D operator+ (const Vector3D& rhs) const;
		Vector3D& operator+= (const Vector3D& rhs);
		void translateBy (double x0, double x1, double x2);
		
		// normalize
		Vector3D& normalizeEqual ();
		
		// cross product
		void crossProduct (const Vector3D& e2, Vector3D& e3) const;
		void crossProduct (const Vector3D& e2, double e3[3]) const;
		
		// outer product
		void outerProduct (const Vector3D& vec, Matrix3D& mat) const;
		Matrix3D outerProduct (const Vector3D& vec) const;
		void selfOuterProduct (Matrix3D& mat) const;
		void selfOuterProduct (SymTensor3D& sym) const;
		SymTensor3D selfOuterProduct () const;
		
		double norm () const;
		double euclideanDistanceSqTo (const Vector3D& vec) const;
		
		// scalar multiplication
		Vector3D operator* (const double rhs) const;
		Vector3D& operator*= (const double rhs);
		friend Vector3D operator* (const double lhs, const Vector3D& rhs);
		Vector3D operator/ (const double rhs) const;
		Vector3D& operator/= (const double rhs);
		
		static double getSignedVolume (const Vector3D v[4]);
		static double getSignedVolume (const Vector3D& v1, const Vector3D& v2, const Vector3D& v3, const Vector3D& v4);
		
		// output
		friend ostream& operator<< (ostream& out, const Vector3D& vec);
		
		private:
		inline void buildColumn (const double x0, const double x1, const double x2) {
			column[0] = x0;
			column[1] = x1;
			column[2] = x2;
		}
		
	};
	
}

#endif

