/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Rotation3D.h,v $
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


#ifndef _geometry_Rotation3D_H
#define _geometry_Rotation3D_H

#include "Matrix3D.h"

namespace geometry {
	
	class Vector3D;
	
	class Rotation3D : public Matrix3D {
		private:
		void buildMatrix (const double theta, const double phi, const double psi);
		void buildThetaDeriv (const double theta, const double phi, const double psi);
		void buildPhiDeriv (const double theta, const double phi, const double psi);
		void buildPsiDeriv (const double theta, const double phi, const double psi);
		
		public:
		Rotation3D (const double theta, const double phi, const double psi);
		Rotation3D (const double *para);
		Rotation3D (const Rotation3D&);
		Rotation3D ();

		/**
		 * assignment
		 */
		Rotation3D& operator= (const Rotation3D&);
		
		/**
		 * multiplication
		 */
		Rotation3D operator* (const Rotation3D&) const;
		
		Rotation3D& setRotation (const double theta, const double phi, const double psi);
		Rotation3D& setRotation (const double *para);
		
		Rotation3D& transposeEqual ();
		Rotation3D transpose () const;
		
		/**
		 * the Matrix3D implementation is shielded by
		 * the multiplication between Rotation3D
		 */
		Vector3D operator* (const Vector3D& vec) const;
		
		/**
		 * derivatives
		 */
		Rotation3D& toThetaDeriv (const double theta, const double phi, const double psi);
		Rotation3D& toThetaDeriv (const double *para);
		Rotation3D& toPhiDeriv (const double theta, const double phi, const double psi);
		Rotation3D& toPhiDeriv (const double *para);
		Rotation3D& toPsiDeriv (const double theta, const double phi, const double psi);
		Rotation3D& toPsiDeriv (const double *para);
		
		static void toEulerAngles (const Matrix3D& in, double angles[3]);
	};

}

#endif

