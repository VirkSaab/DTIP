/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: DeformationField3D.h,v $
  Language:    C++
  Date:        $Date: 2011/12/21 20:39:21 $
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


#ifndef _volume_DeformationField3D_H
#define _volume_DeformationField3D_H

#include "../geometry/Matrix3D.h"
#include "Vector3DVolume.h"

namespace geometry {
	class Vector3D;
	class Affine3D;
}

namespace volume {
	
	class ScalarVolume;
	class SymTensor3DVolume;
	using namespace geometry;
	
	class DeformationField3D : public Vector3DVolume {
		public:
		enum Derivative {FD, TH, NO};
		
		static Derivative DerivativeOption;
		
		protected:
		mutable Matrix3D currentJacobian;
		
		void computeInverseInsideTetrahedron (const int vertexIndex[4][3], const int cornerIndex[2][2][2][3], const Vector3D *corner[2][2][2], Vector3D ***iVol) const;
		
		public:
		DeformationField3D ();
		DeformationField3D (const int sz[3]);
		DeformationField3D (const char *filename);
		
		DeformationField3D& leftCompositionWith (const Affine3D& aff);
		DeformationField3D& rightCompositionWith (const Affine3D& aff);
		DeformationField3D& rightCompositionWith (const DeformationField3D& lhs);
		void getCurrentJacobian (Matrix3D& jac) const;
		ScalarVolume *getJacobian (const bool useLog, const ScalarVolume *mask = NULL) const;
		void getJacobianComponents (ScalarVolume *jcs[3], const SymTensor3DVolume& altas, const ScalarVolume& mask) const;
		
		void computeTH (Vector3D& vec) const;
		void computeFD (Vector3D& vec) const;
		void compute (Vector3D& vec) const;
		
		static Derivative StrToDerivativeOption (const char *str);
		static void setDerivativeOption (const Derivative in);
		
		DeformationField3D *computeSquareRoot () const;
		DeformationField3D& squaring(const int iterations);
		void convertToDiffeomorphic (const int smcycles = 1);
		DeformationField3D *computeInverse () const;
		
		friend Vector3D operator* (const DeformationField3D& lhs, const Vector3D& rhs);
		friend Vector3D& operator*= (Vector3D& lhs, const DeformationField3D& rhs);
	};
}

#endif

