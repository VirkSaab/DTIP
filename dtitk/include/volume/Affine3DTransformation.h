/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Affine3DTransformation.h,v $
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


#ifndef _volume_Affine3DTransformation_H
#define _volume_Affine3DTransformation_H

#include "../geometry/Matrix3D.h"
#include "../geometry/Rotation3D.h"
#include "../geometry/Translation3D.h"

namespace geometry {
	class Affine3D;
}

namespace volume {
	
	using namespace geometry;
	
	class Affine3DTransformation {
		protected:		
		// the matrix part of the affine transformation
		// M = Q S
		Matrix3D transM;
		
		Matrix3D transMInv;
		
		// the deformation prior determined by S
		double prior;
		
		// derivatives of the deformation prior with respect to the
		// deformation parameters
		double dPrior[6];
		
		// the rotational part of M : Q
		Rotation3D transQ;
		
		Rotation3D transQInv;
		
		// the deformation part of M : S
		// should be symmetric positive definite
		Matrix3D transS;
		
		// the translation
		Translation3D transT;
		
		// the derivatives
		Rotation3D dTransQ[3];
		
		Matrix3D dTransS[6];
		
		Matrix3D dTransM[9];
		
		private:
		void buildCommon (const double *para);
		
		void buildAffine (Affine3D& transF);
		
		void buildDerivatives (const double *para);
		
		void buildPrior (const double *para);
		
		void buildPriorDerivatives (const double *para);
		
		protected:
		Affine3DTransformation();
		
		// para in the order
		// x, y, z, theta, phi, psi, xx, yy, zz, xy, xz, yz
		void setTransformation (Affine3D& transF,
				const double* para);
		void setTransformationAndGrad (Affine3D& transF,
				const double* para);
		
		public:
		void getMatrixGrad (Matrix3D _dTransM[9]) const;
		
		double getPrior () const;
		double getPriorWithDerivatives (double *xi) const;
	};

}

#endif

