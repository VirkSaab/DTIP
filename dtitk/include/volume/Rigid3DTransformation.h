/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Rigid3DTransformation.h,v $
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


#ifndef _volume_Rigid3DTransformation_H
#define _volume_Rigid3DTransformation_H

#include "../geometry/Rotation3D.h"

namespace geometry {
	class Rigid3D;
}

namespace volume {
	
	using namespace geometry;
	
	class Rigid3DTransformation {
		protected:
		Rotation3D transQ;
		
		Rotation3D transQInv;
		
		Rotation3D dTransQ[3];
		
		protected:
		Rigid3DTransformation();
		
		// parameters order :
		// x, y, z, theta, phi and psi
		void setTransformation (Rigid3D& transF, const double *para);
		void setTransformationAndGrad (Rigid3D& transF, const double *para);
		
		public:
		void setRotation (const double *para);
		void setRotationGrad (const double *para);
		void setRotationGrad (const Rotation3D _dTransQ[3]);
		void getRotationGrad (Rotation3D _dTransQ[3]) const;
		
	};
}

#endif

