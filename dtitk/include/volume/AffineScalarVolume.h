/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: AffineScalarVolume.h,v $
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


#ifndef _volume_AffineScalarVolume_H
#define _volume_AffineScalarVolume_H

#include "../geometry/Affine3D.h"
#include "Affine3DTransformation.h"
#include "TransformScalarVolume.h"

namespace volume {
	
	class AffineScalarVolume : public TransformScalarVolume<Affine3D>
							,  public Affine3DTransformation {
		protected:
		double computeComponentSimilarity (const double& s1, const double& s2) const;
		double computeComponentSimilarityGradient (
			const double& r0, const double& s0, double *gs0,
			const Vector3D& vec, double *xi) const;
		
		public:
		AffineScalarVolume (const int sz[3], bool enableGrad = false);
		AffineScalarVolume (const char *filename, bool enableGrad = false);
		~AffineScalarVolume () {}
		
		void setTransformation (const double *para);
		void setTransformationAndGrad (const double *para);
		
	};
	
}

#endif

