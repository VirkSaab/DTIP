/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: EddyDistortionScalarVolume.h,v $
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


#ifndef _volume_EddyDistortionScalarVolume_H
#define _volume_EddyDistortionScalarVolume_H

#include "../geometry/Affine3D.h"
#include "EddyDistortionTransformation.h"
#include "TransformScalarVolume.h"

namespace volume {
	
	class EddyDistortionScalarVolume : public TransformScalarVolume<Affine3D>
							,  public EddyDistortionTransformation {
		private:
		// intensity based calculation is disabled
		double computeComponentSimilarity (const double&, const double&) const {
			return 0.0;
		}
		
		// gradient is disabled
		double computeComponentSimilarityGradient (
			const double& r0, const double& s0, double *gs0,
			const Vector3D& vec, double *xi) const {
			return 0.0;
		}
		
		public:
		EddyDistortionScalarVolume (const int sz[3]);
		EddyDistortionScalarVolume (const char *filename);
		~EddyDistortionScalarVolume () {}
		
		void setTransformation (const double *para);
		
	};
	
}

#endif

