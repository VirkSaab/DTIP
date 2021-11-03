/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: RigidSymTensor3DVolume.h,v $
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


#ifndef _volume_RigidSymTensor3DVolume_H
#define _volume_RigidSymTensor3DVolume_H

#include "../geometry/Rigid3D.h"
#include "Rigid3DTransformation.h"
#include "TransformSymTensor3DVolume.h"

namespace volume {
	
	class RigidSymTensor3DVolume : public TransformSymTensor3DVolume<Rigid3D>
								,  public Rigid3DTransformation {
		protected:
		void objectSpecificTransform (SymTensor3D&) const;
		void objectSpecificTransformInverse (SymTensor3D&) const;
		
		double computeComponentSimilarityGradient (
			const SymTensor3D& r0, const SymTensor3D& s0, SymTensor3D *gs0,
			const Vector3D& vec, double *xi) const;
		
		public:
		RigidSymTensor3DVolume (const int sz[3], bool enableGrad = false);
		RigidSymTensor3DVolume (const char *filename, bool enableGrad = false);
		~RigidSymTensor3DVolume () {}
		
		// set up transformation
		void setTransformation (const double *para);
		void setTransformationAndGrad (const double *para);
		
	};
	
}

#endif

