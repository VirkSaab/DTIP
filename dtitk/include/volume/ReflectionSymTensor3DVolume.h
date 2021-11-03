/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: ReflectionSymTensor3DVolume.h,v $
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


#ifndef _volume_ReflectionSymTensor3DVolume_H
#define _volume_ReflectionSymTensor3DVolume_H

#include "../geometry/Reflection3D.h"
#include "Reflection3DTransformation.h"
#include "TransformSymTensor3DVolume.h"

namespace volume {
	
	class ReflectionSymTensor3DVolume : public TransformSymTensor3DVolume<Reflection3D>
								,  public Reflection3DTransformation {
		protected:
		void objectSpecificTransform (SymTensor3D&) const;
		void objectSpecificTransformInverse (SymTensor3D&) const;
		
		private:
		// gradient is disabled
		double computeComponentSimilarityGradient (
			const SymTensor3D& r0, const SymTensor3D& s0, SymTensor3D *gs0,
			const Vector3D& vec, double *xi) const {
			return 0.0;
		}
		
		public:
		ReflectionSymTensor3DVolume (const int sz[3]);
		ReflectionSymTensor3DVolume (const char *filename);
		~ReflectionSymTensor3DVolume () {}
		
		// set up transformation
		void setTransformation (const double *para);
		void setTransformation (const Reflection3D& input);
		
	};
	
}

#endif

