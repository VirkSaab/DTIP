/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: PiecewiseAffineSymTensor3DVolume.h,v $
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


#ifndef _volume_PiecewiseAffineSymTensor3DVolume_H
#define _volume_PiecewiseAffineSymTensor3DVolume_H

#include "../geometry/PiecewiseAffine3D.h"
#include "Transformation.h"
#include "TransformSymTensor3DVolume.h"

namespace geometry {
	class Affine3D;
}

namespace volume {
	
	using namespace geometry;
	
	class PiecewiseAffineSymTensor3DVolume :
		public TransformSymTensor3DVolume<PiecewiseAffine3D>,
		protected Transformation {
		protected:
		void objectSpecificTransform (SymTensor3D&) const;
		void objectSpecificTransformInverse (SymTensor3D&) const;
		
		public:
		PiecewiseAffineSymTensor3DVolume (const int sz[3]);
		PiecewiseAffineSymTensor3DVolume (const char *filename);
		~PiecewiseAffineSymTensor3DVolume () {}
		
		// set up transformation
		void setTransformation (const PiecewiseAffine3D& rhs);
		void setTransformation (const char *filename);
		
	};
	
}

#endif

