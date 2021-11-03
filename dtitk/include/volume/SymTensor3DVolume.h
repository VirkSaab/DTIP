/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: SymTensor3DVolume.h,v $
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


// SymTensor3DVolume

#ifndef _volume_SymTensor3DVolume_H
#define _volume_SymTensor3DVolume_H

#include "TransformSymTensor3DVolume.h"
#include "../geometry/Translation3D.h"

namespace geometry {
	class Matrix3D;
}

namespace volume {
	
	class ScalarVolume;
	class Vector3DVolume;
	
	class SymTensor3DVolume : public TransformSymTensor3DVolume<Translation3D> {
		public:
		SymTensor3DVolume (const int sz[3], bool enableGrad = false);
		SymTensor3DVolume (const char *filename, bool enableGrad = false);
		
		static SymTensor3DVolume *convertFromEigenSystem (const ScalarVolume *eigsSV[3], const Vector3DVolume *eigvVV[3]);
		
		static SymTensor3DVolume *average (const char *group, const string& option);
		static bool getVoxelAtMT (const vector<SymTensor3DVolume*>& vols, const Vector3D& vec, vector<SymTensor3D>& out, const int intp = 0);
	};
	
}

#endif

