/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: ScalarVolume.h,v $
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


#ifndef _volume_ScalarVolume_H
#define _volume_ScalarVolume_H

#include "../geometry/Translation3D.h"
#include "TransformScalarVolume.h"

namespace volume {

	class ScalarVolume : public TransformScalarVolume<Translation3D> {
		public:
		ScalarVolume (const VoxelSpace& vs, bool enableGrad = false);
		ScalarVolume (const int sz[3], bool enableGrad = false);
		ScalarVolume (const char *filename, bool enableGrad = false);
		
		ScalarVolume *compute2DHistogram (const double range[2][2], const int bins[2], const ScalarVolume& in, const ScalarVolume& mask) const;
		
		double diceCoefficient (const ScalarVolume& rhs) const;
		static void average (const char *group, const char *meanName, const char *stdName);
		static void grid (const VoxelSpace& vs);
	};

}

#endif

