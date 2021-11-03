/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: HierarchicalPiecewiseAffine3D.h,v $
  Language:    C++
  Date:        $Date: 2011/12/21 20:39:19 $
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


#ifndef _geometry_HierarchicalPiecewiseAffine3D_H
#define _geometry_HierarchicalPiecewiseAffine3D_H

#include "Matrix3D.h"
#include "Affine3D.h"
#include "PiecewiseAffine3DSmooth.h"
#include <vector>

namespace volume {
	class Vector3DVolume;
}

namespace geometry {
	
	using namespace std;
	using namespace volume;
	
	class Vector3D;
	
	class HierarchicalPiecewiseAffine3D {
		protected:
		mutable Matrix3D currentJacobian;
		vector<unsigned char> transType;
		vector<unsigned short> transIdx;
		bool useAffine;
		vector<Affine3D> affList;
		vector<PiecewiseAffine3DSmooth> pwaList;
		
		public:
		HierarchicalPiecewiseAffine3D () {};
		HierarchicalPiecewiseAffine3D (const char *filename, const int level = -1);
		virtual ~HierarchicalPiecewiseAffine3D () {};
		
		void setUseAffine (const bool in);
		void load (const char *filename, const int level = -1);
		void getCurrentJacobian (Matrix3D& jac) const;
		void writeAsDispField (const char *filename, const int size[3], const double vsize[3])  const;
		void writeAsJacobian (const char *filename, const int size[3], const double vsize[3], bool useLog = false) const;
		friend Vector3D& operator *= (Vector3D& lhs, const HierarchicalPiecewiseAffine3D& rhs);
		
		static Vector3DVolume *average (const char *group, const int size[3], const double vsize[3]);
	};
	
}

#endif

