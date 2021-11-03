/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Vector3DVolume.h,v $
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


#ifndef _volume_Vector3DVolume_H
#define _volume_Vector3DVolume_H

#include "Volume.h"

namespace volume {
	
	class ScalarVolume;
	
	class Vector3DVolume : public Volume<Vector3D> {
		private:
		void convertToLog (const bool force = true) {} // not applicable
		void convertToExp () {}; // not applicable
		
		protected:
		void objectSpecificOrientationMapping (const int mapping[3], const bool dir[3]);
		
		public:
		Vector3DVolume ();
		Vector3DVolume (const int sz[3], const bool enableGrad = false);
		Vector3DVolume (const char *filename, const bool enableGrad = false);
		
		void load (const char *filename, const bool enableGrad = false);
		void load (const char *filename, const bool flip[3], const bool enableGrad = false);
		
		ScalarVolume *warp (const ScalarVolume& in) const;
		
		ScalarVolume *getNorm () const;
		double getMaxNorm() const;
		
		void replaceBy (const Vector3DVolume& in, const ScalarVolume& mask);
		
		void computeSimilarity (const Vector3DVolume& rhs, const ScalarVolume& mask, double stats[2]) const;
		
		bool writeVolAs (const char *filename, const int nifti_intent_code = NIFTI_INTENT_VECTOR);
		bool writeVol (const int nifti_intent_code = NIFTI_INTENT_VECTOR);
		
		Vector3D computeCenterOfMass () const;
		SymTensor3D computeMomentOfInertia () const;
		static Vector3DVolume *average (const char *group);
	};

}

#endif

