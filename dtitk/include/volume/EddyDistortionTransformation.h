/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: EddyDistortionTransformation.h,v $
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


#ifndef _volume_EddyDistortionTransformation_H
#define _volume_EddyDistortionTransformation_H

#include "../geometry/geometry.h"

namespace volume {
	using namespace geometry;
	
	class EddyDistortionTransformation {
		protected:		
		Matrix3D transM;
		
		Matrix3D transMInv;
		
		/**
		 * the rotational part of M : Q, motion
		 */
		Rotation3D transQ;
		
		Rotation3D transQInv;
		
		/**
		 * the shear and scaling parts of M : S, gradient induced and along y only
		 */
		Matrix3D transS;
		
		/**
		 * the translation, motion + gradient induced
		 */
		Translation3D transT;
		
		private:
		void buildCommon (
				double x, double y, double z,
				double theta, double phi, double psi,
				double yy, double xy, double yz);
		
		void buildAffine (Affine3D& transF);
		
		protected:
		EddyDistortionTransformation();
		
		void setTransformation (Affine3D& transF,
				double x, double y, double z,
				double theta, double phi, double psi,
				double yy, double xy, double yz);
		void setTransformation (Affine3D& transF,
				const double* para);
		
	};

}

#endif

