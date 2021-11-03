/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: VoxelSpace.h,v $
  Language:    C++
  Date:        $Date: 2012/08/01 16:36:34 $
  Version:     $Revision: 1.2 $

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


// class that captures voxel space definitions

#ifndef _volume_VoxelSpace_H
#define _volume_VoxelSpace_H

#include "../geometry/Matrix3D.h"
#include "nifti1_io.h"
#include <iostream>
#include <vector>

namespace geometry {
	class Vector3D;
}

namespace volume {
	
	using namespace geometry;
	
	class VoxelSpace {
		protected:
		string name;
		int size[3];
		double vsize[3];
		double origin[3];
		
		// step size when iteration through the volume
		int step[3];
		
		// region definitions
		double regionOriginAbs[3];
		int regionOriginRel[3];
		int regionSize[3];
		int regionEndRel[3]; // exclusive
		double regionCenterRel[3];
		
		// inverse matrix used in the tetrahedron
		mutable Matrix3D tetraInv[2][5];
		mutable bool tetraInvBuilt;
		
		private:
		void buildTetraInv (const int tetrahedron[2][5][4][3]) const;
		
		public:
		VoxelSpace ();
		VoxelSpace (const char *filename);
		VoxelSpace (const int size[3]);
		VoxelSpace (const VoxelSpace& vs);
		VoxelSpace (const vector<int>& size, const vector<double>& vsize, const vector<double>& origin);
		
		string getName () const;
		void setName (const char *);
		void setName (const string&);
		int getXSize () const;
		int getYSize () const;
		int getZSize () const;
		void getSize (int[3]) const;
		bool checkSize (const int size[3]) const;
		
		double getXVSize () const;
		double getYVSize () const;
		double getZVSize () const;
		void getVSize (double[3]) const;
		void setXVSize (double);
		void setYVSize (double);
		void setZVSize (double);
		void setVSize (const double[3]);
		
		void getOrigin (double[3]) const;
		void setOrigin (const double[3]);
		
		// nifti related
		void fromNifti (const char *filename);
		nifti_image *fromNifti (const char *filename, const int dim, const int intent_code);
		void printNiftiInfo (const char *filename);
		nifti_image *toNifti (const char *filename, const int dim, const int intent_code) const;
		void getOrientationCode (const nifti_image *nim, int nifti_orient_code[3]) const;
		void getOrientationAxeAndDir (const nifti_image *nim, int orientationAxe[3], bool orientationDir[3]) const;
		string getOrientationString (const nifti_image *nim) const;
		void convertToLPI (const nifti_image *nim, int mapping[3], bool dir[3]) const;
		void remap (const int mapping[3]);
		
		void setXStep (int);
		void setYStep (int);
		void setZStep (int);
		void setStepSize (const double sep[3]);
		
		void getRegionSize (int[3]) const;
		void getRegionOriginRel (int[3]) const;
		void getRegionOriginAbs (double[3]) const;
		void getRegionCenterRel (double[3]) const;
	 	
		// default set region to the whole volume
		void setRegion ();
		void setRegion (const int[3], const int[3]);
		
		// in terms of absolute scale
		void toAbs (Vector3D&) const;
		// in terms of grid scale
		void toRel (Vector3D&) const;
		
		// compute smoothing kernel sigma for given sampling rate
		// sampling rate specified in mm unit
		void computeSigma (const double sep[3], double sigma[3]) const;
		
		// compute the bottom left corner and the lambda's for interpolation
		// given a vector in space
		// 
		// input: vec
		// output: bottomLeft, lambda and the return value
		// return value is true if within range
		bool computeBottomLeftCornerIndexAndLambda (
			const Vector3D& abs, int bottomeLeft[3], double lambda[3]) const;
		// for region
		bool computeBottomLeftCornerIndexAndLambdaRegion (
			const Vector3D& abs, int bottomeLeft[3], double lambda[3]) const;
		
		// Compute the indices for the corners of an interpolation cube
		// given the bottom left corner index
		// 
		// input: bottomLeft
		// output: cornersIndex
		void computeCornerIndices (const int bottomLeft[3],
			int cornerIndex[2][2][2][3]) const;
		
		// Compute the tetrahedron membership
		// returning vertexIndex relative to the cube
		void computeTetrahedronVertexIndices (const int bottomLeft[3], const double lambda[3], int vertexIndex[4][3], Matrix3D& inverse) const;
		
		static bool parseOrientationCode (const string& orientation_code, int orientationAxe[3], bool orientationDir[3]);
		static void computeOrientationMapping (int orientationAxes[2][3], bool orientationDirs[2][3], int mapping[3], bool dir[3]);
		static void computeOrientationMapping (const char* input, const char *output, int mapping[3], bool dir[3]);
		static void computeSize (const int size[3], const double vsize[3], const double out_vsize[3], int out_size[3]);
		
		friend ostream& operator<< (ostream&, const VoxelSpace&);
	};
	
}

#endif

