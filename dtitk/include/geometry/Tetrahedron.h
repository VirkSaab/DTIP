/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Tetrahedron.h,v $
  Language:    C++
  Date:        $Date: 2011/12/21 20:39:20 $
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


#ifndef _geometry_Tetrahedron_H
#define _geometry_Tetrahedron_H

#include "Vector3D.h"

namespace geometry {
	
	class Tetrahedron {
		public:
		Vector3D vertices[4];
		
		public:
		Tetrahedron () {};
		Tetrahedron (const Vector3D& v1, const Vector3D& v2, const Vector3D& v3, const Vector3D& v4);
		Tetrahedron (const Vector3D v[4]);
		~Tetrahedron () {};

		// array-like indexing
		inline Vector3D& operator[] (int index) {
			return vertices[index];
		}
		
		inline const Vector3D& operator[] (int index) const {
			return vertices[index];
		}
		
		double getSignedVolume () const;
		
		void getBoundingBox (int bb[3][2]) const;
		
		Vector3D getBarycentricCombination (const Vector3D& v1, const Vector3D& v2, const Vector3D& v3, const Vector3D& v4, const double bc[4]) const;
		void getBarycentricCombination (const Vector3D& v1, const Vector3D& v2, const Vector3D& v3, const Vector3D& v4, const double bc[4], Vector3D& vec) const;
		Vector3D getBarycentricCombination (const Vector3D comp[4], const Vector3D& pt) const;
		Vector3D getBarycentricCombination (const Vector3D comp[4], const double bc[4]) const;
		
		void getBarycentricCoords (const Vector3D& pt, double bc[4]) const;
		bool getBarycentricCoordsInside (const Vector3D& pt, double bc[4]) const;
		
		static void getVertexIndex (const bool type, const int tetrahedronIndex, int vertexIndex[4][3]);
		static const int checker[2][5][4][3];
		
		friend ostream& operator<< (ostream& out, const Tetrahedron& tetra);
	};
}

#endif

