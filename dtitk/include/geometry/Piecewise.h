/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Piecewise.h,v $
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


#ifndef _geometry_Piecewise_H
#define _geometry_Piecewise_H

namespace geometry {
	
	class Piecewise {
		protected:
		// current sample point (cell) index
		mutable int current[3];
		
		// origin
		double origin[3];
		// number of points per dimension
		int size[3];
		// size of the cell
		double vsize[3];
		
		public:
		Piecewise ();
		virtual ~Piecewise () {};
		
		virtual void configure (const int size[3], const double vsize[3]) = 0;
		
		// accessors
		void getSize (int size[3]) const;
		void getVSize (double vsize[3]) const;
		
		// regularization measure
		// sampled at the same locations as the image voxels
		virtual double computeDiscontinuityOfCell (const int cell[3], const double rs_vsize[3], const bool lowerHalfOnly = false) const = 0;
		double computeDiscontinuityOfCurrentCell (const double rs_vsize[3], const bool lowerHalfOnly = false) const;
		double computeDiscontinuity (const double rs_vsize[3]) const;
		
		// the last argument is the vsize of the image volume
		// we sample at the same location as the image voxels
		void build_interfaces (const int current_piece[3], int interfaces[6][3][2], const double rs_vsize[3]) const;
		
		void build_neighbor_pieces (const int current_piece[3], int neighbor_pieces[6][3]) const;
		
	};
	
}

#endif

