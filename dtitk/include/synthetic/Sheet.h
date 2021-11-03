/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Sheet.h,v $
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


#ifndef _Sheet_H
#define _Sheet_H

#include "Synthetic.h"

namespace synthetic {
	
	class Sheet : public Synthetic {
		protected:
		double radius;
		double thickness;
		double length;
		double width;
		double bending;
		double eigs[3];
		double shape_deviation;
		double tensor_deviation;
		double rigid_deviation;
		
		void construct ();
		
		public:
		Sheet (const int size[3], const double vsize[3], const double eigs[3], const double alpha, const double shape_deviation, const double tensor_deviation, const double rigid_deviation, const bool random_seed = true);
		
	};

}

#endif

