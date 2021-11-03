/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: MinBracket.h,v $
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


#ifndef _numerics_MinBracket_H
#define _numerics_MinBracket_H

namespace numerics {
	
	class Function;
	
	class MinBracket {
		protected:
		static const double Gold;
		static const double GLimit;
		const double Tiny;
		
		public:
		MinBracket (double tiny = 1.0E-20);
		
		void mnbrak(double& ax, double& bx, double& cx,
					double& fa, double& fb, double& fc,
					Function& func, bool hasLastValue = false);
					
	};
	
}

#endif

