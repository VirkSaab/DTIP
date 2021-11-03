/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: LineMinimizer.h,v $
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


#ifndef _numerics_LineMinimizer_H
#define _numerics_LineMinimizer_H

namespace numerics {
	
	class Function;
	class Gradient;
	
	class LineMinimizer {
		private:
		const double Tol;
		const bool Debug;
		
		public:
		LineMinimizer (bool debug = false);
		LineMinimizer (double tol, bool debug = false);
		
		// params and xi store and return new values
		// after the execution
		double standard (double params[], double xi[],
			Function& func, bool hasLastValue = false, double lastValue = 0.0);
		
		// params and xi store and return new values
		// after the execution
		double derivative (double params[], double xi[],
			Gradient& func);
		
	};
		
}

#endif

