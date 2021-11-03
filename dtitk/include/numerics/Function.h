/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Function.h,v $
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


#ifndef _numerics_Function_H
#define _numerics_Function_H

namespace numerics {
	
	class Function {
		protected:
		// dimension of the parameter space
		const int dim;
		// initial point
		double *pcom;
		// line search direction
		double *xicom;
		// parameter space scaling facors
		double *scale;
		// temporay storage of point
		double *xt;
		
		// input cost function
		double (*function)(const double[]);
		
		private:
		double *setDefaultScaling ();
		
		public:
		Function (int noOfParams, double (*func)(const double[]));
		virtual ~Function ();
		
		inline int getDim() const {return dim;}
		
		// input initial point to the function
		void setInitialPoint (const double p[]);
		// input the line search directions
		void setDirection (const double xi[]);
		// input scaling factors
		void setScaling (const double s[]);
		
		inline double compute (const double p[]){
			return function(p);
		}
		
		// line function value
		double funcOneDim (const double stepLength);
		
		// get new point for the inpute stepLength
		void getPoint (double p[], const double stepLength);
	};
	
}

#endif

