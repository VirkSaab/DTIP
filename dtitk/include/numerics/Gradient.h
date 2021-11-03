/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Gradient.h,v $
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


#ifndef _numerics_Gradient_H
#define _numerics_Gradient_H

#include "Function.h"

namespace numerics {
	
	class Gradient : public Function {
		protected:
		double *df;
		
		// take the params and return the derivatives
		void (*gradient) (const double[], double[]);
		
		// take the params and return the function value
		// and the derivatives
		double (*funcAndGrad) (const double[], double[]);
		
		public:
		Gradient (int noOfParams, double (*func) (const double[]), 
				void (*grad) (const double[], double[]),
				double (*fag) (const double[], double[]) );
		
		~Gradient ();
		
		inline void computeGradient (const double p[], double xi[]) {
			gradient(p, xi);
		}
		
		inline double computeFuncAndGrad (const double p[], double xi[]) {
			return funcAndGrad(p, xi);
		}
		
		// line/directional gradient
		double gradOneDim (const double stepLength);
		
		// added to support the simultaneous computation of
		// both the function and gradient values.
		// it can save about 25% recomputation
		double funcAndGradOneDim (const double stepLength, double& grad);
		 
	};
	
}

#endif

