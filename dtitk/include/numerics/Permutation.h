/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Permutation.h,v $
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


#ifndef _numerics_Permutation_h
#define _numerics_Permutation_h

#include <vector>
#include "TestStatistics.h"

namespace numerics {
	
	using namespace std;
	
	class Permutation {
		protected:
		int noOfPermutations;
		TestStatistics *ts;
		int noOfObservations;
		int noOfGroup1Observations;
		int noOfHypotheses;
		vector<int> inPermutation;
		vector<double> adjustedFDR;
		vector<double> adjustedFWER;
		
		public:

		Permutation (TestStatistics*, int noOfPerms = 1000);
		
		void setNoOfPermutations (int noOfPermutations);
		
		void run ();
		
		void getAdjustedFDR (vector<double>&) const;
		void getAdjustedFWER (vector<double>&) const;
		
		static void randomSubset (const vector<int>& input, vector<int>& output, const int k);
		
		static int randomInteger (const int n1, const int n2);

		static void sortWithIndex (const vector<double>& stats, vector<int>& sortedIndex);
		
	};
	
	typedef struct pair {
		double stat;
		int index;
	};
	
	class GreaterThan {
		public:
		bool operator() (const pair& p1, const pair& p2) {
			return p1.stat > p2.stat;
		}
	};
}

#endif

