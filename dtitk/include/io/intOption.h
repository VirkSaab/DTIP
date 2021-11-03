/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: intOption.h,v $
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


#ifndef _io_intOption_H
#define _io_intOption_H

#include "option.h"

namespace io {
	
	class intOption : public option {
		public:
		vector<int> defaultVar;
		vector<int> var;
		
		intOption (const string& op);
		intOption (const string& op, const size_t dim);
		intOption (const string& op, const string& initial);
		
		string type () const;
		string getDefault () const;
		string getValue () const;
		void insertValue (const string&);
		bool isValidInput ();
	};
}

#endif

