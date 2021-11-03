/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: option.h,v $
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


#ifndef _io_option_H
#define _io_option_H

#include <iostream>
#include <vector>
#include <string>

namespace io {
	
	using namespace std;
	
	class option {
		public:
		string op;
		bool inputRequired;
		bool withDefault;
		string description;
		bool inputFound;
		size_t dim;
		
		option ();
		option (const string& op, const size_t dim = 0);
		option (const option& rhs);
		virtual ~option () {}
		option& operator= (const option& rhs);
		
		virtual string type () const {return "flag";}
		virtual string getDefault () const {return "";}
		virtual string getValue () const {return "";}
		virtual void insertValue (const string&) {};
		virtual bool isValidInput () {return true;}
		
		friend ostream& operator << (ostream& out, const option& opt);
		static void parseOptions (int, char **, vector<option*>&);
		static void printUsage (string, vector<option*>&);
		static void printReleaseStatus (const long, const long);
		static void checkReleaseStatus (const long, const long);
		static bool isOption (const char *input);
	};
	
}

#endif

