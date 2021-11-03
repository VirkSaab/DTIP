/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: SymMatrix.h,v $
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

#ifndef _numerics_SymMatrix_h
#define _numerics_SymMatrix_h

namespace numerics {

	class SymMatrix {
		public:
		
		// input:
		// in -- the symmetric matrix, only the upper triangle needs be given
		// dim -- the dimension
		// 
		// output:
		// in -- the lower triangle of the cholesky factor
		// diag -- the diagonal components of the cholesky factor
		// return value -- positive definite or not
		static bool cholesky (double **in, int dim, double *diag);
		
		// input:
		// in -- the cholesky factor previously computed
		// dim -- the dimension
		// diag -- the cholesky factor as well
		// 
		// output:
		// in -- the inverted cholesky factor lower triangle and the diagonals
		static void invertCholesky (double **in, int dim, double *diag);
		
		// input:
		// in -- the symmetric matrix, only the upper triangle needs to be given
		// dim -- the dimension
		// 
		// output:
		// in -- its inverse, the upper trangle
		// diag -- the diagonal of the inverse
		static bool choleskyInvert (double **in, int dim, double *diag);
		
		// input:
		// in -- the symmetric matrix in full matrix form
		// dim -- the dimension of row(column)
		// 
		// output:
		// in -- the orthogonal matrix that renders the reduction
		// diag -- pre-allocated vector of size "dim" saving
		//         the computed diagonal
		// subDiag -- pre-allocated vector of size "dim" saving
		//         the computed subDiagonal. "0" element is a dummy
		static void tridiagonalReduction (
			double **in, int dim, double* diag, double* subDiag);
		
		// input:
		// diag -- the diagonal
		// subDiag -- the subdiagonal with "0" element a dummy
		// dim -- the dimension of the diagonal
		// eigv -- the orthogonal matrix rendering the tridiagonal
		//         form or a dummy
		// 
		// output:
		// diag -- the eigenvalues
		// eigv -- the orthogonal matrix containing the eigenvectors
		static bool tridiagonalEigenDecomposition (
			double *diag, double *subDiag, int dim, double **eigv);
		
	};

}

#endif

