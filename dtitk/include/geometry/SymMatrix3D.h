/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: SymMatrix3D.h,v $
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

#ifndef _geometry_SymMatrix3D_h
#define _geometry_SymMatrix3D_h

namespace geometry {

	class Matrix3D;
	
	class SymMatrix3D {
		public:
		
		/**
		 * input:
		 * in -- the symmetric matrix, only the upper triangle needs be given
		 * 
		 * output:
		 * in -- the lower triangle of the cholesky factor (without diagonal)
		 * diag -- the diagonal part of the cholesky factor
		 * return value -- positive definite or not
		 */
		static bool cholesky (double in[][3], double diag[3]);
		
		/**
		 * input:
		 * in -- the cholesky factor previously computed
		 * diag -- the diagonal part of the factor
		 * 
		 * output:
		 * in -- the inverted cholesky factor lower triangle
		 */
		static void invertCholesky (double in[][3], double diag[3]);
		
		/**
		 * input:
		 * in -- the symmetric matrix, only the upper triangle needs to be given
		 * 
		 * output:
		 * in -- its inverse
		 */
		static bool choleskyInvert (double in[][3]);
		static bool choleskyInvert (Matrix3D& mat);
		
		/**
		 * input:
		 * in -- the symmetric matrix in full matrix form
		 * 
		 * output:
		 * in -- the orthogonal matrix that renders the reduction
		 * diag -- pre-allocated vector of size "dim" saving
		 *         the computed diagonal
		 * subDiag -- pre-allocated vector of size "dim" saving
		 *         the computed subDiagonal. "0" element is a dummy
		 */
		static void tridiagonalReduction (
			double in[][3], double diag[3], double subDiag[3]);
		
		/**
		 * input:
		 * diag -- the diagonal
		 * subDiag -- the subdiagonal with "0" element a dummy
		 * eigv -- the orthogonal matrix rendering the tridiagonal
		 *         form or a dummy
		 * 
		 * output:
		 * diag -- the eigenvalues
		 * eigv -- the orthogonal matrix containing the eigenvectors
		 */
		static bool tridiagonalEigenDecomposition (
			double diag[3], double subDiag[3], double eigv[][3]);

		/*
		 * square root
		 *
		 * input:
		 * in -- the symmetric matrix
		 *
		 * output:
		 * in -- the orthogonal matrix containing the eigenvectors
		 * eigs -- the square roots of the eigenvalues
		 */
		static void matrixSqrt (double in[][3], double eigs[3]);
	};

}

#endif

