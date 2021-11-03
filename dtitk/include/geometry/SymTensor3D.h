/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: SymTensor3D.h,v $
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


#ifndef _geometry_SymTensor3D_H
#define _geometry_SymTensor3D_H

#include <iostream>
#include <cmath>

namespace io {
	class SymTensor3DVTKReader;
	class SymTensor3DVTKWriter;
}

namespace geometry {
	
	//forward declaration
	class Vector3D;
	class Matrix3D;
	
	using namespace std;
	
	typedef double const *(const_ptr);
	
	class SymTensor3D {
		public:
		enum ScalarIndex {TRACE, FA, AD, RD, TSP, NORM, DTSP, DNORM, DYDISP, DYCOH, PDMP};
		enum SimilarityMeasure {EDS, ED, TP, NTP, GDS, TRS, FAS, DDS};
		enum Reorient {FS, PPD, NO};
		enum Interpolation {EI, LEI};
		
		static SimilarityMeasure SMOption;
		static Reorient ReorientOption;
		static Interpolation InterpolationOption;
		
		private:
		double matrix[6];
		
		public:
		// default constructor
		// matrix is filled with zeros
		SymTensor3D ();
		
		// when id = true, the matrix is identity
		// when id = false, the matrix is undefined
		SymTensor3D (bool id);
		
		// initialization by components
		SymTensor3D (double xx, double yx, double yy, double zx, double zy, double zz);
		
		// copy constructor
		SymTensor3D (const SymTensor3D& other);
		
		// conversion from Matrix3D
		// in general it makes no sense
		explicit SymTensor3D (const Matrix3D& other);
		
		// destructor, required!
		~SymTensor3D ();
		
		// assignment operator
		SymTensor3D& operator= (const SymTensor3D& rhs);
		SymTensor3D& operator= (const Matrix3D& other);
		
		SymTensor3D operator+ (const SymTensor3D& rhs) const;
		SymTensor3D& operator+= (const SymTensor3D& rhs);
		SymTensor3D& operator+= (const Matrix3D& rhs);
		SymTensor3D operator- (const SymTensor3D& rhs) const;
		SymTensor3D& operator-= (const SymTensor3D& rhs);
		
		void set (double xx, double yx, double yy, double zx, double zy, double zz);
		
		// array-like indexing
		inline double& operator[] (int index) {
			return matrix[index];
		}
		
		// interface to different similarity measures
		double computeSimilarity (const SymTensor3D& rhs) const;
		
		// deviatoric distance to another tensor
		double deviatoricDistanceSqTo (const SymTensor3D& other) const;
		
		// geometric distance to another tensor
		double geometricDistanceSqTo (const SymTensor3D& other) const;
		
		// euclidean distance to another tensor
		double euclideanDistanceSqTo (const SymTensor3D& other) const;
		double euclideanDistanceTo (const SymTensor3D& other) const;
		
		// tensor product
		double tensorProductWith (const SymTensor3D& other) const;
		double normalizedTensorProductWith (const SymTensor3D& other) const;
		
		// scalar indices based measures
		double traceDistanceSqTo (const SymTensor3D& rhs) const;
		double faDistanceSqTo (const SymTensor3D& rhs) const;
		
		// reorientation interface
		SymTensor3D& reorientBy (const Matrix3D& mat);
		
		// multiplication by scalar
		SymTensor3D& operator*= (const double rhs);
		friend SymTensor3D operator* (const double lhs, const SymTensor3D& rhs);
		SymTensor3D& operator/= (const double rhs);
		friend SymTensor3D operator/ (const SymTensor3D& lhs, const double rhs);
		
		// norm2
		double getNorm2 () const {
			double norm = getTSP();
			return sqrt(norm);
		}
		
		// normalization
		SymTensor3D& normalizeEqual () {
			double norm = getNorm2() + 1.0E-30;
			for (int i = 0; i < 6; ++i) {
				matrix[i] /= norm;
			}
			return *this;
		}
		
		// determinant
		inline double getDeterminant () const {
			double det = matrix[0] * matrix[2] * matrix[5];
			det -= matrix[0] * matrix[4] * matrix[4];
			det -= matrix[1] * matrix[1] * matrix[5];
			det += 2.0 * matrix[1] * matrix[3] * matrix[4];
			det -= matrix[2] * matrix[3] * matrix[3];
			return det;
		}
		
		// trace, moment1
		inline double getTrace () const {
			double trace = matrix[0];
			trace += matrix[2];
			trace += matrix[5];
			return trace;
		}
		
		// trace of squared tensor
		// equal to the sum of the square of all 9 components
		inline double getTraceSq () const {
			double sq = matrix[0] * matrix[0];
			sq += 2.0 * matrix[1] * matrix[1];
			sq += matrix[2] * matrix[2];
			sq += 2.0 * matrix[3] * matrix[3];
			sq += 2.0 * matrix[4] * matrix[4];
			sq += matrix[5] * matrix[5];
			return sq;
		}
		
		// Tensor Scalar Product
		inline double dot (const SymTensor3D& other) const {
			const_ptr matrix2 = other.matrix;
			double tsp = matrix[0] * matrix2[0];
			double tmp = matrix[1] * matrix2[1];
			tmp *= 2.0;
			tsp += tmp;
			tsp += matrix[2] * matrix2[2];
			tmp = matrix[3] * matrix2[3];
			tmp *= 2.0;
			tsp += tmp;
			tmp = matrix[4] * matrix2[4];
			tmp *= 2.0;
			tsp += tmp;
			tsp += matrix[5] * matrix2[5];
			return tsp;
		}
		
		// geometric tensor scalar product
		// a constant factor of 2/15 is dropped
		inline double geometricDot (const SymTensor3D& other) const {
			// dot product part
			double sum = dot(other);
			// trace
			double tmp = getTrace();
			tmp *= other.getTrace();
			tmp *= 0.5;
			sum += tmp;
			return sum;
		}
		
		// deviatoric tensor scalar product
		// a constant factor of 2/15 is dropped
		inline double deviatoricDot (const SymTensor3D& other) const {
			// dot product part
			double sum = dot(other);
			// trace
			double tmp = getTrace();
			tmp *= other.getTrace();
			tmp /= 3.0;
			sum -= tmp;
			return sum;
		}
		
		// Tensor Self Scalar Product
		inline double getTSP () const {
			return dot(*this);
		}
		
		// moment2
		inline double getMoment2 () const {
			double m1 = getTrace();
			m1 /= 3.0;
			m1 *= m1;
			m1 *= 9.0;
			double tsp = getTSP();
			tsp *= 2.0;
			m1 += tsp;
			m1 /= 15.0;
			return m1;
		}
		
		// geometric tensor self scalar product
		inline double getGeometricTSP () const {
			return geometricDot(*this);
		}
		
		// deviatoric tensor self scalar product
		inline double getDeviatoricTSP () const {
			return deviatoricDot(*this);
		}
		
		inline double getDeviatoricNorm2 () const {
			double norm = getDeviatoricTSP();
			return sqrt(norm);
		}
		
		// deviatoric
		SymTensor3D getDeviatoric () const;
		void convertToDeviatoric ();
		
		// anisotropy
		inline double getAnisotropy () const {
			return getFractionalAnisotropy();
		}
		
		// fractional anisotropy
		double getFractionalAnisotropy () const;
		
		// eigenvalue-eigenvector pairs overlap
		double getEEPairsOverlap (const SymTensor3D& in) const;
		
		// angular separation of principal eigenvector
		double getAngleOfPDs (const SymTensor3D& in) const;
		
		// similarity transformation
		SymTensor3D& similarityTransformByEqual (const Matrix3D& rot);
		SymTensor3D similarityTransformBy (const Matrix3D& rot) const;
		
		SymTensor3D& similarityTransformByEqual (
			const Matrix3D& rot, const Matrix3D& gRot);
		SymTensor3D similarityTransformBy (
			const Matrix3D& rot, const Matrix3D& gRot) const;
		
		// eigendecomposition
		bool toSPD ();
		SymTensor3D& log (bool force = true);
		SymTensor3D& exp ();
		double getAD () const;
		double getRD () const;
		void getEigenSystem (double eigs[3], Vector3D eigv[3]) const;
		void getEigenvalues (double eigs[3]) const;
		void getEigenvectors (Vector3D eigv[3]) const;
		void eigenDecomposition (Vector3D& eig, Matrix3D& eigv) const;
		void eigenDecomposition (double eigs[3], double eigv[][3]) const;
		void eigvOrder (const double eigs[3], int index[3]) const;
		void getLPSMeasures (double lps[3]) const;
		void toDyadicTensor (int index);
		double getDyadicDispersion () const;
		double getDyadicCoherence () const;
		double getPD (Vector3D& pd, const bool useFA = 0) const;
		double getPDColorCode (Vector3D& color, const bool useFA = 1, const double scale = 1.0) const;
		int getPDMaxProjection (const double faThreshold) const;
		SymTensor3D PPDTransformBy (const Matrix3D& trans) const;
		SymTensor3D& PPDTransformByEqual (const Matrix3D& trans);
		SymTensor3D& build (const double eigs[3], const double eigv[][3]);
		SymTensor3D& build (const double eigs[3], const Vector3D eigv[3]);
		
		// flip
		SymTensor3D& flip (const int direction);
	   		
		// check for NaN
		bool isNaN () const;
		
		// option controls
		static SimilarityMeasure getSMOption ();
		static void setSMOption (const SimilarityMeasure sm);
		static Reorient getReorientOption ();
		static void setReorientOption (const Reorient r);
		static Interpolation getInterpolationOption ();
		static void setInterpolationOption (const Interpolation in);
		
		static SimilarityMeasure StrToSMOption (const char *str);
		static Reorient StrToReorientOption (const char *str);
		static Interpolation StrToInterpolationOption (const char *str);
		static void SMOptionToStr (SimilarityMeasure sm, char *str);
		static void ReorientOptionToStr (Reorient ro, char *str);
		static void InterpolationOptionToStr (Interpolation in, char *str);
		static void sort (const SymTensor3D *corner[2][2][2][2]);
		
		friend class Matrix3D;
		friend class io::SymTensor3DVTKReader;
		friend class io::SymTensor3DVTKWriter;
		friend ostream& operator<< (ostream&, const SymTensor3D& rhs);
	};
	
}

#endif

