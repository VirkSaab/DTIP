/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: TransformScalarVolume.h,v $
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


#ifndef _volume_TransformScalarVolume_H
#define _volume_TransformScalarVolume_H

#include "TransformVolume.h"
#include "../geometry/Translation3D.h"

namespace volume {

	// declaration
	template <class Transform>
	class TransformScalarVolume : public TransformVolume<double, Transform> {
		public:
		
		TransformScalarVolume (const int sz[3], bool enableGrad = false) : TransformVolume<double, Transform> (sz, enableGrad) {
			this->zero = 0.0;
			this->bg = this->zero;
		}
		
		TransformScalarVolume (const VoxelSpace& vs, bool enableGrad = false) : TransformVolume<double, Transform> (vs, enableGrad) {
			this->zero = 0.0;
			this->bg = this->zero;
		}
		
		TransformScalarVolume (const char *filename, bool enableGrad = false) : TransformVolume<double, Transform> (enableGrad) {
			if (!this->readScalarNifti(filename)) {
				cerr << "Fail to load the volume " << filename << endl;
				exit (1);
			}
			
			// has to be initialized before gradient computation
			this->zero = 0.0;
			this->bg = this->zero;
			
			if (enableGrad) {
				this->buildGradient();
			}
			
			this->setRegion();
		}
		
		~TransformScalarVolume () {}
		
		bool writeVol () {
			if (this->name.size() != 0) {
				return writeVolAs(this->name.c_str());
			} else {
				cerr << "Fail to save volume: no filename provided" << endl;
				return false;
			}
		}
		
		bool writeVolAs (const char *filename) {
			if (!this->writeScalarNifti(filename)) {
				cerr << "Fail to save the volume as " << filename << endl;
				return false;
			}
			return true;
		}
		
		void computeHistogram (const double min, const double max, const int bins, const Volume<double>& mask, int type, bool normalize) const {
			int *histogram = new int[bins];
			for (int i = 0; i < bins; ++i) {
				histogram[i] = 0;
			}
			
			_SIZE
			int bin = 0;
			_ITERATE_BEGIN
				if ((int)(mask.voxel[i][j][k]) == 1) {
					double tmp = this->voxel[i][j][k];
					if (isnan(tmp)) {
						cerr << "found a NaN value : ";
						cerr << "[ " << i << ", " << j << ", " << k;
						cerr << "] = " << tmp;
						cerr << ", set to 0.0 (to deal with spm2)" << endl;
						tmp = 0.0;
					}
					if (tmp < min || tmp > max) {
						cerr << "found a value outside the specified range : ";
						cerr << "[ " << i << ", " << j << ", " << k;
						cerr << "] = " << tmp;
						cerr << ", ignored" << endl;
					} else {
						bin = (int)((tmp - min)/(max - min) * (bins - 1));
						histogram[bin] ++;
					}
				}
			_ITERATE_END
			
			int total = 0;
			for (int i = 0; i < bins; ++i) {
				total += histogram[i];
			}
			cout << "total voxels = " << total << endl;
			
			if (type == 1) {
				int tmp1 = 0;
				for (int i = 0; i < bins; ++i) {
					int tmp2 = histogram[i];
					histogram[i] += tmp1;
					tmp1 += tmp2;
				}
				cout << "the CDF is " << endl;
			} else {
				cout << "the PDF is " << endl;
			}
			
			for (int i = 0; i < bins; ++i) {
				cout << i << '\t';
				cout << min + i*(max - min)/bins << '\t';
				if (normalize) {
					cout << histogram[i]*1.0/total << endl;
				} else {
					cout << histogram[i] << endl;
				}
			}
			
			delete[] histogram;
		}
		
		void convertToSq () {
			_SIZE
			_ITERATE_BEGIN
				this->voxel[i][j][k] *= this->voxel[i][j][k];
			_ITERATE_END
		}
		
		void sqrt () {
			_SIZE
			_ITERATE_BEGIN
				this->voxel[i][j][k] = std::sqrt(this->voxel[i][j][k]);
			_ITERATE_END
		}
		
		void exp () {
			_SIZE
			_ITERATE_BEGIN
				this->voxel[i][j][k] = std::exp(this->voxel[i][j][k]);
			_ITERATE_END
		}
		
		void log (const bool force = true) {
			_SIZE
			_ITERATE_BEGIN
				const double value = this->voxel[i][j][k];
				if (value <= 0.001) {
					if (force) {
						this->voxel[i][j][k] = std::log(0.001);
					} else {
						cerr << "negative or small value detected when computing logarithm" << endl;
						exit (1);
					}
				} else {
					this->voxel[i][j][k] = std::log(this->voxel[i][j][k]);
				}
			_ITERATE_END
		}
		
		void multiply (const Volume<double>& in) {
			_SIZE
			_ITERATE_BEGIN
				this->voxel[i][j][k] *= in.voxel[i][j][k];
			_ITERATE_END
		}
		
		void computeBoundingBox (int bb[3][2], const double bgvalue = 0.0) {
			for (int i = 0; i < 3; ++i) {
				bb[i][0] = 0;
				bb[i][1] = 0;
			}
			_SIZE
			bool isfirst = true;
			_ITERATE_BEGIN
				if (this->voxel[i][j][k] > bgvalue) {
					if (isfirst) {
						bb[0][0] = bb[0][1] = i;
						bb[1][0] = bb[1][1] = j;
						bb[2][0] = bb[2][1] = k;
						isfirst = false;
					} else {
						if (i < bb[0][0]) {
							bb[0][0] = i;
						} else if ( i > bb[0][1]) {
							bb[0][1] = i;
						}
						if (j < bb[1][0]) {
							bb[1][0] = j;
						} else if ( j > bb[1][1]) {
							bb[1][1] = j;
						}
						if (k < bb[2][0]) {
							bb[2][0] = k;
						} else if ( k > bb[2][1]) {
							bb[2][1] = k;
						}
					}
				}
			_ITERATE_END
		}
		
		Vector3D computeCenterOfMass () const {
			_SIZE
			// compute center of mass
			Vector3D com;
			double sum = 0.0;
			_ITERATE_BEGIN
				const double mass = this->voxel[i][j][k];
				sum += mass;
				Vector3D vec(i, j, k);
				this->toAbs(vec);
				com += mass * vec;
			_ITERATE_END
			const double eps = 1.0e-16;
			sum += eps;
			com *= 1.0/sum;
			return com;
		}
		
		SymTensor3D computeMomentOfInertia () const {
			_SIZE
			Vector3D com = computeCenterOfMass();
			SymTensor3D moi;
			double sum = 0.0;
			_ITERATE_BEGIN
				const double mass = this->voxel[i][j][k];
				sum += mass;
				Vector3D vec(i, j, k);
				this->toAbs(vec);
				vec -= com;
				moi[0] += mass * (vec[1] * vec[1] + vec[2] * vec[2]);
				moi[1] -= mass * vec[0] * vec[1];
				moi[2] += mass * (vec[0] * vec[0] + vec[2] * vec[2]);
				moi[3] -= mass * vec[0] * vec[2];
				moi[4] -= mass * vec[1] * vec[2];
				moi[5] += mass * (vec[0] * vec[0] + vec[1] * vec[1]);
			_ITERATE_END
			const double eps = 1.0e-16;
			sum += eps;
			moi *= 1.0/sum;
			return moi;
		}
		
		double computeComponentSimilarity (const double& s1, const double& s2) const {
			double diff = s1 - s2;
			diff *= diff;
			return diff;
		}
		
		virtual double computeComponentSimilarityGradient (const double& r0, const double& s0, double *gs0, const Vector3D& vec, double *xi) const {
			cerr << "the default placeholder implementation of " << endl;
			cerr << "TransformScalar3DVolume::computeComponentSimilarityGradient" << endl;
			return 0.0;
		}
		
		void getMaxMin (double& max, double& min) const {
			_SIZE
			max = 0.0;
			min = 0.0;
			double curr;
			_ITERATE_BEGIN
				curr = this->voxel[i][j][k];
				if (curr > max) {
					max = curr;
				} else if (curr < min) {
					min = curr;
				}
			_ITERATE_END
		}
		
		void computeStatistics (double stats[4], const Volume<double> *mask) const {
			_SIZE
			double min = 0.0;
			double max = 0.0;
			double mean = 0.0;
			int count = 0;
			if (mask == 0) {
				cout << "zero-valued voxels are ignored" << endl;
			}
			
			_ITERATE_BEGIN
				if (mask == 0) {
					if (this->voxel[i][j][k] == 0) {
						continue;
					}
				} else {
					if ((int)(mask->voxel[i][j][k]) == 0) {
						continue;
					}
				}
				
				if (count == 0) {
					max = min = this->voxel[i][j][k];
				} else {
					mean += this->voxel[i][j][k];
					if (this->voxel[i][j][k] > max) {
						max = this->voxel[i][j][k];
					} else if (this->voxel[i][j][k] < min) {
						min = this->voxel[i][j][k];
					}
				}
				count++;
			_ITERATE_END
			
			mean /= count;
			
			stats[0] = mean;
			stats[1] = min;
			stats[2] = max;
			stats[3] = count;
			
			return;
		}
		
		// mutual information
		void computeNormalizedValue (const double min, const double delta, const int bin, double& value) {
			const double eps = 1.0e-16;
			value -= min;
			value /= (delta + eps);
			value *= bin - 1;
		}
		
		void computeJointHistogramRegion (const TransformScalarVolume<Transform>& vol, const Volume<double>* mask, TransformScalarVolume<Translation3D>& hvol, const bool smooth = true) {
			const double eps = 1.0e-16;
			const int bin[2] = {hvol.getYSize(), hvol.getZSize()};
			// initialize
			for (int i = 0; i < bin[0]; ++i) {
				for (int j = 0; j < bin[1]; ++j) {
					hvol.voxel[0][i][j] = eps;
				}
			}
			hvol.setBackground(eps);
			
			int bottomLeft[3];
			int cornerIndex[2][2][2][3];
			double lambda[3];
			const double *corner[2][2][2];
			
			Vector3D vec;
			
			// get the max/min for both image volumes
			double max[2];
			double min[2];
			getMaxMin(max[0], min[0]);
			vol.getMaxMin(max[1], min[1]);
			const double delta[2] = {max[0] - min[0], max[1] - min[1]};
			
			double current = 0.0;
			double other = 0.0;
			for (int i = this->regionOriginRel[0]; i < this->regionEndRel[0]; i += this->step[0]) {
				for (int j = this->regionOriginRel[1]; j < this->regionEndRel[1]; j += this->step[1]) {
					for (int k = this->regionOriginRel[2]; k < this->regionEndRel[2]; k += this->step[2]) {
						// skip zero entries in mask volume
						if (mask != 0) {
							if ((int)(mask->voxel[i][j][k]) == 0) {
								continue;
							}
						}
						// the template object at (i,j,k)
						current = this->voxel[i][j][k];
						// scale the value
						computeNormalizedValue(min[0], delta[0], bin[0], current);
						
						// the vector at (i,j,k)
						vec[0] = i;
						vec[1] = j;
						vec[2] = k;
						this->toAbs(vec);
						
						// inverse transform the vector
						// (the input is the inverse transformation)
						vec *= this->trans;
						
						// if within range
						if (vol.computeBottomLeftCornerIndexAndLambdaRegion(vec, bottomLeft, lambda)) {
							vol.computeCornerIndices(bottomLeft, cornerIndex);
							// interpolate for the subject object
							vol.computeCornerObjects(cornerIndex, corner);
							other = this->interpolate8(corner, lambda);
							computeNormalizedValue(min[1], delta[1], bin[1], other);
							hvol.voxel[0][(int)current][(int)other] += 1.0;
						} else {
							hvol.voxel[0][(int)current][(int)this->bg] += 1.0;
						}
					}
				}
			}
			
			if (smooth) {
				const int dir[2] = {1, 2};
				const double sigma[2] = {7/std::sqrt(8*std::log(2.0)), 7/std::sqrt(8*std::log(2.0))};
				hvol.gaussianSmoothing2D(dir, sigma);
			}
			
			double sum = 0.0;
			for (int i = 0; i < bin[0]; ++i) {
				for (int j = 0; j < bin[1]; ++j) {
					sum += hvol.voxel[0][i][j];
				}
			}
			
			// normalize
			for (int i = 0; i < bin[0]; ++i) {
				for (int j = 0; j < bin[1]; ++j) {
					hvol.voxel[0][i][j] /= sum;
				}
			}
			
			return;
		}
		
		double computeMutualInformationRegion (const TransformScalarVolume<Transform>& vol, const Volume<double>* mask, const int type = 0) {
			// the joint histogram image
			// properly initialized with zero
			const int bin[2] = {256, 256};
			const int hsize[3] = {1, bin[0], bin[1]};
			TransformScalarVolume<Translation3D> hvol(hsize);
			
			// compute the histograms
			computeJointHistogramRegion(vol, mask, hvol);
			
			// the 1D projections
			const double eps = 1.0e-16;
			double *hist[2];
			hist[0] = new double[bin[0]];
			for (int i = 0; i < bin[0]; ++i) {
				hist[0][i] = eps;
			}
			hist[1] = new double[bin[1]];
			for (int i = 0; i < bin[1]; ++i) {
				hist[1][i] = eps;
			}
			
			for (int i = 0; i < bin[0]; ++i) {
				for (int j = 0; j < bin[1]; ++j) {
					hist[0][i] += hvol.voxel[0][i][j];
					hist[1][j] += hvol.voxel[0][i][j];
				}
			}
			
			// compute H(M), H(N), and H(M,N)
			double hm = 0.0;
			for (int i = 0; i < bin[0]; ++i) {
				hm -= hist[0][i] * std::log(hist[0][i]);
			}
			double hn = 0.0;
			for (int i = 0; i < bin[1]; ++i) {
				hn -= hist[1][i] * std::log(hist[1][i]);
			}
			double hmn = 0.0;
			for (int i = 0; i < bin[0]; ++i) {
				for (int j = 0; j < bin[1]; ++j) {
					hmn -= hvol.voxel[0][i][j] * std::log(hvol.voxel[0][i][j]);
				}
			}
			
			// MI = H(M) + H(N) - H(M,N)
			// NMI = MI/H(M,N)
			double metric = 0.0;
			switch (type) {
				default:
				case 0: metric = (hm + hn - hmn) / hmn;
					   break;
				case 1: metric = hm + hn - hmn;
					   break;
			}
			
			// clear up the memory
			delete[] hist[0];
			delete[] hist[1];
			
			return metric;
		}
	
	};
}

#endif

