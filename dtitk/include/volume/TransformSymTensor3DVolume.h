/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: TransformSymTensor3DVolume.h,v $
  Language:    C++
  Date:        $Date: 2012/07/31 21:15:35 $
  Version:     $Revision: 1.2 $

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


#ifndef _volume_TransformSymTensor3DVolume_H
#define _volume_TransformSymTensor3DVolume_H

#include "TransformVolume.h"
#include "../geometry/Translation3D.h"
#include "../io/SymTensor3DVTKReader.h"
#include "../io/SymTensor3DVTKWriter.h"
#include "../io/Endian.h"
#include "ScalarVolume.h"
#include "Vector3DVolume.h"

namespace volume {
	
	using io::SymTensor3DVTKReader;
	using io::SymTensor3DVTKWriter;
	using io::Endian;
	
	// declaration
	template <class Transform>
	class TransformSymTensor3DVolume : public TransformVolume<SymTensor3D, Transform> {
		protected:
		void objectSpecificOrientationMapping (const int mapping[3], const bool dir[3]) {
			_SIZE
			_ITERATE_BEGIN
			// back up first
			double input[3][3];
			input[0][0] = this->voxel[i][j][k][0];
			input[1][0] = this->voxel[i][j][k][1];
			input[1][1] = this->voxel[i][j][k][2];
			input[2][0] = this->voxel[i][j][k][3];
			input[2][1] = this->voxel[i][j][k][4];
			input[2][2] = this->voxel[i][j][k][5];
			// apply the necessary transformation
			double output[3][3];
			for (int m = 0; m < 3; ++m) {
				for (int n = 0; n <= m; ++n) {
					if ((dir[m] && dir[n]) || (!dir[m] && !dir[n])) {
						output[mapping[m]][mapping[n]] = input[m][n];
						output[mapping[n]][mapping[m]] = input[m][n];
					} else {
						output[mapping[m]][mapping[n]] = -input[m][n];
						output[mapping[n]][mapping[m]] = -input[m][n];
					}
				}
			}
			// copy back to the internal storage
			this->voxel[i][j][k][0] = output[0][0];
			this->voxel[i][j][k][1] = output[1][0];
			this->voxel[i][j][k][2] = output[1][1];
			this->voxel[i][j][k][3] = output[2][0];
			this->voxel[i][j][k][4] = output[2][1];
			this->voxel[i][j][k][5] = output[2][2];
			_ITERATE_END
		}
		
		public:
		TransformSymTensor3DVolume (const int sz[3], bool enableGrad = false) : TransformVolume<SymTensor3D, Transform> (sz, enableGrad) {}
		
		TransformSymTensor3DVolume (const VoxelSpace& vs, bool enableGrad = false) : TransformVolume<SymTensor3D, Transform> (vs, enableGrad) {}
		
		TransformSymTensor3DVolume (const char *filename, bool enableGrad = false) : TransformVolume<SymTensor3D, Transform> (enableGrad) {
			string suffix(filename);
			string::size_type pos = suffix.rfind(".");
			suffix = suffix.substr(pos);
			if (suffix == ".vtk") {
				SymTensor3DVTKReader in(filename);
				if (!this->readVol(in)) {
					cerr << "Fail to build SymTensor3D volume from ";
					cerr << filename << endl;
					exit(1);
				}
			} else {
				if (!this->readVectorialNifti(filename, 6, NIFTI_INTENT_SYMMATRIX)) {
					cerr << "Fail to load the volume " << filename << endl;
					exit(1);
				}
			}
			
			if (enableGrad) {
				this->buildGradient();
			}
			
			this->setRegion();
		}
		
		~TransformSymTensor3DVolume () {}
		
		bool writeVol () {
			if (this->name.size() != 0) {
				return writeVolAs(this->name.c_str());
			} else {
				cerr << "Fail to save volume: no filename provided" << endl;
				return false;
			}
		}
		
		bool writeVolAs (const char *filename) {
			string suffix(filename);
			string::size_type pos = suffix.rfind(".");
			suffix = suffix.substr(pos);
			if (suffix == ".vtk") {
				SymTensor3DVTKWriter out(filename);
				if (!this->writeVTKVol(out, this->voxel)) {
					cerr << "Fail to write volume to ";
					cerr << filename << endl;
					return false;
				}
			} else {
				if (!this->writeVectorialNifti(filename, 6, NIFTI_INTENT_SYMMATRIX)) {
					cerr << "Fail to save the volume as " << filename << endl;
					return false;
				}
			}
			
			return true;
		}
		
		bool saveAsDTIStudio (const char *prefix) {
			// flip along y axis
			flip(0, 1, 0);
			string aniFile(prefix);
			aniFile += ".ani";
			string vecFile(prefix);
			vecFile += ".vec";
			// writing the files
			cout << "Writing the DTIStudio FA map as " << aniFile << endl;
			cout << "Writing the DTIStudio Principal Direction map as " << vecFile << endl;
			FILE *fpAni = fopen(aniFile.c_str(), "w");
			FILE *fpVec = fopen(vecFile.c_str(), "w");
			Vector3D pd;
			float pdFLT[3];
			_SIZE
			// can't use the _ITERATE_BEGIN/_END macros
			// the loop ordering is different
			for (int k = 0; k < zsize; ++k) {
				for (int j = 0; j < ysize; ++j) {
					for (int i = 0; i < xsize; ++i) {
						float fa = (float)this->voxel[i][j][k].getAnisotropy();
						this->voxel[i][j][k].getPD(pd);
						fa = Endian::endianNeutral(fa);
						fwrite(&fa, sizeof(float), 1, fpAni);
						for (int m = 0; m < 3; ++m) {
							pdFLT[m] = (float)pd[m];
							pdFLT[m] = Endian::endianNeutral(pdFLT[m]);
						}
						fwrite(pdFLT, sizeof(float), 3, fpVec);
					}
				}
			}
			fclose(fpAni);
			fclose(fpVec);
			cout << "Done" << endl;
			return true;
		}
		
		void computeTransformGeneric (Volume<SymTensor3D>& out, const bool backward, const int intp = 0) {
			if (intp == 0 && SymTensor3D::InterpolationOption == SymTensor3D::LEI) {
				log();
				const double value = std::log(0.001);
				SymTensor3D bg(value, 0.0, value, 0.0, 0.0, value);
				this->setBackground(bg);
				out.setBackground(bg);
				TransformVolume<SymTensor3D, Transform>::computeTransformGeneric(out, backward, intp);
				exp();
				out.exp();
			} else {
				TransformVolume<SymTensor3D, Transform>::computeTransformGeneric(out, backward, intp);
			}
		}
		
		void computeTransform (Volume<SymTensor3D>& out, const int intp = 0) {
			computeTransformGeneric(out, true, intp);
		}
		
		double computeComponentSimilarity (const SymTensor3D& s1, const SymTensor3D& s2) const {
			return s1.computeSimilarity(s2);
		}
		
		virtual double computeComponentSimilarityGradient (const SymTensor3D& r0, const SymTensor3D& s0, SymTensor3D *gs0, const Vector3D& vec, double *xi) const {
			cerr << "the default placeholder implementation of " << endl;
			cerr << "TransformSymTensor3DVolume::computeComponentSimilarityGradient" << endl;
			return 0.0;
		}
		
		void gaussianSmoothing (const double sigma = 1.0) {
			const double sigma3[3] = {sigma, sigma, sigma};
			gaussianSmoothing(sigma3);
		}
		
		void gaussianSmoothing (const double sigma[3]) {
			if (SymTensor3D::InterpolationOption == SymTensor3D::LEI) {
				log(true);
				Volume<SymTensor3D>::gaussianSmoothing(sigma);
				exp();
			} else {
				Volume<SymTensor3D>::gaussianSmoothing(sigma);
			}
		}
		
		void isNaN () const {
			cout << "Identifying NaN voxels in " << this->name << " ... " << flush;
			clock_t t1 = clock();
			_SIZE
			// to record the NaN voxels
			ScalarVolume sv(this->size);
			sv.setVSize(this->vsize);
			sv.setOrigin(this->origin);
			int count = 0;
			_ITERATE_BEGIN
				if (this->voxel[i][j][k].isNaN()) {
					count++;
					sv.voxel[i][j][k] = 1.0;
				} else {
					sv.voxel[i][j][k] = 0.0;
				}
			_ITERATE_END
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			sv.setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(),"NaN"));
			sv.writeVol();
			cout << "NaN tensors count = " << count << endl;
			return;
		}
		
		void log (const bool force = true) {
			_SIZE
			_ITERATE_BEGIN
				this->voxel[i][j][k].log(force);
			_ITERATE_END
			return;
		}
		
		void exp () {
			_SIZE
			_ITERATE_BEGIN
				this->voxel[i][j][k].exp();
			_ITERATE_END
			return;
		}
		
		void spd () {
			cout << "Converting " << this->name;
			cout << " to be symmetric positive definite ... " << flush;
			clock_t t1 = clock();
			_SIZE
			// to record the non-spd voxels
			ScalarVolume sv(this->size);
			sv.setVSize(this->vsize);
			sv.setOrigin(this->origin);
			SymTensor3D tmp;
			int count = 0;
			_ITERATE_BEGIN
				tmp = this->voxel[i][j][k];
				if (!this->voxel[i][j][k].toSPD()) {
					count++;
					sv.voxel[i][j][k] = 1.0;
				} else {
					sv.voxel[i][j][k] = 0.0;
				}
			_ITERATE_END
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			sv.setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(),"nonSPD"));
			sv.writeVol();
			this->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "spd"));
			cout << "non-spd tensors count = " << count << endl;
			return;
		}
		
		void dyadic (int index) {
			cout << "Converting " << this->name;
			cout << " to dyadic of eigenvector " << index + 1;
			cout << " ... " << flush;
			clock_t t1 = clock();
			_SIZE
			_ITERATE_BEGIN
				this->voxel[i][j][k].toDyadicTensor(index);
			_ITERATE_END
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			this->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "dyadic"));
			return;
		}
		
		void deviatoric () {
			cout << "Converting " << this->name;
			cout << " to deviatoric tensors ... " << flush;
			clock_t t1 = clock();
			_SIZE
			_ITERATE_BEGIN
				this->voxel[i][j][k].convertToDeviatoric();
			_ITERATE_END
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			this->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "dev"));
			return;
		}
		
		void convertToNormalized () {
			_SIZE
			_ITERATE_BEGIN
				this->voxel[i][j][k].normalizeEqual();
			_ITERATE_END
			return;
		}	
		
		void maskBy (const Volume<double>& mask, double mean = 0.0, double stdev = 0.0) {
			cout << "Masking " << this->name << " by ";
			cout << mask.getName() << " ... " << flush;
			clock_t t1 = clock();
			_SIZE
			_ITERATE_BEGIN
				if (mask.voxel[i][j][k] == 0) {
					if (mean > 0.0) {
						this->voxel[i][j][k][0] = mean + 0.01*(rand()%100)*stdev;
						this->voxel[i][j][k][1] = 0.01*(rand()%100)*stdev;
						this->voxel[i][j][k][2] = mean + 0.01*(rand()%100)*stdev;
						this->voxel[i][j][k][3] = 0.01*(rand()%100)*stdev;
						this->voxel[i][j][k][4] = 0.01*(rand()%100)*stdev;
						this->voxel[i][j][k][5] = mean + 0.01*(rand()%100)*stdev;
					} else {
						this->voxel[i][j][k] = this->bg;
					}
				}
			_ITERATE_END
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			this->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "masked"));
			return;
		}
		
		void replaceBy (const Volume<SymTensor3D>& in, const Volume<double>& mask) {
			_SIZE
			_ITERATE_BEGIN
				if (mask.voxel[i][j][k] == 1) {
					this->voxel[i][j][k] = in.voxel[i][j][k];
				}
			_ITERATE_END
			return;
		}
		
		TransformSymTensor3DVolume<Transform>& operator= (const Volume<SymTensor3D>& rhs) {
			this->Volume<SymTensor3D>::operator=(rhs);
			return *this;
		}
		
    TransformSymTensor3DVolume<Transform>& replacementWithReflection (const Volume<double>* mask, const Reflection3D& trans) {
      this->Volume<SymTensor3D>::replacementWithReflection(mask, trans);
      Matrix3D transM;
      trans.getMatrix(transM);
			_SIZE
			_ITERATE_BEGIN
				if (mask->voxel[i][j][k] == 1) {
					this->voxel[i][j][k].similarityTransformByEqual(transM);
				}
			_ITERATE_END
      return *this;
    }

		void extractRegionOf (const Volume<SymTensor3D>& in, const char *align) {
			if (SymTensor3D::InterpolationOption == SymTensor3D::LEI) {
				int lsize[3];
				double lvsize[3];
				double lorigin[3];
				in.getSize(lsize);
				in.getVSize(lvsize);
				in.getOrigin(lorigin);
				TransformSymTensor3DVolume<Translation3D> internal(lsize);
				internal.setVSize(lvsize);
				internal.setOrigin(lorigin);
				internal = in;
				internal.log();
				const double value = std::log(0.001);
				SymTensor3D bg(value, 0.0, value, 0.0, 0.0, value);
				internal.setBackground(bg);
				this->setBackground(bg);
				Volume<SymTensor3D>::extractRegionOf(internal, align);
				exp();
			} else {
				Volume<SymTensor3D>::extractRegionOf(in, align);
			}
		}
		
		void reorient (const Matrix3D& mat) {
			cout << "Reorienting " << this->name << " ... " << flush;
			clock_t t1 = clock();
			_SIZE
			_ITERATE_BEGIN
				this->voxel[i][j][k].PPDTransformByEqual(mat);
			_ITERATE_END
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			this->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "reorient"));
			return;
		}
		
		ScalarVolume *getScalarIndex (const SymTensor3D::ScalarIndex type, const double faThreshold = 0.0) const {
			_SIZE
			ScalarVolume *sv = new ScalarVolume(this->size);
			double*** sVol = sv->voxel;
			sv->setVSize(this->vsize);
			sv->setOrigin(this->origin);
			string suffix = "";
			cout << "Computing the ";
			switch (type) {
					case SymTensor3D::TRACE:
						suffix = "tr";
						cout << "tensor trace (mean diffusion) ";
						break;
					case SymTensor3D::AD:
						suffix = "ad";
						cout << "axial (parallel) diffusion ";
						break;
					case SymTensor3D::RD:
						suffix = "rd";
						cout << "radial (perpendicular) diffusion ";
						break;
					case SymTensor3D::FA:
						suffix = "fa";
						cout << "fractional anisotropy ";
						break;
					case SymTensor3D::TSP:
						suffix = "tsp";
						cout << "tensor scalar product ";
						break;
					case SymTensor3D::NORM:
						suffix = "norm";
						cout << "tensor Euclidean norm ";
						break;
					case SymTensor3D::DTSP:
						suffix = "dtsp";
						cout << "deviatoric tensor scalar product ";
						break;
					case SymTensor3D::DNORM:
						suffix = "dnorm";
						cout << "deviatoric tensor Euclidean norm ";
						break;
					case SymTensor3D::DYDISP:
						suffix = "dydisp";
						cout << "dyadic dispersion ";
						break;
					case SymTensor3D::DYCOH:
						suffix = "dycoh";
						cout << "dyadic coherence ";
						break;
					case SymTensor3D::PDMP:
						suffix = "pdmp";
						cout << "PD maximum projection ";
						break;
			}
			cout << "map ... " << flush;
			clock_t t1 = clock();
			_ITERATE_BEGIN
				double tmp = 0.0;
				switch (type) {
					case SymTensor3D::TRACE:
						tmp = this->voxel[i][j][k].getTrace();
						break;
					case SymTensor3D::AD:
						tmp = this->voxel[i][j][k].getAD();
						break;
					case SymTensor3D::RD:
						tmp = this->voxel[i][j][k].getRD();
						break;
					case SymTensor3D::FA:
						tmp = this->voxel[i][j][k].getAnisotropy();
						break;
					case SymTensor3D::TSP:
						tmp = this->voxel[i][j][k].getTSP();
						break;
					case SymTensor3D::NORM:
						tmp = this->voxel[i][j][k].getNorm2();
						break;
					case SymTensor3D::DTSP:
						tmp = this->voxel[i][j][k].getDeviatoricTSP();
						break;
					case SymTensor3D::DNORM:
						tmp = this->voxel[i][j][k].getDeviatoricNorm2();
						break;
					case SymTensor3D::DYDISP:
						tmp = this->voxel[i][j][k].getDyadicDispersion();
						break;
					case SymTensor3D::DYCOH:
						tmp = this->voxel[i][j][k].getDyadicCoherence();
						break;
					case SymTensor3D::PDMP:
						tmp = this->voxel[i][j][k].getPDMaxProjection(faThreshold);
						break;
				}
				sVol[i][j][k] = tmp;
			_ITERATE_END
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			sv->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), suffix.c_str()));
			return sv;
		}
		
		ScalarVolume *getVoxelwiseSimilarityTo (const Volume<SymTensor3D>& in) const {
			cout << "Computing voxelwise similarity to " << in.getName();
			cout << "  ... " << flush;
			clock_t t1 = clock();
			_SIZE
			ScalarVolume *sv = new ScalarVolume(this->size);
			sv->setVSize(this->vsize);
			sv->setOrigin(this->origin);
			double*** sVol = sv->voxel;
			double tmp = 0.0;
			_ITERATE_BEGIN
				tmp = this->voxel[i][j][k].computeSimilarity(in.voxel[i][j][k]);
				sVol[i][j][k] = tmp;
			_ITERATE_END
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			sv->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(),"sm"));
			return sv;
		}
		
		ScalarVolume *getAngleOfPDs (const Volume<SymTensor3D>& in, const Volume<double>& mask) const {
			cout << "Computing the angle of PD map to ";
			cout << in.getName() << " in the mask " << mask.getName();
			cout << " ... " << flush;
			clock_t t1 = clock();
			_SIZE
			ScalarVolume *sv = new ScalarVolume(this->size);
			sv->setVSize(this->vsize);
			sv->setOrigin(this->origin);
			double*** sVol = sv->voxel;
			double tmp = 0.0;
			_ITERATE_BEGIN
				if (mask.voxel[i][j][k] == 1) {
					tmp = this->voxel[i][j][k].getAngleOfPDs(in.voxel[i][j][k]);
					sVol[i][j][k] = tmp;
				} else {
					sVol[i][j][k] = 0.0;
				}
			_ITERATE_END
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			sv->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "aop"));
			return sv;
		}
		
		void getEigenSystem (ScalarVolume *eigsSV[3], Vector3DVolume *eigvVV[3]) const {
			_SIZE
			for (int m = 0; m < 3; ++m) {
				eigsSV[m] = new ScalarVolume(this->size);
				eigsSV[m]->setVSize(this->vsize);
				eigsSV[m]->setOrigin(this->origin);
				eigvVV[m] = new Vector3DVolume(this->size);
				eigvVV[m]->setVSize(this->vsize);
				eigvVV[m]->setOrigin(this->origin);
			}
			
			_ITERATE_BEGIN
				double eigs[3];
				Vector3D eigv[3];
				this->voxel[i][j][k].getEigenSystem(eigs, eigv);
				for (int m = 0; m < 3; ++m) {
					eigsSV[m]->voxel[i][j][k] = eigs[m];
					eigvVV[m]->voxel[i][j][k] = eigv[m];
				}
			_ITERATE_END
		}
		
		void getEigenvalues (ScalarVolume *eigsSV[3]) const {
			cout << "Computing the tensor eigenvalue maps ... " << flush;
			clock_t t1 = clock();
			_SIZE
			for (int m = 0; m < 3; ++m) {
				eigsSV[m] = new ScalarVolume(this->size);
				eigsSV[m]->setVSize(this->vsize);
				eigsSV[m]->setOrigin(this->origin);
			}

			_ITERATE_BEGIN
				double eigs[3];
				this->voxel[i][j][k].getEigenvalues(eigs);
				for (int m = 0; m < 3; ++m) {
					eigsSV[m]->voxel[i][j][k] = eigs[m];
				}
			_ITERATE_END
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			eigsSV[0]->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "lambda1"));
			eigsSV[1]->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "lambda2"));
			eigsSV[2]->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "lambda3"));
		}
		
		void getLPSMeasures (ScalarVolume *lps[3]) const {
			cout << "Computing the tensor shape index maps ... " << flush;
			clock_t t1 = clock();
			_SIZE
			for (int m = 0; m < 3; ++m) {
				lps[m] = new ScalarVolume(this->size);
				lps[m]->setVSize(this->vsize);
				lps[m]->setOrigin(this->origin);
			}
			
			double tmp[3];
			_ITERATE_BEGIN
				this->voxel[i][j][k].getLPSMeasures(tmp);
				for (int m = 0; m < 3; ++m) {
					lps[m]->voxel[i][j][k] = tmp[m];
				}
			_ITERATE_END
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			lps[0]->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "linear"));
			lps[1]->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "planar"));
			lps[2]->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "spherical"));
		}
		
		Vector3DVolume *getPD (const bool useFA) const {
			if (useFA) {
				cout << "Computing the FA-weighted principal diffusion direction map ... " << flush;
			} else {
				cout << "Computing the principal diffusion direction map ... " << flush;
			}
			clock_t t1 = clock();
			_SIZE
			Vector3DVolume *vv = new Vector3DVolume(this->size);
			vv->setOrigin(this->origin);
			vv->setVSize(this->vsize);
			Vector3D*** vVol = vv->voxel;
			_ITERATE_BEGIN
				this->voxel[i][j][k].getPD(vVol[i][j][k], useFA);
			_ITERATE_END
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			vv->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "pd"));
			return vv;
		}
		
		Vector3DVolume *getPDColorCode (const bool useFA, const double scale = 1.0) const {
			if (useFA) {
				cout << "Computing the FA-weighted RGB map ... " << flush;
			} else {
				cout << "Computing the RGB map ... " << flush;
			}
			clock_t t1 = clock();
			_SIZE
			Vector3DVolume *vv = new Vector3DVolume(this->size);
			vv->setOrigin(this->origin);
			vv->setVSize(this->vsize);
			Vector3D*** vVol = vv->voxel;
			_ITERATE_BEGIN
				this->voxel[i][j][k].getPDColorCode(vVol[i][j][k], useFA, scale);
			_ITERATE_END
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			vv->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "rgb"));
			return vv;
		}
		
		double getAveragedEEPairsOverlap (const Volume<SymTensor3D>& in, const double faThreshold) const {
			_SIZE
			double sum = 0.0;
			double counter = 0;
			double fa;
			_ITERATE_BEGIN
				fa = this->voxel[i][j][k].getAnisotropy();
				if (fa >= faThreshold) {
					sum += this->voxel[i][j][k].getEEPairsOverlap(in.voxel[i][j][k]);
					counter += 1.0;
				}
			_ITERATE_END
			// averaged over all voxels
			if (counter != 0.0) { 
				sum /= counter;
			} else {
				sum = 0.0;
			}
			
			return sum;
		}
		
		double getAveragedEEPairsOverlap (const Volume<SymTensor3D>& in, const Volume<double>& mask, const double faThreshold) const {
			_SIZE
			double sum = 0.0;
			double counter = 0;
			double fa;
			_ITERATE_BEGIN
				fa = this->voxel[i][j][k].getAnisotropy();
				if (fa >= faThreshold && mask.voxel[i][j][k] == 1.0) {
					sum += this->voxel[i][j][k].getEEPairsOverlap(in.voxel[i][j][k]);
					counter += 1.0;
				}
			_ITERATE_END
			// averaged over all voxels
			if (counter != 0.0) { 
				sum /= counter;
			} else {
				sum = 0.0;
			}
			
			return sum;
		}
		
		void getFAWeightedAverageOfAngleOfPDs (const Volume<SymTensor3D>& in, const double faThreshold, double stats[2]) const {
			_SIZE
			double weighted_sum = 0.0;
			double weighted_sumSquare = 0.0;
			double sum_of_weights = 0.0;
			double sum_of_weightsSquare = 0.0;
			_ITERATE_BEGIN
				const double fa1 = this->voxel[i][j][k].getAnisotropy();
				if (fa1 >= faThreshold) {
					const double fa2 = in.voxel[i][j][k].getAnisotropy();
					const double weightSquare = fa1 * fa2;
					//const double weightSquare = fa1 * fa1;
					sum_of_weightsSquare += weightSquare;
					const double weight = sqrt(weightSquare);
					sum_of_weights += weight;
					const double aop = this->voxel[i][j][k].getAngleOfPDs(in.voxel[i][j][k]);
					const double tmp = weight * aop;
					weighted_sum += tmp;
					weighted_sumSquare += tmp * aop;
				}
			_ITERATE_END
			
			if (sum_of_weights >= 1.0e-6) {
				// weighted average
				stats[0] = weighted_sum / sum_of_weights;
				// weighted standard deviation
				double multiplier = sum_of_weights;
				multiplier /= sum_of_weights * sum_of_weights - sum_of_weightsSquare;
				stats[1] = weighted_sumSquare - weighted_sum * weighted_sum / sum_of_weights;
				stats[1] *= multiplier;
			} else {
				cerr << "sum of weights is 0.0" << endl;
				exit (1);
			}
		}
		
		void getFAWeightedAverageOfAngleOfPDs (const Volume<SymTensor3D>& in, const Volume<double>& mask, const double faThreshold, double stats[2]) const {
			_SIZE
			double weighted_sum = 0.0;
			double weighted_sumSquare = 0.0;
			double sum_of_weights = 0.0;
			double sum_of_weightsSquare = 0.0;
			_ITERATE_BEGIN
				if (mask.voxel[i][j][k] == 0.0) {
					continue;
				}
				const double fa1 = this->voxel[i][j][k].getAnisotropy();
				if (fa1 >= faThreshold) {
					const double fa2 = in.voxel[i][j][k].getAnisotropy();
					const double weightSquare = fa1 * fa2;
					//const double weightSquare = fa1 * fa1;
					sum_of_weightsSquare += weightSquare;
					const double weight = sqrt(weightSquare);
					sum_of_weights += weight;
					const double aop = this->voxel[i][j][k].getAngleOfPDs(in.voxel[i][j][k]);
					const double tmp = weight * aop;
					weighted_sum += tmp;
					weighted_sumSquare += tmp * aop;
				}
			_ITERATE_END
			
			if (sum_of_weights >= 1.0e-6) {
				// weighted average
				stats[0] = weighted_sum / sum_of_weights;
				// weighted standard deviation
				double multiplier = sum_of_weights;
				multiplier /= sum_of_weights * sum_of_weights - sum_of_weightsSquare;
				stats[1] = weighted_sumSquare - weighted_sum * weighted_sum / sum_of_weights;
				stats[1] *= multiplier;
				stats[1] = sqrt(stats[1]);
			} else {
				cerr << "sum of weights is 0.0" << endl;
				exit (1);
			}
		}
		
		Vector3D computeCenterOfMass () const {
			_SIZE
			// compute center of mass
			Vector3D com;
			double sum = 0.0;
			_ITERATE_BEGIN
				const double mass = this->voxel[i][j][k].getNorm2();
				sum += mass;
				Vector3D vec(i, j, k);
				this->toAbs(vec);
				com += mass * vec;
			_ITERATE_END
			com *= 1.0/sum;
			return com;
		}
		
		SymTensor3D computeMomentOfInertia () const {
			_SIZE
			Vector3D com = computeCenterOfMass();
			SymTensor3D moi;
			double sum = 0.0;
			_ITERATE_BEGIN
				const double mass = this->voxel[i][j][k].getNorm2();
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
			moi *= 1.0/sum;
			return moi;
		}
		
		void flip (const bool flipx, const bool flipy, const bool flipz) {
			_SIZE
			if (flipx) {
				_ITERATE_BEGIN
					if (i < xsize/2) {
						SymTensor3D tmp = this->voxel[i][j][k].flip(0);
						this->voxel[i][j][k] = this->voxel[xsize - 1 - i][j][k].flip(0);
						this->voxel[xsize - 1 - i][j][k] = tmp;
					}
				_ITERATE_END
			}
			if (flipy) {
				_ITERATE_BEGIN
					if (j < ysize/2) {
						SymTensor3D tmp = this->voxel[i][j][k].flip(1);
						this->voxel[i][j][k] = this->voxel[i][ysize - 1 - j][k].flip(1);
						this->voxel[i][ysize - 1 - j][k] = tmp;
					}
				_ITERATE_END
			}
			if (flipz) {
				_ITERATE_BEGIN
					if (k < zsize/2) {
						SymTensor3D tmp = this->voxel[i][j][k].flip(2);
						this->voxel[i][j][k] = this->voxel[i][j][zsize - 1 - k].flip(2);
						this->voxel[i][j][zsize - 1 - k] = tmp;
					}
				_ITERATE_END
			}
		}
	};
}

#endif

