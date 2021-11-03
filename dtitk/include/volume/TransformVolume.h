/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: TransformVolume.h,v $
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


#ifndef _volume_TransformVolume_H
#define _volume_TransformVolume_H

#include "Volume.h"

namespace volume {
	// macros for iteration
	#define _SIZE const int xsize = this->size[0]; const int ysize = this->size[1]; const int zsize = this->size[2];
	#define _ITERATE_BEGIN for (int i = 0; i < xsize; ++i) { for (int j = 0; j < ysize; ++j) { for (int k = 0; k < zsize; ++k) {
	#define _ITERATE_END } } }
	
	// current:
	// the subject object
	// 
	// other:
	// the template object
	#define _COMMON_OBJ Object current; Object other;
			
	//declaration
	template <class Object, class Transform>
	class TransformVolume : public Volume<Object> {
		
		protected:
		// in a typical registration situation, although our goal
		// is to align the subject to the template, we actually
		// need to compute the transformation that warps the template
		// to the subject.
		//
		// this transformation is the INVERSE to the one that warps
		// the subject to the template.
		// 
		// this inverse transformation allows us to create the
		// registered subject in the template image space.
		// 
		// so the transformation (trans) defined here refers to
		// this INVERSE transformation!!!
		// 
		// the transformation in general is a nonlinear function.
		// the corresponding jacobian is therefore a function of
		// position in space.
		// 
		// in the case of rigid and affine transformations, the
		// transformation is a linear function and the jacobian is
		// a constant matrix.  this allows us to represent the rigid
		// transformation by the jacobian matrix (Rigid3D),
		// while represent the affine transformation by the jacobian
		// matrix and the translation (Affine3D).
		// 
		// more generally, the transformation should be represented
		// in its complete functional form (analytically or numerically),
		// accompanied by its jacobian
		Transform trans;
		
		// center of the volume
		// in the absolute scale
		Vector3D center;
		
		protected:
		TransformVolume (bool enableGrad = false)
			: Volume<Object> (enableGrad) {}
		
		// default implementation here does nothing
		// 
		// for objects such as vectors or tensors, subclass
		// can extend this function to take care of issues
		// such as reorientation
		// 
		// this version is for reorienting an object from
		// the untransformed (not interpolated) volume
		virtual void objectSpecificTransform (Object& object) const {};
		
		// default implementation here does nothing
		// 
		// for objects such as vectors or tensors, subclass
		// can extend this function to take care of issues
		// such as reorientation
		// 
		// this version is for reorienting an object from the
		// transformed (interpolated) volume, and is required
		// for computing a transformed volume
		virtual void objectSpecificTransformInverse (Object& object) const {};
		
		// component wise similarity metrics
		// leave to the subclasses
		virtual double computeComponentSimilarity (
			const Object& o1, const Object& o2) const = 0;
		virtual double computeComponentSimilarityGradient (
			const Object& t, const Object& s, Object *g, const Vector3D& v,
			double *xi) const = 0;
		
		public:
		TransformVolume (const int sz[3], bool enableGrad = false)
			: Volume<Object> (sz, enableGrad) {}
		
		TransformVolume (const VoxelSpace& vs, bool enableGrad = false)
			: Volume<Object> (vs, enableGrad) {}
		
		virtual ~TransformVolume () {};
		
		// set the center of the transformation
		// as the center of the volume
		void setCenter () {
			// macros
			_SIZE
			center.set(0.5 * (xsize - 1), 0.5 * (ysize - 1), 0.5 * (zsize - 1));
			this->toAbs(center);
		}
		
		// set the center of the transformation
		// as the nifti origin of the volume
		void setCenterToNiftiOrigin () {
			// macros
			center.set(this->origin[0], this->origin[1], this->origin[2]);
			this->toAbs(center);
		}
		
		// set the center of the transformation
		// as the center of the region
		void setCenterToRegionCenter () {
			center.set(this->regionCenterRel[0], this->regionCenterRel[1], this->regionCenterRel[2]);
			this->toAbs(center);
		}
		
		void setCenter (double x, double y, double z) {
			center.set(x, y, z);
		}
		
		// get the center
		void getCenter (Vector3D& _center) const {
			_center = center;
		}
		
		// compute the displacement between center-of-masses
		Vector3D computeDisplacementOfCenterOfMass (const Volume<Object>& other) const {
			Vector3D disp = other.computeCenterOfMass();
			disp -= this->computeCenterOfMass();
			return disp;
		}
		
		// set the transformation
		void setTransformation (const Transform& in) {
			trans = in;
		}
		
		// get the transformation
		// relative to the origin
		void getTransformation (Transform& out) const {
			out = trans;
		}
		
		void computeTransformGeneric (Volume<Object>& out, const bool backward, const int intp = 0) {
			if (backward) {
				computeTransformBackward(out, intp);
			} else {
				computeTransformForward(out, intp);
			}
		}
		
		void computeTransform (Volume<Object>& out, const int intp = 0) {
			computeTransformBackward(out, intp);
		}
		
		// compute the transformed object projected onto the grid
		// defined by input volume: out
		// 
		// the input transformation is assumed to be the inverse
		// transformation
		// 
		// make sure the reorientation matrix is the inverse of the
		// jacobian of the inverse transformation
		void computeTransformBackward (Volume<Object>& out, const int intp = 0) const {
			int xsize = out.getXSize();
			int ysize = out.getYSize();
			int zsize = out.getZSize();
			
			cout << "backward resampling ..." << flush;
			clock_t t1 = clock();
			
			Vector3D vec;
			_ITERATE_BEGIN
				// vector at (i,j,k) of volume out
				vec[0] = i;
				vec[1] = j;
				vec[2] = k;
				out.toAbs(vec);
				
				// trans is the INVERSE transformation
				vec *= trans;
				
				if (this->getVoxelAt(vec, out.voxel[i][j][k], intp)) {
					// if within range
					// reorientation, for tensor objects, for instance
					objectSpecificTransformInverse(out.voxel[i][j][k]);
				}
			_ITERATE_END
			
			clock_t t2 = clock();
			cout << "time consumed = " << (t2 - t1)/(double)CLOCKS_PER_SEC << endl;
		}
		
		// compute the FORWARD transformed object projected onto the grid
		// defined by input volume: out
		// 
		// the input transformation is assumed to be the inverse
		// transformation
		// 
		// make sure the reorientation matrix is the
		// jacobian of the inverse transformation
		void computeTransformForward (Volume<Object>& out, const int intp = 0) const {
			const int xsize = this->size[0];
			const int ysize = this->size[1];
			const int zsize = this->size[2];
			
			cout << "forward resampling ..." << flush;
			clock_t t1 = clock();
			
			// a volume to store the normalization coefficients
			int out_size[3];
			double out_vsize[3];
			double out_origin[3];
			out.getSize(out_size);
			out.getVSize(out_vsize);
			out.getOrigin(out_origin);
			Volume<double> normalize(out_size);
			normalize.setVSize(out_vsize);
			normalize.setOrigin(out_origin);
			normalize.setBackground(0.0);
			normalize.fillWithBackground();
			
			// remember to set the background
			out.fillWithBackground();
			
			Vector3D vec;
			_ITERATE_BEGIN
				// vector at (i,j,k) of *this volume
				vec[0] = i;
				vec[1] = j;
				vec[2] = k;
				this->toAbs(vec);
				
				// trans is the INVERSE transformation
				vec *= trans;
				
				// reorientation, for tensor objects, for instance
				// use a copy to avoid modifying the original
				Object current = this->voxel[i][j][k];
				objectSpecificTransform(current);
				out.putVoxelAt(vec, current, normalize);
			_ITERATE_END
			
			if (intp == 0) {
				// apply the coefficient normalization
				out.divide(normalize, 0.0);
			}
			
			clock_t t2 = clock();
			cout << "time consumed = " << (t2 - t1)/(double)CLOCKS_PER_SEC << endl;
			if (intp != 0) {
				// save the coefficient normalization if not applied
				normalize.writeScalarNifti("normalize.nii.gz");
			}
		}
		
		double computeSimilarityRegion () {
			double sum = 0.0;
			Object current;
			
			for (int i = this->regionOriginRel[0]; i < this->regionEndRel[0]; i += this->step[0]) {
				for (int j = this->regionOriginRel[1]; j < this->regionEndRel[1]; j += this->step[1]) {
					for (int k = this->regionOriginRel[2]; k < this->regionEndRel[2]; k += this->step[2]) {
						// the subject object at (i,j,k)
						current = this->voxel[i][j][k];
						sum += computeComponentSimilarity(current, this->zero);
					}
				}
			}
			
			const double factor = this->step[0] * this->step[1] * this->step[2] * this->vsize[0] * this->vsize[1] * this->vsize[2];
			return sum * factor;
		}
		
		// similarity computed with inverse transformation
		// 
		// transform the template image instead
		// 
		// "this" is the template
		// "vol" is the subject
		// 
		// for region
		double computeSimilarityRegion (const Volume<Object>& vol, const Volume<double> *mask,
			const int intp) {
			// macros
			_COMMON_OBJ
			
			Vector3D vec;
			double sum = 0.0;
			
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
						
						// the vector at (i,j,k)
						vec[0] = i;
						vec[1] = j;
						vec[2] = k;
						this->toAbs(vec);
						
						// inverse transform the vector
						// (the input is the inverse transformation)
						vec *= trans;
						
						if (vol.getVoxelAt(vec, other, intp)) {
							// if within range
							// object specific transformation of the template object
							objectSpecificTransform(current);
						}
						sum += computeComponentSimilarity(current, other);
					}
				}
			}
			
			const double factor = this->step[0] * this->step[1] * this->step[2] * this->vsize[0] * this->vsize[1] * this->vsize[2];
			return sum * factor;
		}
		
		// backward compatibility
		double computeSimilarityRegion (const Volume<Object>& vol, const int intp) {
			return computeSimilarityRegion(vol, 0, intp);
		}
		
		// for command line output
		double computeSimilarityRegionWithInfo (const Volume<Object>& vol, const Volume<double> *mask, int intp) {
			cout << "Computing the image similarity between ";
			cout << this->name << " and " << vol.getName() << " ... " << flush;
			clock_t t1 = clock();
			const double similarity = computeSimilarityRegion(vol, mask, intp);
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			cout << "Similarity = " << similarity << endl;
			return similarity;
		}
		
		// for region
		// gradient
		double computeSimilarityGradientRegion (const Volume<Object>& vol, const Volume<double> *mask,
			double *xi, int xiDim, int intp) {
			// make sure the subject's gradient volumes are computed
			if (!vol.getGradEnabled()) {
				cerr << "The subject volume has gradient feature disabled" << endl;
				exit(1);
			}
			
			switch (intp) {
				default:
				case 0: break;
			}
			
			// macros
			_COMMON_OBJ
			
			// the interploated gradient objects at the new coordinate
			Object gradOther[3];
			
			Vector3D vec;
			
			// accumulate the value of the similarity integral
			double sum = 0.0;
			
			// accumulate the values of the similarity gradients integral
			for (int i = 0; i < xiDim; ++i) {
				xi[i] = 0.0;
			}
			
			for (int i = this->regionOriginRel[0]; i < this->regionEndRel[0]; i += this->step[0]) {
				for (int j = this->regionOriginRel[1]; j < this->regionEndRel[1]; j += this->step[1]) {
					for (int k = this->regionOriginRel[2]; k < this->regionEndRel[2]; k += this->step[2]) {
						// skip zero entries in mask volume
						if (mask != 0) {
							if ((int)(mask->voxel[i][j][k]) == 0) {
								continue;
							}
						}
						
						// the template object
						current = this->voxel[i][j][k];
						
						// the vector at (i,j,k)
						vec[0] = i;
						vec[1] = j;
						vec[2] = k;
						this->toAbs(vec);
						
						// inverse transform the vector
						// (the input is the inverse transformation)
						vec *= trans;
						
						if (vol.getVoxelAt(vec, other, gradOther, intp)) {
							// if within range
							// reset the vec back to untransformed version
							// in the grid scale
							vec[0] = i;
							vec[1] = j;
							vec[2] = k;
							
							// the similarity and the gradients
							// are computed together in the following
							// virtual function
							// 
							// note that voxel[i][j][k], the template
							// object is not transformed.
							// the appropriate transformation is done
							// in the called funtion
							sum += computeComponentSimilarityGradient(
									current, other, gradOther, vec, xi);
						} else {
							// the out of bound subject object assumed to be
							// background
							sum += computeComponentSimilarity(current, other);
							
							// do nothing for the similarity gradient integrals
						}
					}
				}
			}
			
			const double factor = this->step[0] * this->step[1] * this->step[2] * this->vsize[0] * this->vsize[1] * this->vsize[2];
			for (int i = 0; i < xiDim; ++i) {
				xi[i] *= factor;
			}
			return sum * factor;
		}
		
		void printValues (const Volume<double>& mask) const {
			cout << "Print the voxel values in the mask ";
			cout << mask.getName() << " ... " << endl << flush;
			_SIZE
			_ITERATE_BEGIN
				if ((int)(mask.voxel[i][j][k]) != 0.0) {
					cout << "At Voxel [";
					cout << i << ", ";
					cout << j << ", ";
					cout << k << "]" << endl;
					cout << this->voxel[i][j][k] << endl;
				}
			_ITERATE_END
		}
		
		// backward compatibility
		double computeSimilarityGradientRegion (const Volume<Object>& vol,
			double *xi, int xiDim, int intp) {
			return computeSimilarityGradientRegion(vol, 0, xi, xiDim, intp);
		}
		
	}; // end of the class
	
}

#endif

