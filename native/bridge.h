#pragma once
#include <cstdint>
extern "C" {
int nps_abi_version();
const char* nps_last_error();
int nps_forward(int P,int M,int D,int H,int W,float tx,float ty,
 const float* means,const float* cov,const float* sh,const float* opacity,
 const float* view,const float* proj,const float* cam,const float* bg,
 float* color,float* depth,int* radii,std::int64_t* handle);
int nps_backward(std::int64_t handle,int P,int M,int D,int H,int W,float tx,float ty,
 const float* means,const float* cov,const float* sh,
 const float* view,const float* proj,const float* rawproj,const float* cam,const float* bg,
 const int* radii,const float* gc,const float* gd,
 float* gm,float* gcv,float* gsh,float* go,float* gtau);
void nps_release_all();
}

