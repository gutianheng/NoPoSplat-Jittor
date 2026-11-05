// Jittor bridge, deliberately independent of libtorch/ATen/Python C API.
// Rasterizer kernels retain their original non-commercial research license.
#include "bridge.h"
#include "cuda_rasterizer/rasterizer.h"
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
thread_local std::string last_error;
void check(cudaError_t error) { if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
struct Buffer {
    char* p=nullptr;
    ~Buffer(){if(p)cudaFree(p);}
    char* resize(size_t n){if(p){check(cudaFree(p));p=nullptr;}if(n)check(cudaMalloc((void**)&p,n));return p;}
};
struct State {Buffer geom,bin,img,alpha,touched;int rendered=0;};
std::unordered_map<std::int64_t,std::unique_ptr<State>> states;
std::mutex gate;
std::int64_t sequence=1;
float* zeros(Buffer& b,size_t count){float* p=(float*)b.resize(count*sizeof(float));if(count)check(cudaMemset(p,0,count*sizeof(float)));return p;}
}

extern "C" int nps_abi_version(){return 1;}
extern "C" const char* nps_last_error(){return last_error.c_str();}
extern "C" int nps_forward(int P,int M,int D,int H,int W,float tx,float ty,
 const float* means,const float* cov,const float* sh,const float* opacity,
 const float* view,const float* proj,const float* cam,const float* bg,
 float* color,float* depth,int* radii,std::int64_t* handle){
 try {
    std::lock_guard<std::mutex> lock(gate);
    check(cudaDeviceSynchronize());
    auto state=std::make_unique<State>();
    auto& s=*state;
    check(cudaMemset(color,0,3*H*W*sizeof(float)));
    check(cudaMemset(depth,0,H*W*sizeof(float)));
    check(cudaMemset(radii,0,P*sizeof(int)));
    float* alpha=zeros(s.alpha,H*W);
    int* touched=(int*)s.touched.resize(P*sizeof(int));
    check(cudaMemset(touched,0,P*sizeof(int)));
    s.rendered=CudaRasterizer::Rasterizer::forward(
      [&](size_t n){return s.geom.resize(n);},[&](size_t n){return s.bin.resize(n);},[&](size_t n){return s.img.resize(n);},
      P,D,M,bg,W,H,means,sh,nullptr,opacity,nullptr,1.f,nullptr,cov,
      view,proj,cam,tx,ty,false,color,depth,alpha,radii,touched,false);
    check(cudaDeviceSynchronize());
    std::int64_t id=sequence++;
    states.emplace(id,std::move(state));
    check(cudaMemcpy(handle,&id,sizeof(id),cudaMemcpyHostToDevice));
    return 0;
 } catch(const std::exception& e){last_error=e.what();return 1;}
}

extern "C" int nps_backward(std::int64_t handle,int P,int M,int D,int H,int W,float tx,float ty,
 const float* means,const float* cov,const float* sh,
 const float* view,const float* proj,const float* rawproj,const float* cam,const float* bg,
 const int* radii,const float* gc,const float* gd,
 float* gm,float* gcv,float* gsh,float* go,float* gtau){
 try {
    std::lock_guard<std::mutex> lock(gate);
    check(cudaDeviceSynchronize());
    auto it=states.find(handle);
    if(it==states.end())throw std::runtime_error("Raster state already released; release only after backward completes");
    auto& s=*it->second;
    Buffer m2,conic,col,dep,scale,rotation;
    for(auto item:std::vector<std::pair<float*,size_t>>{{gm,(size_t)P*3},{gcv,(size_t)P*6},{gsh,(size_t)P*M*3},{go,(size_t)P},{gtau,(size_t)P*6}})
      check(cudaMemset(item.first,0,item.second*sizeof(float)));
    CudaRasterizer::Rasterizer::backward(P,D,M,s.rendered,bg,W,H,means,sh,nullptr,nullptr,1.f,nullptr,cov,
      view,proj,rawproj,cam,tx,ty,radii,s.geom.p,s.bin.p,s.img.p,gc,gd,
      zeros(m2,P*3),zeros(conic,P*4),go,zeros(col,P*3),zeros(dep,P),gm,gcv,gsh,zeros(scale,P*3),zeros(rotation,P*4),gtau,false);
    check(cudaDeviceSynchronize());
    return 0;
 } catch(const std::exception& e){last_error=e.what();return 1;}
}

extern "C" void nps_release_all(){
 std::lock_guard<std::mutex> lock(gate);
 cudaDeviceSynchronize();
 states.clear();
}

