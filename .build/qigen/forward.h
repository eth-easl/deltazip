#include<omp.h>
#include<immintrin.h>
#include<fstream>

#define mymin(a,b) ((a)<(b)?(a):(b))
#define mymax(a,b) ((a)>(b)?(a):(b))
inline
void q4gemm(const float* __restrict__ input, 
const int* __restrict__ W, 
const float* __restrict__ scales, 
const float* __restrict__ zeros, 
const float* __restrict__ bias, 
 const float* __restrict__ sums, 
 float* __restrict__ output,
const int n,
const int m,
const int t,
const int nb,
const int mb,
const int tb,
int ogtt,
const int cutoff){
#pragma omp parallel num_threads(16)
{
int tid;
const int mu = 16;
const int nu = 1;
const int tu = 16;
const int on = n / nb;
const int om = m / mb;
const __m256i mask = _mm256_set1_epi32(15);
tid = omp_get_thread_num();
int tt = ogtt;
if(tid >= cutoff){
tt -= tb;
}
const int base_output = tid >= cutoff ?
 (tid-cutoff)*tt + (tt+tb)*cutoff: 
 tid*tt;
const int base_W = tid >= cutoff ?
 ((tid-cutoff)*tt + (tt+tb)*cutoff)*m/8: 
 tid*tt*m/8;
for(int j = 0; j < tt; j+=tb){
for(int i = 0; i < on; i++) {
for(int k = 0; k < om; k++) {
for(int i1 = 0; i1 < nb; i1+=nu) {
int j1 = 0;
for(; j1 < tb-tu+1; j1+=tu) {
__m256 acc0_0 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+0]);
__m256 acc0_8 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+8]);
for(int k1 = 0; k1 < mb; k1+=mu) {
for(int k2 = k1; k2 < k1+mu; k2+=8){
__m256i w0 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/8 + k*mb*tb/8 + k2*tb/8 + j1+0]);
__m256i w8 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/8 + k*mb*tb/8 + k2*tb/8 + j1+8]);
__m256 v0_7 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+7)*nb + i1+0]);
__m256i ws0_7 = _mm256_srli_epi32(w0, 28);
__m256i ws8_7 = _mm256_srli_epi32(w8, 28);
__m256i wsa0_7= _mm256_and_si256(ws0_7, mask);
__m256i wsa8_7= _mm256_and_si256(ws8_7, mask);
__m256 l0_7 = _mm256_cvtepi32_ps(wsa0_7);
__m256 l8_7 = _mm256_cvtepi32_ps(wsa8_7);
acc0_0 = _mm256_fmadd_ps(v0_7, l0_7, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_7, l8_7, acc0_8);
__m256 v0_6 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+6)*nb + i1+0]);
__m256i ws0_6 = _mm256_srli_epi32(w0, 24);
__m256i ws8_6 = _mm256_srli_epi32(w8, 24);
__m256i wsa0_6= _mm256_and_si256(ws0_6, mask);
__m256i wsa8_6= _mm256_and_si256(ws8_6, mask);
__m256 l0_6 = _mm256_cvtepi32_ps(wsa0_6);
__m256 l8_6 = _mm256_cvtepi32_ps(wsa8_6);
acc0_0 = _mm256_fmadd_ps(v0_6, l0_6, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_6, l8_6, acc0_8);
__m256 v0_5 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+5)*nb + i1+0]);
__m256i ws0_5 = _mm256_srli_epi32(w0, 20);
__m256i ws8_5 = _mm256_srli_epi32(w8, 20);
__m256i wsa0_5= _mm256_and_si256(ws0_5, mask);
__m256i wsa8_5= _mm256_and_si256(ws8_5, mask);
__m256 l0_5 = _mm256_cvtepi32_ps(wsa0_5);
__m256 l8_5 = _mm256_cvtepi32_ps(wsa8_5);
acc0_0 = _mm256_fmadd_ps(v0_5, l0_5, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_5, l8_5, acc0_8);
__m256 v0_4 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+4)*nb + i1+0]);
__m256i ws0_4 = _mm256_srli_epi32(w0, 16);
__m256i ws8_4 = _mm256_srli_epi32(w8, 16);
__m256i wsa0_4= _mm256_and_si256(ws0_4, mask);
__m256i wsa8_4= _mm256_and_si256(ws8_4, mask);
__m256 l0_4 = _mm256_cvtepi32_ps(wsa0_4);
__m256 l8_4 = _mm256_cvtepi32_ps(wsa8_4);
acc0_0 = _mm256_fmadd_ps(v0_4, l0_4, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_4, l8_4, acc0_8);
__m256 v0_3 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+3)*nb + i1+0]);
__m256i ws0_3 = _mm256_srli_epi32(w0, 12);
__m256i ws8_3 = _mm256_srli_epi32(w8, 12);
__m256i wsa0_3= _mm256_and_si256(ws0_3, mask);
__m256i wsa8_3= _mm256_and_si256(ws8_3, mask);
__m256 l0_3 = _mm256_cvtepi32_ps(wsa0_3);
__m256 l8_3 = _mm256_cvtepi32_ps(wsa8_3);
acc0_0 = _mm256_fmadd_ps(v0_3, l0_3, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_3, l8_3, acc0_8);
__m256 v0_2 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+2)*nb + i1+0]);
__m256i ws0_2 = _mm256_srli_epi32(w0, 8);
__m256i ws8_2 = _mm256_srli_epi32(w8, 8);
__m256i wsa0_2= _mm256_and_si256(ws0_2, mask);
__m256i wsa8_2= _mm256_and_si256(ws8_2, mask);
__m256 l0_2 = _mm256_cvtepi32_ps(wsa0_2);
__m256 l8_2 = _mm256_cvtepi32_ps(wsa8_2);
acc0_0 = _mm256_fmadd_ps(v0_2, l0_2, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_2, l8_2, acc0_8);
__m256 v0_1 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+1)*nb + i1+0]);
__m256i ws0_1 = _mm256_srli_epi32(w0, 4);
__m256i ws8_1 = _mm256_srli_epi32(w8, 4);
__m256i wsa0_1= _mm256_and_si256(ws0_1, mask);
__m256i wsa8_1= _mm256_and_si256(ws8_1, mask);
__m256 l0_1 = _mm256_cvtepi32_ps(wsa0_1);
__m256 l8_1 = _mm256_cvtepi32_ps(wsa8_1);
acc0_0 = _mm256_fmadd_ps(v0_1, l0_1, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_1, l8_1, acc0_8);
__m256 v0_0 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+0)*nb + i1+0]);
__m256i ws0_0 = _mm256_srli_epi32(w0, 0);
__m256i ws8_0 = _mm256_srli_epi32(w8, 0);
__m256i wsa0_0= _mm256_and_si256(ws0_0, mask);
__m256i wsa8_0= _mm256_and_si256(ws8_0, mask);
__m256 l0_0 = _mm256_cvtepi32_ps(wsa0_0);
__m256 l8_0 = _mm256_cvtepi32_ps(wsa8_0);
acc0_0 = _mm256_fmadd_ps(v0_0, l0_0, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_0, l8_0, acc0_8);
}
}
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+0], acc0_0);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+8], acc0_8);
}
}
}
}
}
#pragma omp barrier
for (int i = 0; i < n; i++) {
__m256 r = _mm256_set1_ps(sums[i]);
for (int j = 0; j < tt; j+=16){
__m256 o0 = _mm256_loadu_ps(&output[i*t + base_output + j + 0]);
__m256 o8 = _mm256_loadu_ps(&output[i*t + base_output + j + 8]);
__m256 z0 = _mm256_loadu_ps(&zeros[base_output + j + 0]);
__m256 z8 = _mm256_loadu_ps(&zeros[base_output + j + 8]);
__m256 b0 = _mm256_loadu_ps(&bias[base_output + j + 0]);
__m256 b8 = _mm256_loadu_ps(&bias[base_output + j + 8]);
__m256 s0 = _mm256_loadu_ps(&scales[base_output + j + 0]);
__m256 s8 = _mm256_loadu_ps(&scales[base_output + j + 8]);
__m256 zr0 = _mm256_fmadd_ps(z0, r, o0);
__m256 zr8 = _mm256_fmadd_ps(z8, r, o8);
__m256 o20 = _mm256_fmadd_ps(zr0, s0, b0);
__m256 o28 = _mm256_fmadd_ps(zr8, s8, b8);
_mm256_storeu_ps(&output[i*t + base_output + j + 0], o20);
_mm256_storeu_ps(&output[i*t + base_output + j + 8], o28);
}
}
}
}
inline void qforward(const float* __restrict__ input, 
 const int* __restrict__ W, 
const float* __restrict__ scales, 
const float* __restrict__ zeros, 
const float* __restrict__ bias, 
const float* __restrict__ sums, 
float* __restrict__ output, 
int n, 
 int m, 
 int t) {
q4gemm(input, W, scales, zeros, bias, sums, output, n, m, t, 1, 1024, 32, 256, 17);
}
inline void pack_input(float* A, float* B){
  // copy the full matrix A in blocked format into B
  uint64_t idx = 0;
  const int N = 1;
  const int M = 4096;
  const int nb = 1;
  const int mb = 1024;
  for(int i = 0; i < N; i+=nb){ 
             for(int j = 0; j < M; j+=mb){
                 for(int jj = j; jj < mymin(j+mb, M); jj++){
                     for(int ii = i; ii < mymin(i+nb, N); ii++){
                         B[idx] = A[ii*M+jj];
                         idx++;
                     }
                 }
             }
         }
     }
inline void pack_qw_inner(int* A, int* B, int cutoff){
  // copy the full matrix A in blocked format into B
  uint64_t idx = 0;
  const int N = 512;
  const int M = 4096;
  const int nb = 128;
int mb = 32;
    for(int j = 0, tid = 0; j < M; j+=mb, tid++){
 for(int i = 0; i < N; i+=nb){
                     for(int ii = i; ii < mymin(i+nb, N); ii++){
                         for(int jj = j; jj < mymin(j+mb, M); jj++){
                             B[idx] = A[ii*M+jj];
                             idx++;
                         }
                     }
                 }
}
}
inline void pack_qw(int* A, int* B){
  pack_qw_inner(A, B, 65);
}
inline void pack_output(float* A, float* B){
  // copy the full matrix A in blocked format into B
  uint64_t idx = 0;
  const int N = 1;
  const int M = 4096;
  const int nb = 1;
  const int mb = 32;
  for(int i = 0; i < N; i+=nb){ 
             for(int j = 0; j < M; j+=mb){
                 for(int ii = i; ii < mymin(i+nb, N); ii++){
                     for(int jj = j; jj < mymin(j+mb, M); jj++){
                         B[idx] = A[ii*M+jj];
                         idx++;
                     }
                 }
             }
         }
     }
void print_parameters(){
std::ofstream outfile;
outfile.open(".build/qigen/tmp.csv", std::ios_base::app);
outfile << 4 << "," << 1 << "," << 16 << "," << 16 << "," << 1 << "," << 16  << "," << -1 << ",";
}
