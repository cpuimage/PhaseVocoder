#ifndef PHASE_VOCODER
#define PHASE_VOCODER

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stb_fft.h"
#include <sys/types.h>
#include <math.h>

#ifndef cartesian
#define cartesian cmplx
#endif

typedef struct {
    float magnitude;
    float phase;
} polar;

/*
  Feed-type function states
*/
typedef struct {
    int window_size;
    stb_fft_real_plan *plan;
    float *input;
    cmplx *output;
    float *window;
} stft_forward_state;

typedef struct {
    int window_size;
    stb_fft_real_plan *plan;
    cmplx *input;
    float *output;
    float *window;
} stft_backward_state;

typedef struct {
    int window_size;
    float factor, position, *phases;
    polar *last_frame;
} stft_stretch_state;

/*
  Basic linked list with cartesian values
*/
struct CartesianListNode;
typedef struct CartesianListNode CartesianListNode;

struct CartesianListNode {
    cartesian *value;
    CartesianListNode *next;
};

/*
  Windowing function
*/
float hanning_window(int, int);

/*
  Conversion functions between different complex number types
*/
cartesian package(cmplx);

polar polarize(cartesian);

cartesian unpolarize(polar);

/*
  Arap a phase to within PI and an -PI
*/
float phase_modulo(float);

/*
  Forward short-time fourier transform functions
*/
stft_forward_state *stft_forward_init(int /* Window size */, float * /* Initial window */);

cartesian *stft_forward_feed(stft_forward_state *, float *);

void stft_forward_free(stft_forward_state *);

/*
  Backward short-time fourier transform functions
*/
stft_backward_state *stft_backward_init(int /* Window size */, float * /* Initial window */);

float *stft_backward_feed(stft_backward_state *, cartesian *);

void stft_backward_free(stft_backward_state *);

/*
  Functions to time-stretch a short-time fourier trasnform
*/
stft_stretch_state *stft_stretch_init(int /* Window size */, float /* Stretch factor */, cartesian * /* First frame */);

CartesianListNode *stft_stretch_feed(stft_stretch_state *, cartesian *);

void stft_stretch_free(stft_stretch_state *);

#endif
