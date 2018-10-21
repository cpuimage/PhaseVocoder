#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "phase_vocoder.h"

#define STB_FFT_IMPLEMENTAION

#include "stb_fft.h"

#ifndef PI
#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062f
#endif

/*
  The scaling factor at point (pos) in window (window) for a hanning window.
*/
float hanning_window(int pos, int window) {
    return sinf(PI * pos / window);
}

/*
  Conversion functions between different complex number representation
*/
cartesian package(cmplx c) {
    cartesian r;
    r.real = (c.real != NAN && c.real != INFINITY ? c.real : 0);
    r.imag = (c.imag != NAN && c.imag != INFINITY ? c.imag : 0);
    return r;
}

polar polarize(cartesian c) {
    polar r;
    r.magnitude = sqrtf(c.real * c.real + c.imag * c.imag);
    r.phase = atan2f(c.imag, c.real);
    return r;
}

cartesian unpolarize(polar p) {
    cartesian r;
    r.real = p.magnitude * cosf(p.phase);
    r.imag = p.magnitude * sinf(p.phase);
    return r;
}

/*
  Initialize an stft_forward_state
*/
stft_forward_state *stft_forward_init(int window_size, float *initial_window) {
    stft_forward_state *state = malloc(sizeof(stft_forward_state));

    //Record the window size and calculate the output size
    state->window_size = window_size;

    //Allocate the input and output
    state->input = calloc(window_size * 2, sizeof(float));
    state->output = calloc(window_size + 1, sizeof(cmplx));

    //Create the fft plan
    int plan_size = stb_fft_real_plan_dft_1d(window_size * 2, NULL);
    if (plan_size > 0)
        state->plan = (stb_fft_real_plan *) calloc(plan_size, 1);
    if (state->plan != NULL) {
        stb_fft_real_plan_dft_1d(window_size * 2, state->plan);
    }
    //Set the initial window
    state->window = malloc(window_size * sizeof(float));
    for (int i = 0; i < window_size; ++i)
        state->window[i] = initial_window[i];

    //Return the state we just initialized.
    return state;
}

/*
  Feed an new window to an stft_forward_state
*/
cartesian *stft_forward_feed(stft_forward_state *state, float *next) {
    cartesian *result = (cartesian *) malloc((state->window_size + 1) * 2 * sizeof(cartesian));

    //Remember a couple common values for this execution
    int n = state->window_size;

    //Assemble the window we are going to transform.
    for (int i = 0; i < n; ++i)
        state->input[i] = state->window[i] * hanning_window(i, n * 2);
    for (int i = 0; i < n; ++i)
        state->input[i + n] = next[i] * hanning_window(i + n, n * 2);

    //Update our cached window
    for (int i = 0; i < n; ++i)
        state->window[i] = next[i];

    //Transform it.
    stb_fft_r2c_exec(state->plan, state->input, state->output);

    //Put it in the heap
    for (int i = 0; i <= n; ++i)
        result[i] = package(state->output[i]);

    //Return a pointer to it.
    return result;
}

/*
  Destroy (free) an stft_forward_state
*/
void stft_forward_free(stft_forward_state *state) {
    free(state->plan);
    free(state->input);
    free(state->output);
    free(state->window);
    free(state);
}

stft_backward_state *stft_backward_init(int window_size, float *initial_window) {
    stft_backward_state *state = malloc(sizeof(stft_backward_state));

    //Remember the given window size
    state->window_size = window_size;

    //Allocate space for fft input and output
    state->input = calloc(window_size + 1, sizeof(cmplx));
    state->output = calloc(window_size * 2, sizeof(float));

    //Create an fft plan
    int plan_size = stb_fft_real_plan_dft_1d(window_size * 2, NULL);
    if (plan_size > 0)
        state->plan = (stb_fft_real_plan *) calloc(plan_size, 1);
    if (state->plan != NULL) {
        stb_fft_real_plan_dft_1d(window_size * 2, state->plan);
    }
    //Set the initial window
    state->window = malloc(window_size * sizeof(float));
    for (int i = 0; i < window_size; ++i)
        state->window[i] = initial_window[i] * window_size * 2 *
                           hanning_window(i + window_size, window_size * 2);

    //Return the heap address to the state.
    return state;
}

/*
  Feed a new frame to an stft_backward_state
*/
float *stft_backward_feed(stft_backward_state *state, cartesian *next) {
    float *result = (float *) malloc(state->window_size * sizeof(float));

    //Remember a couple common values for this execution
    int n = state->window_size;

    //Transform this window.
    for (int i = 0; i <= n; ++i) {
        state->input[i] = next[i];
    }

    stb_fft_c2r_exec(state->plan, state->input, state->output);

    //"Stitch" it to the last one, storing in the heap and normalizing
    for (int i = 0; i < n; ++i) {
        result[i] = (state->window[i] * hanning_window(i + n, n * 2) + state->output[i] * hanning_window(i, n * 2)) /
                    (n * 2);
    }

    //Update our state window
    for (int i = 0; i < n; ++i) state->window[i] = state->output[i + n];

    //Return a pointer to the data.
    return result;
}

/*
  Destroy (free) an stft_backward_state
*/
void stft_backward_free(stft_backward_state *state) {
    free(state->plan);
    free(state->input);
    free(state->output);
    free(state->window);
    free(state);
}

/*
  Initialize an stft_stretch_state
*/
stft_stretch_state *stft_stretch_init(int window_size, float factor, cartesian *first_frame) {
    //Allocate the state
    stft_stretch_state *state = malloc(sizeof(stft_stretch_state));

    //Remember some arguments for convenience
    state->window_size = window_size;
    state->factor = factor;

    //Initialize the first frame
    state->last_frame = malloc((window_size + 1) * sizeof(polar));
    for (int i = 0; i <= window_size; ++i)
        state->last_frame[i] = polarize(first_frame[i]);

    //Allocate space for the first frame's phases and initialize them properly
    state->phases = malloc((window_size + 1) * sizeof(float));
    for (int i = 0; i <= window_size; ++i)
        state->phases[i] = state->last_frame[i].phase;

    //Return the state we just constructed.
    return state;
}

/*
  Perform all possible time stretching now that we know stft frame *next, from stft_stretch_state *state.
*/
CartesianListNode *stft_stretch_feed(stft_stretch_state *state, cartesian *next) {
    CartesianListNode *result = NULL;

    //Get polar coordinates for all of the cartesian coordinates we are given
    polar polar_next[state->window_size + 1];
    for (int i = 0; i < state->window_size; ++i) polar_next[i] = polarize(next[i]);

    //Interpolate all possible new frames given this new one
    for (; state->position < 1; state->position += state->factor) {
        //Allocate memory for the new frame
        cartesian *frame = malloc((state->window_size + 1) * sizeof(cartesian));

        //Interpolate
        for (int i = 0; i <= state->window_size; ++i) {
            polar polar_interpolated;

            //Interpolate magnitude
            polar_interpolated.magnitude =
                    state->last_frame[i].magnitude * (1 - state->position) + polar_next[i].magnitude * state->position;

            //Interpolate phases
            polar_interpolated.phase = state->phases[i];
            state->phases[i] = state->phases[i] + PI * i +
                               (polar_next[i].phase - state->last_frame[i].phase - PI * i) * state->factor;

            //Append the interpolated polar, in cartesian form, to our heap data
            frame[i] = unpolarize(polar_interpolated);
        }

        //Append the new frame to the existent list
        CartesianListNode *new_node = malloc(sizeof(CartesianListNode));
        new_node->next = result;
        new_node->value = frame;

        //Update result to be the true head of the list
        result = new_node;
    }

    //Update our state
    for (int i = 0; i < state->window_size; ++i)
        state->last_frame[i] = polar_next[i];

    //Bring state->position back to within [0, 1]
    state->position -= (int) state->position;

    return result;
}

void stft_stretch_free(stft_stretch_state *state) {
    free(state->last_frame);
    free(state);
}
