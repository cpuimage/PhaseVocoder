#include <stdio.h>
#include <stdlib.h>
#include "phase_vocoder.h"
#include "timing.h"
// https://github.com/mackron/dr_libs/blob/master/dr_wav.h
#define DR_WAV_IMPLEMENTATION

#include "dr_wav.h"

#define DEBUG 0

void wavWrite_f32(char *filename, float *buffer, size_t sampleRate, size_t totalSampleCount) {
    drwav_data_format format;
    format.container = drwav_container_riff;     // <-- drwav_container_riff = normal WAV files, drwav_container_w64 = Sony Wave64.
    format.channels = 1;
    format.sampleRate = (drwav_uint32) sampleRate;
    format.bitsPerSample = sizeof(float) * 8;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;

    drwav *pWav = drwav_open_file_write(filename, &format);
    if (pWav) {
        drwav_uint64 samplesWritten = drwav_write(pWav, totalSampleCount, buffer);
        drwav_uninit(pWav);
        if (samplesWritten != totalSampleCount) {
            fprintf(stderr, "ERROR\n");
            exit(1);
        }
    }
}

float *wavRead_f32(char *filename, uint32_t *sampleRate, uint64_t *totalSampleCount) {
    unsigned int channels;
    float *buffer = drwav_open_and_read_file_f32(filename, &channels, sampleRate,
                                                 totalSampleCount);
    if (buffer == 0) {
        fprintf(stderr, "ERROR\n");
        exit(1);
    }
    if (channels != 1) {
        drwav_free(buffer);
        buffer = 0;
        *sampleRate = 0;
        *totalSampleCount = 0;
    }
    return buffer;
}

void splitpath(const char *path, char *drv, char *dir, char *name, char *ext) {
    const char *end;
    const char *p;
    const char *s;
    if (path[0] && path[1] == ':') {
        if (drv) {
            *drv++ = *path++;
            *drv++ = *path++;
            *drv = '\0';
        }
    } else if (drv)
        *drv = '\0';
    for (end = path; *end && *end != ':';)
        end++;
    for (p = end; p > path && *--p != '\\' && *p != '/';)
        if (*p == '.') {
            end = p;
            break;
        }
    if (ext)
        for (s = end; (*ext = *s++);)
            ext++;
    for (p = end; p > path;)
        if (*--p == '\\' || *p == '/') {
            p++;
            break;
        }
    if (name) {
        for (s = p; s < end;)
            *name++ = *s++;
        *name = '\0';
    }
    if (dir) {
        for (s = path; s < p;)
            *dir++ = *s++;
        *dir = '\0';
    }
}


int main(int argc, char *argv[]) {
    printf("a phase vocoder example\n");
    printf("blog:http://cpuimage.cnblogs.com/\n");
    if (argc > 1) {
        char *in_file = argv[1];
        char drive[3];
        char dir[256];
        char fname[256];
        char ext[256];
        char out_file[1024];
        splitpath(in_file, drive, dir, fname, ext);
        sprintf(out_file, "%s%s%s_out%s", drive, dir, fname, ext);
        uint32_t sampleRate = 0;
        uint64_t nSampleCount = 0;
        float stretchFactor = 0.9;
        if (argc > 2)
            stretchFactor = (float) atof(argv[2]);
        if (stretchFactor > 1.0f)
            stretchFactor = 1;
        float *data_in = wavRead_f32(in_file, &sampleRate, &nSampleCount);
        float *data_out = (float *) calloc((nSampleCount / stretchFactor) + 1, sizeof(float));
        int64_t nSampleOut = 0;
        int64_t input_left = nSampleCount;

        if (data_in != NULL && data_out != NULL) {
            double startTime = now();
            int window_size = sampleRate / 20;
            float *input = data_in;
            float *output = data_out;
            //Get the initial window for everyone who needs it
            float window[window_size];
            for (int i = 0; i < window_size; ++i) {
                window[i] = input[i];
            }
            input += window_size;
            input_left -= window_size;

            //Set up the forward and backward states
            stft_forward_state *forward = stft_forward_init(window_size, window);
            stft_backward_state *backward = stft_backward_init(window_size, window);

            //Transform the first window for the stretch state
            for (int i = 0; i < window_size; ++i) {
                window[i] = input[i];
            }
            input += window_size;
            input_left -= window_size;
            cartesian *frame = stft_forward_feed(forward, window);

            //Initialize the stretch state
            stft_stretch_state *stretch = stft_stretch_init(window_size, stretchFactor, frame);

            //Stretch and write along the entire file.
            while (input_left > window_size) {
                //Get the next window
                for (int i = 0; i < window_size; ++i) {
                    window[i] = input[i];
                }
                input += window_size;
                input_left -= window_size;

                //Feed it to the forward stft transformer
                frame = stft_forward_feed(forward, window);

                //Stretch as much as we can
                CartesianListNode *list = stft_stretch_feed(stretch, frame);

                //Backward-transform everything we got from our stretcher and write it.
                while (list != NULL) {
                    float *back_transformed = stft_backward_feed(backward, list->value);
                    free(list->value);
                    for (int i = 0; i < window_size; ++i) {
                        output[i] = back_transformed[i];
#if DEBUG
                        fprintf(stderr, "%f\n", back_transformed[i]);
#endif
                    }
                    output += window_size;
                    nSampleOut += window_size;
                    free(back_transformed);
                    list = list->next;
#if DEBUG
                    fputs("WINDOW BREAK\n", stderr);
#endif
                }
                free(frame);
            }
            stft_forward_free(forward);
            stft_backward_free(backward);
            stft_stretch_free(stretch);
            double time_interval = calcElapsed(startTime, now());

            wavWrite_f32(out_file, data_out, sampleRate, (uint32_t) nSampleOut);
            free(data_out);
            free(data_in);
            printf("time interval: %d ms\n ", (int) (time_interval * 1000));
            printf("press any key to exit.\n");
            getchar();
        }
    } else {
        puts("Usage: phase_vocoder INPUT_FILE FACTOR[0.0,1.0]");
        exit(1);
    }
}
